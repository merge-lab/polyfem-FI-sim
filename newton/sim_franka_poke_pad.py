import argparse
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd     # Used for mesh pre-processing

import newton
import newton.ik as ik
import warp as wp
from newton._src.solvers.vbd.particle_vbd_kernels import evaluate_volumetric_neo_hookean_force_and_hessian
from newton._src.solvers.vbd.rigid_vbd_kernels import _eval_body_particle_contact
from newton._src.utils.selection import match_labels
from pxr import Usd

import newton_builders
import utils_FI

now = datetime.now()
now_str = now.strftime("%d-%m-%Y_%H:%M:%S")
@dataclass
class PokeParams:
    ### Legacy motion parameters (unused in franka script, kept for reference)
    z_zero            = -0.0855
    compression_rate  = 0.02
    compression_depth = 0.002
    t_start_wait      = 0.1
    t_stop_wait       = 0.05

    ### Multi-poke pad geometry
    n_pokes       = 20
    pad_center_x  = 0.0
    pad_center_y  = -0.5
    pad_half_x    = 0.075 / 2    # ±0.0375 m
    pad_half_y    = 0.065 / 2    # ±0.0325 m
    z_pad_surface = 0.014        # top surface height [m]
    z_home        = 0.10         # robot home height [m]
    z_approach    = 0.024        # z_pad_surface + 0.010 [m]
    z_poke        = 0.010        # z_pad_surface - 0.004 [m]

    ### Timing per poke phase [s]
    t_approach = 1.0
    t_descend  = 4.0
    t_hold     = 2.0
    t_ascend   = 4.0

class PokeState(Enum):
    INIT     = 0   # brief settle before first poke
    APPROACH = 1   # move to z_approach above current poke XY
    DESCEND  = 2   # slow descent from z_approach to z_poke
    HOLD     = 3   # dwell at z_poke
    ASCEND   = 4   # rise back to z_approach
    DONE     = 5   # all pokes complete

@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    joint_q[0, idx0] = finger_pos[0]
    joint_q[0, idx1] = finger_pos[0]


@wp.kernel
def compute_joint_qd(
    target_q: wp.array[float],
    current_q: wp.array[float],
    out_qd: wp.array[float],
    inv_frame_dt: float,
):
    i = wp.tid()
    out_qd[i] = (target_q[i] - current_q[i]) * inv_frame_dt


@wp.kernel(enable_backward=False)
def accumulate_soft_contact_body_forces(
    dt: float,
    friction_epsilon: float,
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    soft_contact_count: wp.array[int],
    soft_contact_max: int,
    soft_contact_particle: wp.array[int],
    soft_contact_shape: wp.array[int],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    contact_penalty_k: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    body_to_row: wp.array[int],
    # output
    body_contact_force: wp.array[wp.vec3],
):
    """Accumulate the reaction force that particle-body (soft) contacts exert on selected rigid bodies.

    Replicates the penalty force law the VBD particle solve applies to each particle
    (newton _eval_body_particle_contact) and atomically adds the negated force to the
    row assigned to the contacted body. Parallelizes over soft contacts.
    """
    t_id = wp.tid()
    if t_id >= min(soft_contact_max, soft_contact_count[0]):
        return

    shape_idx = soft_contact_shape[t_id]
    body_idx = shape_body[shape_idx]
    if body_idx < 0:
        return
    row = body_to_row[body_idx]
    if row < 0:
        return

    particle_idx = soft_contact_particle[t_id]
    f_particle, _hessian = _eval_body_particle_contact(
        particle_idx,
        particle_q[particle_idx],
        particle_q_prev[particle_idx],
        t_id,
        contact_penalty_k[t_id],
        contact_material_kd[t_id],
        contact_material_mu[t_id],
        friction_epsilon,
        particle_radius,
        shape_body,
        body_q,
        body_q_prev,
        body_qd,
        body_com,
        soft_contact_shape,
        soft_contact_body_pos,
        soft_contact_body_vel,
        soft_contact_normal,
        dt,
    )
    wp.atomic_add(body_contact_force, row, -f_particle)


@wp.kernel(enable_backward=False)
def flag_contacted_particles(
    shape_body: wp.array[int],
    soft_contact_count: wp.array[int],
    soft_contact_max: int,
    soft_contact_particle: wp.array[int],
    soft_contact_shape: wp.array[int],
    body_to_row: wp.array[int],
    # output
    particle_contact_flag: wp.array[int],
):
    """Mark particles that have an active soft contact against a sensed rigid body."""
    t_id = wp.tid()
    if t_id >= min(soft_contact_max, soft_contact_count[0]):
        return

    body_idx = shape_body[soft_contact_shape[t_id]]
    if body_idx < 0:
        return
    if body_to_row[body_idx] < 0:
        return
    particle_contact_flag[soft_contact_particle[t_id]] = 1


@wp.kernel(enable_backward=False)
def accumulate_elastic_vertex_forces(
    dt: float,
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    particle_contact_flag: wp.array[int],
    # output
    elastic_force: wp.array[wp.vec3],
):
    """Accumulate the internal FEM (elastic + damping) force at contacted vertices.

    Nodal residual method: at equilibrium of a contacted vertex, internal force +
    contact force + gravity = 0, so summing the internal force over the contact
    patch yields the reaction force on the contacting rigid body (patch gravity is
    negligible). Parallelizes over tets; each tet adds its contribution to each of
    its flagged vertices.
    """
    tet_id = wp.tid()

    f_tet = wp.vec3(0.0)
    for v_order in range(4):
        vid = tet_indices[tet_id, v_order]
        if particle_contact_flag[vid] == 1:
            f, _hessian = evaluate_volumetric_neo_hookean_force_and_hessian(
                tet_id,
                v_order,
                particle_q_prev,
                particle_q,
                tet_indices,
                tet_poses[tet_id],
                tet_materials[tet_id, 0],
                tet_materials[tet_id, 1],
                tet_materials[tet_id, 2],
                dt,
            )
            f_tet += f

    if wp.length_sq(f_tet) > 0.0:
        wp.atomic_add(elastic_force, 0, f_tet)


class ThighpadPokeTest:
    def __init__(self, viewer, args, verbose=False):
        self.sim_start_time = 0.0
        self.sim_time = 0.0
        self.sim_params = newton_builders.SimParams()
        self.sim_params.particle_radius = 0.00005
        self.sim_params.sim_substeps = 5
        self.sim_params.iterations = 10
        self.sim_params.fps = 15
        self.poke_params = PokeParams()
        self._generate_poke_points()

        # State machine init
        self.poke_state      = PokeState.INIT
        self.t_state_start   = 0.0
        self.pos_state_start = np.array([-0.005, -0.5, self.poke_params.z_home], dtype=np.float32)
        self.pos_state_end   = np.array([-0.005, -0.5, self.poke_params.z_home], dtype=np.float32)

        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -self.sim_params.g), dtype=wp.vec3)

        self.args = args
        self.verbose = verbose

        # Create scene
        self.init_q = -0.0855 + 0.0005
        self.current_q = self.init_q
        init_qs = self.init_q * wp.ones(9, dtype=wp.float32)
        self.model = self.init_models()

        self.add_sensors(self.model) # Add sensors now that the model has been finalized

        # Initialize camera
        self.viewer = viewer
        self.viewer.set_model(self.model)
        if not self.args["headless"]:
            self.viewer.set_camera(wp.vec3([0.15, -0.15, 0.1]), -3.7, -118.0)
            self.viewer._cam_speed = 0.15

        ### Initialize both solvers
        self.solver_vbd = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.sim_params.iterations,
            integrate_with_external_rigid_solver=True,
            # particle_enable_self_contact=True,
            particle_enable_self_contact=True,
            # particle_enable_tile_solve=True,
            particle_self_contact_radius=self.sim_params.particle_self_contact_radius,
            particle_self_contact_margin=self.sim_params.particle_self_contact_margin,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=self.sim_params.rigid_contact_k_start,
            rigid_avbd_beta=self.sim_params.rigid_avbd_beta,
        )


        # Preallocate variables for trajectory, control, and contacts
        self.state_now = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()
        self.control.joint_target_pos = init_qs     # Initialize configuration of poke fixture
        self.contacts = self.model.contacts()

        # Initialize rigid body solver
        # self.solver_rigid = newton.solvers.SolverMuJoCo(self.model)
        self.solver_rigid = newton.solvers.SolverFeatherstone(
            self.model,
            update_mass_matrix_interval=self.sim_params.sim_substeps
        )

        # Initialize collision pipeline with non-default soft contact margin
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.sim_params.particle_self_contact_margin
        )

        # Initialize IK
        self.init_ik()
        self.solver_ik = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives = [self.pos_obj, self.rot_obj, self.obj_joint_limits]
        )

        self.graph = None
        self.capture()

        ### Initialize gui and logging
        if not self.args["headless"]:
            self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="side")
        
        # Initialize logging
        self.vol_initial = -1
        self.log_sim_times: list[float] = []
        self.log_volumes: list[float] = []
        self.log_pressures: list[float] = []
        self.log_pokes:     list[int] = []
        self.log_forces:    list[float] = []
        self.log_forces_penalty: list[float] = []
        self.log_forces_elastic: list[float] = []

        self.terminated = False


    def init_models(self):
        """
            Create the simulation scene
        """
        # Initialize the scene
        self.scene = newton.ModelBuilder()
        self.scene.default_particle_radius = self.sim_params.particle_radius

        ## ======= Add thigh pad =============
        # self.create_thighpad(self.scene)
        self.pad_start_particle_idx = len(self.scene.particle_q)
        self._fixed_particle_ids, dict_surf_select = newton_builders.create_thighpad(
            self.scene,
            pos         = wp.vec3(0.0, -0.5, 0.0),
            rot         = wp.quat_identity(),
            rho         = self.sim_params.material_rho,
            k_mu        = self.sim_params.material_k_mu,
            k_lambda    = self.sim_params.material_k_lambda,
            k_damp      = self.sim_params.material_k_damp,
        )
        id_channel = 2
        self.ids_channel = dict_surf_select[id_channel]

        ## ======= Add Franka arm =============
        self.add_articulated_franka(self.scene)
        self.set_franka_targets()

        ## ======== Add ground plane =======
        # TODO - understand the meaning of these numbers by looking at VBD docs
        # builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu))
        self.scene.add_ground_plane()

        ## ========= Make world replicas
        replicator = newton.ModelBuilder()
        n_worlds = 1
        replicator.replicate(self.scene, n_worlds, spacing=(0.2, 0.2, 0.0))

 
        ## ======== Finalize and export the model =======
        # self.scene.color()
        # model = self.scene.finalize()
        replicator.color()
        model = replicator.finalize()

        newton.eval_fk(model, model.joint_q, model.joint_qd, model)

        model.soft_contact_ke = self.sim_params.soft_contact_ke
        model.soft_contact_kd = self.sim_params.soft_contact_kd
        model.soft_contact_mu = self.sim_params.soft_contact_mu

        self.setup_debug_viz(model)

        # Diagnostic: print shape flags to check COLLIDE_PARTICLES
        from newton._src.geometry.flags import ShapeFlags
        shape_flags_np = model.shape_flags.numpy()
        shape_labels = model.shape_label if hasattr(model, 'shape_label') else [str(i) for i in range(len(shape_flags_np))]
        print(f"\n=== Shape flags diagnostic ({len(shape_flags_np)} shapes) ===")
        for i, (flags, label) in enumerate(zip(shape_flags_np, shape_labels)):
            visible = bool(flags & ShapeFlags.VISIBLE)
            collide_shapes = bool(flags & ShapeFlags.COLLIDE_SHAPES)
            collide_particles = bool(flags & ShapeFlags.COLLIDE_PARTICLES)
            print(f"  Shape {i} [{label}]: VISIBLE={visible}, COLLIDE_SHAPES={collide_shapes}, COLLIDE_PARTICLES={collide_particles}")
        print("=== End shape flags ===\n")

        return model

    def add_articulated_franka(self, scene):
        asset_path = Path("../Assets/franka_emika_panda/urdf/fr3_large_sphere.urdf")

        scene.add_urdf(
            asset_path,
            xform=wp.transform((-0.5, -0.5, -0.1), wp.quat_identity()),
            floating=False,
            scale=1.0,  # URDF is in meters
            enable_self_collisions=False,
            # collapse_fixed_joints=True,
            collapse_fixed_joints=False,
            force_show_colliders=False,
        )
        scene.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
    
    def add_sensors(self, model):
        # newton.sensors.SensorContact only reads rigid-rigid contacts, and SolverVBD never
        # populates contacts.force, so the indentor <-> pad contact force is computed manually
        # from the soft (particle-body) contact data in _update_indentor_force().
        indentor_bodies = match_labels(model.body_label, "*indentor")
        assert len(indentor_bodies) == 1, f"Expected exactly one indentor body, matched {indentor_bodies}"
        self.indentor_body_idx = indentor_bodies[0]

        # Create self.sensor_body_to_row, which is a mask vector that is -1 for all body indices, except
        # 0 for the indentor body index.
        body_to_row = np.full(model.body_count, -1, dtype=np.int32)
        body_to_row[self.indentor_body_idx] = 0
        self.sensor_body_to_row = wp.array(body_to_row, dtype=int)
        self.indentor_contact_force = wp.zeros(1, dtype=wp.vec3)

        # Elastic (nodal internal force) sensor buffers
        self.particle_contact_flag = wp.zeros(model.particle_count, dtype=int)
        self.indentor_elastic_force = wp.zeros(1, dtype=wp.vec3)

    def _update_indentor_force(self):
        """Recompute the contact force on the indentor from the last substep.

        Two independent readings are computed:
        - indentor_contact_force: reaction of the VBD contact-resolution penalty.
          With integrate_with_external_rigid_solver=True the VBD coupling is one-way (the pad
          feels the indentor but no rigid reaction force is ever computed), so we re-evaluate
          the solver's own penalty force at the converged state and sum the reaction.
        - indentor_elastic_force: internal FEM (elastic + damping) force summed over the
          contacted vertices (nodal residual method) — the reaction contributed by the
          deformed pad material.

        After the substep state swap, state_now holds post-substep poses and state_next the
        pre-substep poses, matching the (body_q, body_q_prev) pair the particle solve used.
        """
        self.indentor_contact_force.zero_()
        wp.launch(
            accumulate_soft_contact_body_forces,
            dim=self.contacts.soft_contact_max,
            inputs=[
                self.sim_params.sim_dt,
                self.solver_vbd.friction_epsilon,
                self.state_now.particle_q,
                self.solver_vbd.particle_q_prev,
                self.model.particle_radius,
                self.model.shape_body,
                self.state_now.body_q,
                self.state_next.body_q,
                self.state_now.body_qd,
                self.model.body_com,
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_max,
                self.contacts.soft_contact_particle,
                self.contacts.soft_contact_shape,
                self.contacts.soft_contact_body_pos,
                self.contacts.soft_contact_body_vel,
                self.contacts.soft_contact_normal,
                self.solver_vbd.body_particle_contact_penalty_k,
                self.solver_vbd.body_particle_contact_material_kd,
                self.solver_vbd.body_particle_contact_material_mu,
                self.sensor_body_to_row,
            ],
            outputs=[self.indentor_contact_force],
        )

        # Elastic reaction: flag the contacted pad vertices, then sum their internal FEM forces
        self.particle_contact_flag.zero_()
        self.indentor_elastic_force.zero_()
        wp.launch(
            flag_contacted_particles,
            dim=self.contacts.soft_contact_max,
            inputs=[
                self.model.shape_body,
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_max,
                self.contacts.soft_contact_particle,
                self.contacts.soft_contact_shape,
                self.sensor_body_to_row,
            ],
            outputs=[self.particle_contact_flag],
        )
        wp.launch(
            accumulate_elastic_vertex_forces,
            dim=self.model.tet_indices.shape[0],
            inputs=[
                self.sim_params.sim_dt,
                self.state_now.particle_q,
                self.solver_vbd.particle_q_prev,
                self.model.tet_indices,
                self.model.tet_poses,
                self.model.tet_materials,
                self.particle_contact_flag,
            ],
            outputs=[self.indentor_elastic_force],
        )

        print(f"Soft countact count: {self.contacts.soft_contact_count.numpy()[0]}/{self.contacts.soft_contact_max}")

    def set_franka_targets(self):
        gripper_open = 1.0
        gripper_close = 0.5

        # Keyframe sequence: approach, descend, pinch, lift, hold, place, release, retract
        # [duration, px, py, pz, qx, qy, qz, qw, gripper_activation] (positions in meters)
        self.robot_key_poses = np.array(
            [
                # Initial pose
                [1.0, -0.005, -0.5, 0.1, 1, 0.0, 0.0, 0.0],
                # Descend to right abovbe pad
                [1.0, -0.005, -0.5, 0.02, 1, 0.0, 0.0, 0.0],
                # Press
                [4.0, -0.005, -0.5, 0.01, 1, 0.0, 0.0, 0.0],
                # Lift back to just above pad
                [4.0, -0.005, -0.5, 0.02, 1, 0.0, 0.0, 0.0],
                # Return to original configuration and end
                [2.0, -0.005, -0.5, 0.35, 1, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]

        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        print(f"Robot key poses times: {self.robot_key_poses_time}")

    def init_ik(self):
        """
        IK setup (single problem, single EE)
        Taken from newton ik_franka example
        """
        # TODO - abstract this function and move to utils.
        # self.ee_index = 10
        # body_q_np = self.state_now.body_q.numpy()
        # self.ee_tf = wp.transform(*body_q_np[self.ee_index])
        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model.state())

        self.n_coords = self.model.joint_coord_count
        self.n_dofs = self.model.joint_dof_count
        self.ik_joint_q =  wp.array(self.model.joint_q, shape=(1, self.n_coords))
        
        self.ee_idx = self.scene.body_count - 1
        # Allocate buffers for values (need to be wp.arrays for graph capture)
        self.target_joint_q = wp.zeros(self.n_coords, dtype=float) # Buffer to store ik result (joint q targets)
        self.target_joint_qd = wp.empty_like(self.state_now.joint_qd)

        init_target_pos = wp.vec3(*self.targets[0][:3].tolist())
        init_target_rot = wp.vec4(*self.targets[0][3:7].tolist())

        # Position objective
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_idx,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([init_target_pos], dtype=wp.vec3),
        )

        # Rotation objective
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_idx,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([init_target_rot], dtype=wp.vec4),
        )

        # Joint limit objective
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )

        # Variables the solver will update
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24


    def franka_set_ik_target(self):
        """Interpolate keyframes and update IK target arrays (CPU, called before graph launch)."""
        if self.sim_time >= self.robot_key_poses_time[-1]:
            return

        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)

        # Interpolate between previous and current keyframe target
        t_start = self.robot_key_poses_time[current_interval - 1] if current_interval > 0 else 0.0
        t_end = self.robot_key_poses_time[current_interval]
        alpha = float(np.clip((self.sim_time - t_start) / (t_end - t_start), 0.0, 1.0))

        self.target_cur = self.targets[current_interval]
        target_prev = self.targets[current_interval - 1] if current_interval > 0 else self.target_cur
        target_interp = (1.0 - alpha) * target_prev + alpha * self.target_cur

        # Update IK target arrays on GPU (read by IK solver inside captured graph)
        self.pos_obj.set_target_position(0, wp.vec3(*target_interp[:3].tolist()))
        self.rot_obj.set_target_rotation(0, wp.vec4(*target_interp[3:7].tolist()))

    def franka_solve_joint_qd(self):
        # IK solve once per frame (GPU, captured in graph)
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)

        # Copy IK result to target buffer (2D -> 1D, contiguous memory)
        wp.copy(self.target_joint_q, self.ik_joint_q, dest_offset=0, src_offset=0, count=self.n_coords)

        # Compute joint velocity: qd = (target - current) / frame_dt
        wp.launch(
            compute_joint_qd,
            dim=self.n_dofs,
            inputs=[self.target_joint_q, self.state_now.joint_q, self.target_joint_qd, 1.0 / self.sim_params.frame_dt],
        )

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def setup_debug_viz(self, model):
        # Build per-particle debug color array: blue for fixed, gray for free
        debug_radius = self.sim_params.particle_radius
        particle_color_default = [0.6, 0.6, 0.6] # Gray
        particle_color_fixed = [0.0, 0.0, 1.0] # Blue

        colors_np = np.tile(particle_color_default, (model.particle_count, 1)).astype(np.float32)
        for idx in self._fixed_particle_ids:
            colors_np[idx] = particle_color_fixed
        self.particle_debug_colors = wp.array(colors_np, dtype=wp.vec3)
        self.particle_debug_radii = wp.full(model.particle_count, debug_radius, dtype=wp.float32)


    def _generate_poke_points(self):
        pp = self.poke_params
        rng = np.random.default_rng(seed=42)
        poke_xs = rng.uniform(pp.pad_center_x - pp.pad_half_x, pp.pad_center_x + pp.pad_half_x, pp.n_pokes)
        poke_ys = rng.uniform(pp.pad_center_y - pp.pad_half_y, pp.pad_center_y + pp.pad_half_y, pp.n_pokes)

        z_poke_surface_top = 0.014
        self.poke_points = np.column_stack(
            [poke_xs, poke_ys, np.full(pp.n_pokes, z_poke_surface_top)]
        ).astype(np.float32)    # shape (n_pokes, 3)
        self.poke_done      = np.zeros(pp.n_pokes, dtype=bool)
        self.i_current_poke = 0

    def _control_poke(self):
        if self.poke_state == PokeState.DONE:
            return

        t_elapsed = self.sim_time - self.t_state_start

        if self.poke_state == PokeState.INIT:
            if t_elapsed >= 1.0:
                self._transition_poke_state(PokeState.APPROACH)
            return

        pp = self.poke_params
        durations = {
            PokeState.APPROACH: pp.t_approach,
            PokeState.DESCEND:  pp.t_descend,
            PokeState.HOLD:     pp.t_hold,
            PokeState.ASCEND:   pp.t_ascend,
        }
        dur   = durations[self.poke_state]
        alpha = float(np.clip(t_elapsed / dur, 0.0, 1.0))
        target_pos = (1.0 - alpha) * self.pos_state_start + alpha * self.pos_state_end

        self.pos_obj.set_target_position(0, wp.vec3(*target_pos.tolist()))

        if alpha >= 1.0:
            if self.poke_state == PokeState.APPROACH:
                self._transition_poke_state(PokeState.DESCEND)
            elif self.poke_state == PokeState.DESCEND:
                self._transition_poke_state(PokeState.HOLD)
            elif self.poke_state == PokeState.HOLD:
                self._transition_poke_state(PokeState.ASCEND)
                self.poke_done[self.i_current_poke] = True
            elif self.poke_state == PokeState.ASCEND:
                print(f"Poke {self.i_current_poke} finished at sim time {self.sim_time:.3f}, elapsted wall time {time.time() - self.sim_start_time:.3f}")
                self.i_current_poke += 1
                if self.i_current_poke >= self.poke_params.n_pokes:
                    self.poke_state = PokeState.DONE
                    self.terminate()
                else:
                    self._transition_poke_state(PokeState.APPROACH)

    def _transition_poke_state(self, new_state: PokeState):
        pp  = self.poke_params
        px  = self.poke_points[self.i_current_poke]
        start = self.pos_state_end.copy()   # current position = end of previous segment

        if new_state == PokeState.APPROACH:
            end = px.copy()                                             # (poke_x, poke_y, z_approach)
        elif new_state == PokeState.DESCEND:
            end = np.array([px[0], px[1], pp.z_poke], dtype=np.float32)
        elif new_state == PokeState.HOLD:
            end = start.copy()                                          # hold in place
        elif new_state == PokeState.ASCEND:
            end = np.array([px[0], px[1], pp.z_approach], dtype=np.float32)
        else:
            end = start.copy()

        self.pos_state_start = start
        self.pos_state_end   = end
        self.t_state_start   = self.sim_time
        self.poke_state      = new_state

    def _log_states(self):
        tube_diameter = 0.0254/16       # 1/16" in meters
        tube_length = 576.22 / 1e3      # 576.22mm in meters
        tube_vol = (wp.PI * (tube_diameter/2)**2) * tube_length

        # Log the channel volume
        if self.sim_time >= self.poke_params.t_start_wait:
            # Get channel volume
            idxs = self.pad_start_particle_idx + self.ids_channel
            verts = self.state_now.particle_q[self.pad_start_particle_idx:]
            channel_volume = utils_FI.compute_volume_mesh(verts, idxs)
            self.log_volumes.append(channel_volume)

            if self.vol_initial == -1:
                self.vol_initial = channel_volume

            # Compute channel pressure using ideal gas laws
            # p1 * v1 = p2 * v2
            # So p2 = p1 * v1/v2
            p_now = 1 * (self.vol_initial + tube_vol) / (channel_volume + tube_vol)
            self.log_pressures.append(p_now)

        else:
            self.log_volumes.append(0.0)
            self.log_pressures.append(0.0)
        self.log_pokes.append(self.i_current_poke)
        self.log_sim_times.append(self.sim_time)

        # Log the force sensor reading: contact penalty reaction + elastic (material) reaction
        self._update_indentor_force()
        f_penalty = self.indentor_contact_force.numpy()[0]
        f_elastic = self.indentor_elastic_force.numpy()[0]
        force_penalty = float(np.linalg.norm(f_penalty))
        force_elastic = float(np.linalg.norm(f_elastic))
        force_total = float(np.linalg.norm(f_penalty + f_elastic))
        self.viewer.log_scalar("Indentor Contact Force", force_total)
        self.viewer.log_scalar("Indentor Force (penalty)", force_penalty)
        self.viewer.log_scalar("Indentor Force (elastic)", force_elastic)
        self.log_forces.append(force_total)
        self.log_forces_penalty.append(force_penalty)
        self.log_forces_elastic.append(force_elastic)
        
    
    def _push_targets_from_gizmos(self):
        """Read gizmo-updated transform and push into IK objectives."""
        pos = wp.transform_get_translation(self.ee_tf)
        pos = wp.vec3(pos[0], pos[1], max(pos[2], 0.11))
        self.pos_obj.set_target_position(0, pos)
        q = wp.transform_get_rotation(self.ee_tf)
        self.rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def capture(self):
        """
            Run through the simulation once, capturing the CUDA graph.
            This will make subsequent execution faster
            TODO could we get a rough sense of how much faster this makes it?
        """
        assert wp.get_device().is_cuda, "No device available or device isn't CUDA"
        
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph


    def step(self):
        self._control_poke()

        if self.graph:
            wp.capture_launch(self.graph)
            self.sim_time += self.sim_params.frame_dt
        else:
            self.simulate()
        
        self._log_states()

    
    def simulate(self):
        """
            Go through all of the sub-steps, co)rresponding to one macro step...??
            Not quite sure how this makes sense
        """

        self.franka_solve_joint_qd()

        self.solver_vbd.rebuild_bvh(self.state_now)
        for i in range(self.sim_params.sim_substeps):
            # Reset forces on the current state
            self.state_now.clear_forces()
            self.state_next.clear_forces()
            self.viewer.apply_forces(self.state_now)

            ### Mujoco sim step of rigid bodies (disable particles + gravity)
            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0

            self.state_now.joint_qd.assign(self.target_joint_qd) 
            self.solver_rigid.step(self.state_now, self.state_next, self.control, None, self.sim_params.sim_dt)

            ### Soft body step
            self.state_now.particle_f.zero_()
            self.model.particle_count = particle_count
            self.model.gravity.assign(self.gravity_earth)
            
            # Collision detection
            # self.model.collide(self.state_now, self.contacts)
            self.collision_pipeline.collide(self.state_now, self.contacts)

            # VBD sim step of soft bodies (it'll ignore rigids bc we configured it to earlier.)
            self.solver_vbd.step(self.state_now, self.state_next, self.control, self.contacts, self.sim_params.sim_dt)

            # # Swap the states (update state_now to be state_next)
            # # We can swap because state_next can really be anything
            self.state_now, self.state_next = self.state_next, self.state_now
            self.sim_time += self.sim_params.sim_dt


    def run(self):
        self.sim_start_time = time.time()
        while self.viewer.is_running() and not self.terminated:
            if not self.viewer.is_paused():
                self.step()
            with wp.ScopedTimer("render", active=False):
                self.render()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_now)
        self.viewer.log_contacts(self.contacts, self.state_now)

        # Register gizmo (viewer will draw & mutate transform in-place).
        # body_q_np = self.state_now.body_q.numpy()
        # self.viewer.log_gizmo("target_tcp", self.ee_tf, snap_to=wp.transform(*body_q_np[self.ee_index]))

        if self.args["debug_particles"]:
            self.viewer.log_points(
                name="/debug/fixed_particles",
                points=self.state_now.particle_q,
                radii=self.particle_debug_radii,
                colors=self.particle_debug_colors,
            )

        # Poke target visualization: dark-red = unpoked, dark-green = poked
        colors_np = np.where(
            self.poke_done[:, None],
            np.array([[0.0, 0.4, 0.0]]),
            np.array([[0.5, 0.0, 0.0]])
        ).astype(np.float32)

        r_poke_viz_pts = 0.001
        self.viewer.log_points(
            name="/poke_targets",
            points=wp.array(self.poke_points, dtype=wp.vec3),
            radii=wp.full(self.poke_params.n_pokes, r_poke_viz_pts, dtype=wp.float32),
            colors=wp.array(colors_np, dtype=wp.vec3),
        )

        if self.contacts.soft_contact_count.numpy()[0] > 0:
            contacted_particle_idxs = self.contacts.soft_contact_particle.numpy()
            contacted_particle_idxs = contacted_particle_idxs[contacted_particle_idxs >= 0]
            contacted_color_np = np.array([0.0, 0.0, 0.5])
            contacted_colors_np = np.tile(contacted_color_np, (len(contacted_particle_idxs), 1)).astype(np.float32)
            self.viewer.log_points(
                name="/soft_contact_particles",
                points=wp.array(self.state_now.particle_q.numpy()[contacted_particle_idxs, :], dtype=wp.vec3),
                radii = wp.full(len(contacted_particle_idxs), 0.0005, dtype=wp.float32),
                colors = wp.array(contacted_colors_np, dtype=wp.vec3)
            )
        else:
            self.viewer.log_points(
                name="/soft_contact_particles",
                points=wp.array([]),
                radii=wp.array([]),
                colors=wp.array([])
            )

        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"Wall time: {time.time() - self.sim_start_time}")
        ui.text(f"Sim time: {self.sim_time}")
        ui.text(f"Poke state: {self.poke_state.name}")
        ui.text(f"Poke {self.i_current_poke + 1} / {self.poke_params.n_pokes}")
        ui.text(f"Latest volume [cm^3]: {self.log_volumes[-1] * 100**3}")
        ui.separator()

        w = 250 # Number of past datapoints to plot
        graph_size = ui.ImVec2(-1, 80)

        def padded(data):
            """Return a fixed-width array, padded with first non-zero value if shorter than the window."""
            arr = np.array(data[-w:], dtype=np.float32)
            if len(arr) < w:
                arr = np.pad(arr, (w - len(arr), 0), mode="edge")
            return arr

        ui.text("Channel volume")
        ui.plot_lines("##iters", padded(self.log_volumes), graph_size=graph_size)

    def terminate(self):
        # End the simulation because we've done all 9 pokes
        print(f"End of poke test reached at sim time {self.sim_time}, elapsed wall time {time.time() - self.sim_start_time:.2f}. Terminating sim")
        self.viewer.close()

        # Save log_sim_times, log_volumes, and i_pokes to a csv
        df_out = pd.DataFrame({
            "sim_times_s": self.log_sim_times,
            "volumes_m3": self.log_volumes,
            "pressures_atm": self.log_pressures,
            "i_poke": self.log_pokes,
            "force_total_N": self.log_forces,
            "force_penalty_N": self.log_forces_penalty,
            "force_elastic_N": self.log_forces_elastic
        })
        df_out.to_csv(f"./logs/{self.args['name']}sim-outputs_{now_str}.csv")

        self.terminated = True
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_particles", action="store_true", help="Show debug particles")
    parser.add_argument("--headless", action="store_true", help="Choose to run without visualization, and instead save a simulation log file")
    parser.add_argument("--name", default="", type=str, help="Prefix to filename")
    return vars(parser.parse_args())

if __name__ == "__main__":
    verbose = False

    args = parse_args()
    if args["headless"]:
        viewer = newton.viewer.ViewerFile(f"{now_str}.bin", auto_save=False)
    else:
        viewer = newton.viewer.ViewerGL(headless=False)

    thighpad_poke_test = ThighpadPokeTest(viewer, args, False)
    thighpad_poke_test.run()

    viewer.close()
