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

class ThighpadPokeTest:
    def __init__(self, viewer, args, verbose=False):
        self.sim_start_time = 0.0
        self.sim_time = 0.0
        self.sim_params = newton_builders.SimParams()
        self.sim_params.particle_radius = 0.00005
        self.sim_params.sim_substeps = 5
        self.sim_params.iterations = 10
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
        # self.scene.add_urdf(
        #     newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        #     floating=False,
        # )

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


        # # Update the robot control gizmo & robot configuration
        # self._push_targets_from_gizmos()
        # newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_now)

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
            self.model.collide(self.state_now, self.contacts)

            # VBD sim step of soft bodies (it'll ignore rigids bc we configured it to earlier.)
            self.solver_vbd.step(self.state_now, self.state_next, self.control, self.contacts, self.sim_params.sim_dt)

            # # Swap the states (update state_now to be state_next)
            # # We can swap because state_next can really be anything
            self.state_now, self.state_next = self.state_next, self.state_now
            self.sim_time += self.sim_params.sim_dt

    def run(self):
        self.sim_start_time = time.time()
        while self.viewer.is_running():
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
            "i_poke": self.log_pokes
        })
        df_out.to_csv(f"./logs/{self.args['name']}sim-outputs_{now_str}.csv")
    

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
        viewer = newton.viewer.ViewerFile(f"{now_str}.bin")
    else:
        viewer = newton.viewer.ViewerGL(headless=False)

    thighpad_poke_test = ThighpadPokeTest(viewer, args, False)
    thighpad_poke_test.run()

    viewer.close()
