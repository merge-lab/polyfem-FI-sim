import argparse
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime

import numpy as np
import pandas as pd     # Used for mesh pre-processing

import newton
import warp as wp
from pxr import Usd


@wp.func
def tri_volume_contribution(
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
):
    # Compute the signed volume of the tetrahedron with the 3 points + origin
    vol = wp.dot(v0, wp.cross(v1, v2)) / 6.0  # tetra volume (0,v0,v1,v2)
    return vol

@wp.kernel
def compute_solid_mesh_volume(
    indices: wp.array[int],
    vertices: wp.array[wp.vec3],
    # outputs
    volume: wp.array[float],
):
    i = wp.tid()
    p = vertices[indices[i * 3 + 0]]
    q = vertices[indices[i * 3 + 1]]
    r = vertices[indices[i * 3 + 2]]

    # Sum together the signed volumes of all tetrahedra created by each tri-face with origin
    # The final volume will be the actual volume of the mesh
    v = tri_volume_contribution(p, q, r)
    wp.atomic_add(volume, 0, v)

def compute_volume_mesh(
    vertices: wp.array,
    indices: wp.array,
) -> float:
    """
    Compute the mass, center of mass, inertia, and volume of a triangular mesh.

    Args:
        vertices: A wp.array of vertex positions (3D coordinates).
        indices: A wp.array of triangle indices (each triangle is defined by 3 vertex indices).

    Returns:
        A tuple containing:
            - volume: The signed volume of the mesh.
    """

    indices = np.array(indices).flatten()
    num_tris = len(indices) // 3
    vol_warp = wp.zeros(1, dtype=float) # Preallocate output

    wp_vertices = wp.array(vertices, dtype=wp.vec3)
    wp_indices = wp.array(indices, dtype=int)

    wp.launch(
        kernel=compute_solid_mesh_volume,
        dim=num_tris,
        inputs=[
            wp_indices,
            wp_vertices,
        ],
        outputs=[
            vol_warp,
        ],
    )
    V_tot = float(vol_warp.numpy()[0])  # signed volume

    # If the winding is inward, flip signs
    if V_tot < 0: V_tot = -V_tot
    return V_tot

@dataclass
class SimParams:
    g = 9.81

    ### Material parameters
    # TODO move this to different section to handle multiple materials?
    material_E = 1.35e6            # Young's modulus [N/m^2]
    material_nu = 0.45            # Poisson's ratio [unitless]
    material_rho = 1e3          # Density [kg/m^3]
    # Get Lame parameters from Youngs modulus and Poisson's ratio
    @property
    def material_k_lambda(self):
        return self.material_E * self.material_nu / ((1 + self.material_nu) * (1 - 2 * self.material_nu))
    @property
    def material_k_mu(self):
        return self.material_E / (2 * (1 + self.material_nu))
    material_k_damp = 1e-3

    soft_contact_kd = 1e-7      # Soft contact param #TODO better docs
    soft_contact_ke = 1e8       # Soft contact param
    soft_contact_mu = 2.5       # Soft contact param
    rigid_contact_k_start = 1.0e5       # For avbd rigid-rigid contacts
    rigid_avbd_beta = 1.0e8             # For avbd rigid-rigid contacts

    particle_self_contact_radius = 0.0001
    particle_self_contact_margin = 0.0003
    # particle_radius = 0.00005 # 0.1mm diameter
    particle_radius = 0.0005 # 1mm diameter

    # Motion parameters
    z_zero = -0.0855
    # compression_rate = 0.02/60
    compression_rate = 0.02
    compression_depth = 0.002
    t_start_wait = 0.1
    # t_hold = 0.2
    t_hold = 1/60
    t_stop_wait = 0.05

class PokeState(Enum):
    """
        State flags for the state machine that moves the poker up and down
    """
    INIT = 0
    TOP = 1
    DOWN = 2
    BOTTOM = 3
    UP = 4
    STOP = 5

class ThighpadPokeTest:
    def __init__(self, viewer, verbose=False):
        self.sim_time = 0.0
        self.sim_params = SimParams()

        # Setup simulation parameters
        # TODO move these to sim params as well?
        self.fps = 60
        self.frame_dt = 1.0/self.fps                    # dt of each macro-step
        self.sim_steps = self.fps                             # # of macrosteps in sim
        self.sim_substeps = 10                          # # of substeps per macrostep
        self.iterations = 5 
        self.sim_dt = self.frame_dt / self.sim_substeps # dt of each substep
        self.sim_start_time = 0.0

        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -self.sim_params.g), dtype=wp.vec3)

        self.verbose = verbose
        self.viewer = viewer

        self.args = self.parse_args()

        # Create scene
        self.init_q = -0.0855 + 0.0005
        # self.init_q = 0.0
        self.current_q = self.init_q
        init_qs = self.init_q * wp.ones(9, dtype=wp.float32)
        self.model = self.create_model(init_qs)

        # Initialize camera
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3([-0.15, 0.0, 0.03]), -5, 0.0)
        self.viewer._cam_speed = 0.1

        # Initialize both solvers
        self.solver_vbd = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_radius=self.sim_params.particle_self_contact_radius,
            particle_self_contact_margin=self.sim_params.particle_self_contact_margin,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=self.sim_params.rigid_contact_k_start,
            rigid_avbd_beta=self.sim_params.rigid_avbd_beta,
        )

        self.solver_rigid = None
        if not self.args["no_poke"]:
            self.solver_rigid = newton.solvers.SolverMuJoCo(self.model)
            self.poke_state = PokeState.DOWN
            self.t_poke_state_changed = self.sim_time
            if self.args["i_poke"] == -1:
                self.i_current_poke = 0
            else:
                self.i_current_poke = self.args["i_poke"]

        # Preallocate variables for trajectory, control, and contacts
        self.state_now = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()

        # Initialize configuration of poke fixture
        self.control.joint_target_pos = init_qs

        # TODO - look into what collision_pipeline options do
        # self.model.collide(self.state_now, self.contacts)
        self.contacts = self.model.contacts()

        self.graph = None
        self.capture()

        ### Initialize gui and logging
        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="side")
        
        self.vol_initial = -1
        self.log_sim_times: list[float] = []
        self.log_volumes: list[float] = []
        self.log_pressures: list[float] = []
        self.log_pokes:     list[int] = []


    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-poke", action="store_true", help="Skip loading the poke fixture")
        parser.add_argument("--debug_particles", action="store_true", help="Show debug particles")
        parser.add_argument("--i_poke", default=-1, type=int, help="Which poker to push down. Defaults to (-1), which is 'all'")
        parser.add_argument("--name", default="", type=str, help="Prefix to filename")
        return vars(parser.parse_args())

    def create_model(self, init_qs=None):
        """
            Create the simulation scene
        """
        # Initialize the scene
        self.scene = newton.ModelBuilder()
        self.scene.default_particle_radius = self.sim_params.particle_radius

        ## ======= Add thigh pad =============
        self.create_thighpad(self.scene)

        ## ======= Add poke fixture if not disabled =============
        if not self.args["no_poke"]:
            self.create_poker(self.scene, init_qs)
            # self.create_poker_primitives(self.scene, 0.05)

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

    def create_thighpad(self, scene):
        """
            Load the thigh pad and place it in the simulation scene
        """
        # Fetching thighpad asset using the USD ecosystem
        # modelpath_thighpad = "../Assets/Thigh-pad/tets_coarse/model.usda"
        modelpath_thighpad = "../Assets/Thigh-pad/tets_fine/model.usda"
        # modelpath_thighpad = "../Assets/Thigh-pad/mesh_l=0.05_e=2e-3/model.usda"
        # modelpath_thighpad = "../Assets/Thigh-pad/tets_finer/model.usda"
        stage_thighpad = Usd.Stage.Open(modelpath_thighpad)
        prim_thighpad = stage_thighpad.GetPrimAtPath("/root/Model/TetMesh")
        tetmesh_thighpad = newton.TetMesh.create_from_usd(prim_thighpad)

        print(f"k_mu: {self.sim_params.material_k_mu}, k_lambda: {self.sim_params.material_k_lambda}")

        quat_initial = wp.quat_from_axis_angle(wp.vec3([1, 0, 0]), np.pi/2)
        # quat_initial = wp.quat_identity()
        self.pad_start_particle_idx = len(scene.particle_q)
        scene.add_soft_mesh(
            pos         = wp.vec3(0.0, 0.0, 0.000),
            rot         = quat_initial,
            scale       = 1.0,
            vel         = wp.vec3(0.0, 0.0, 0.0),
            mesh        = tetmesh_thighpad,
            density     = self.sim_params.material_rho,
            k_mu        = self.sim_params.material_k_mu,
            k_lambda    = self.sim_params.material_k_lambda,
            # k_mu = 1e5,
            # k_lambda = 1e5,
            k_damp      = self.sim_params.material_k_damp,
            tri_ke      = 0.0,
            tri_ka      = 0.0,
            tri_kd      = 0.0,
            tri_drag    = 0.0,
            tri_lift    = 0.0,

        )

        # Read surface selection csv to find nodes that we want to fix in place
        col_names = ["id_surf", "v_1", "v_2", "v_3"]
        # df_surf_select = pd.read_csv("../Assets/Thigh-pad/tets_coarse/surface_selections.txt", names=col_names, sep="\s+")
        df_surf_select = pd.read_csv("../Assets/Thigh-pad/tets_fine/surface_selections.txt", names=col_names, sep="\s+")
        # df_surf_select = pd.read_csv("../Assets/Thigh-pad/mesh_l=0.05_e=2e-3/surface_selections.txt", names=col_names, sep="\s+")
        # df_surf_select = pd.read_csv("../Assets/Thigh-pad/tets_finer/surface_selections.txt", names=col_names, sep="\s+")
        surf_id_bottom = 1
        df_verts_bottom = df_surf_select[df_surf_select["id_surf"] == surf_id_bottom][col_names[1:]]
        np_verts_bottom = df_verts_bottom.to_numpy(dtype=np.int64)
        ids_verts_bottom = np.unique(np_verts_bottom)

        surf_id_channel = 2
        self.ids_channel = df_surf_select[df_surf_select["id_surf"] == surf_id_channel][col_names[1:]].to_numpy(dtype=np.int64)

        # Fix bottom surface particles in place — must zero mass AND clear ACTIVE flag,
        # matching what add_cloth_grid does (builder.py:7156-7160).
        # The VBD solver has kernels that check each condition independently.
        self._fixed_particle_ids = [] # TODO - this should probably be a Set
        for vert_id in ids_verts_bottom:
            global_id = self.pad_start_particle_idx + vert_id
            scene.particle_mass[global_id] = 0
            scene.particle_flags[global_id] = scene.particle_flags[global_id] & ~newton.ParticleFlags.ACTIVE
            self._fixed_particle_ids.append(global_id) # For debug viz

    def create_poker(self, scene, init_qs):
        builder_poke_fixture = newton.ModelBuilder()
        path_poker = "./usd/poke_fixture_FLATTENED.usd"

        builder_poke_fixture.add_usd(
            path_poker,
            # For some reason the origin of the poke fixture usd is at Poke 6 (-x, +y corner)
            xform=wp.transform((0.025117, -0.020982, 0.1), wp.quat_identity()),
            enable_self_collisions=False,
            force_show_colliders=True,
        )
        builder_poke_fixture.joint_q = init_qs.list()
        scene.add_world(builder_poke_fixture)

    def create_poker_primitives(self, scene, init_z):
        # NOTE - UNUSED. Leaving here for reference in case we want to use it someday.
        X_SPACING = 0.025117
        Y_SPACING = 0.025117
        R_POKER = 0.007/2
        HALF_H_POKER = 0.01
        x_posns = X_SPACING * np.array([-1, 0, 1])
        y_posns = Y_SPACING * np.array([-1, 0, 1])
        Xs, Ys = np.meshgrid(x_posns, y_posns)
        self.poker_links = []
        self.poker_joints = []
        i_poker = 0
        for i in range(3):
            for j in range(3):
                name_ij = f"Poker_{i_poker}"
                posn_ij = wp.vec3([Xs[i, j], Ys[i, j], init_z])
                # poker_ij_body = scene.add_body(
                #     xform=wp.transform(p=posn_ij, q=wp.quat_identity()),
                #     label=f"Poker_{i_poker}",
                #     is_kinematic=True
                # )
                xform_ij = wp.transform(p=posn_ij, q=wp.quat_identity())
                poker_ij_link = scene.add_link(xform=xform_ij, mass=15.0, is_kinematic=True, label=name_ij)
                scene.add_shape_cylinder(poker_ij_link, radius=R_POKER, half_height=HALF_H_POKER)
                joint_ij = scene.add_joint_prismatic(
                    parent=-1,
                    child=poker_ij_link,
                    axis=newton.Axis.Z,
                    parent_xform=xform_ij,
                    label=name_ij+"_joint"
                )
                scene.add_articulation([joint_ij], label=name_ij+"_art")

                self.poker_links.append(poker_ij_link)
                self.poker_joints.append(joint_ij)
                
                i_poker += 0

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

    
    def simulate(self):
        """
            Go through all of the sub-steps, co)rresponding to one macro step...??
            Not quite sure how this makes sense
        """

        self.solver_vbd.rebuild_bvh(self.state_now)
        for i in range(self.sim_substeps):
            # Old - bill
            # # Reset forces on the current state
            self.state_now.clear_forces()
            self.state_next.clear_forces()
            self.viewer.apply_forces(self.state_now)

            if not self.args["no_poke"]:
                # # Featherstone as kinematic integrator (disable particles + gravity)
                particle_count = self.model.particle_count
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)
                self.model.shape_contact_pair_count = 0

                # # self.state_now.joint_qd.assign(self.target_joint_qd) # TODO - move joints
                self.solver_rigid.step(self.state_now, self.state_next, self.control, None, self.sim_dt)

                self.state_now.particle_f.zero_()
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            # # TODO - look into what collision pipelines do
            self.model.collide(self.state_now, self.contacts)
            self.solver_vbd.step(self.state_now, self.state_next, self.control, self.contacts, self.sim_dt)

            # # Swap the states (update state_now to be state_next)
            # # We can swap because state_next can really be anything
            # # TODO - maybe it can be None? or does that break things
            self.state_now, self.state_next = self.state_next, self.state_now

            # print(f"State now: {self.state_now.__dict__}")
            self.sim_time += self.sim_dt


    def step(self):
        if not self.args["no_poke"]:
            self._control_poker()

        if self.graph:
            wp.capture_launch(self.graph)
            self.sim_time += self.sim_dt
        else:
            self.simulate()
        
        self._log_states()


    def _control_poker(self):

        if self.poke_state == PokeState.DOWN:
            if self.sim_time < self.sim_params.t_start_wait:
                # Hold for a bit at the start to let signal stabilize
                joint_vel = 0.0
            else:
                if self.current_q <= self.sim_params.z_zero - self.sim_params.compression_depth:
                    # If reached bottom out distance, switch to the 0.5sec hold
                    self.poke_state = PokeState.BOTTOM
                    joint_vel = 0.0
                    self.t_poke_state_changed = self.sim_time # Track last state change time
                else:
                    # Compression rate in Gilbert test: 20mm / min
                    joint_vel = -self.sim_params.compression_rate

        elif self.poke_state == PokeState.BOTTOM:
            if self.sim_time - self.t_poke_state_changed >= self.sim_params.t_hold:
                # Transition poke state if we have waited for long enough
                self.poke_state = PokeState.UP
                self.t_poke_state_changed = self.sim_time # Track last state change time
            joint_vel = 0.0     # Wait until enough time has past since last state change

        elif self.poke_state == PokeState.UP:
            if self.current_q >= self.init_q:
                # Stop the poke test once we have reached a higher height
                self.poke_state = PokeState.STOP
                joint_vel = 0.0
                self.t_poke_state_changed = self.sim_time # Track last state change time
            else:
                joint_vel = self.sim_params.compression_rate

        elif self.poke_state == PokeState.STOP:
            if self.sim_time - self.t_poke_state_changed >= self.sim_params.t_stop_wait:
                # Wait 0.15sec before advancing
                if self.args["i_poke"] != -1:
                    self.terminate()
                    return

                self.i_current_poke += 1 # Advance to next poke
                if self.i_current_poke >= 9:
                    self.terminate()
                    return
                else:
                    # Begin the next poke
                    self.poke_state = PokeState.DOWN

            joint_vel = 0.0

        else:
            raise ValueError("Invalid poke state flag value")

        controller_hz = self.frame_dt/self.sim_substeps
        dq = joint_vel * controller_hz
        self.current_q += dq

        # Set the selection array based on which poke ids are selected
        if self.args["i_poke"] == -1:
            # control_mask = wp.ones(9, dtype=wp.float32)
            
            # Set the motion for the currently active poker
            control_mask = np.zeros(9, dtype=np.float32)
            control_mask[self.i_current_poke] = 1
            control_mask = wp.array(control_mask)
        elif self.args["i_poke"] >= 0 and self.args["i_poke"] <= 8:
            control_mask = np.zeros(9, dtype=np.float32) 
            control_mask[self.args["i_poke"]] = 1
            control_mask = wp.array(control_mask)
        else:
            raise ValueError("Invalid poke index selected; value needs to be an integer between 0 and 8")

        target_qs = self.current_q * control_mask.view(wp.float32)
        target_qds = dq * control_mask.view(wp.float32)
        wp.copy(self.state_now.joint_q, wp.array(target_qs, dtype=wp.float32))
        wp.copy(self.state_now.joint_qd, wp.array(target_qds, dtype=wp.float32))

        return

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
        if self.args["debug_particles"]:
            self.viewer.log_points(
                name="/debug/fixed_particles",
                points=self.state_now.particle_q,
                radii=self.particle_debug_radii,
                colors=self.particle_debug_colors,
            )
        self.viewer.end_frame()

    def _log_states(self):

        tube_diameter = 0.0254/16       # 1/16" in meters
        tube_length = 576.22 / 1e3      # 576.22mm in meters
        tube_vol = (wp.PI * (tube_diameter/2)**2) * tube_length

        # Log the channel volume
        if self.sim_time >= self.sim_params.t_start_wait:
            # Get channel volume
            idxs = self.pad_start_particle_idx + self.ids_channel
            verts = self.state_now.particle_q[self.pad_start_particle_idx:]
            channel_volume = compute_volume_mesh(verts, idxs)
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
        self.log_sim_times.append(self.sim_time)
        self.log_pokes.append(self.i_current_poke)

    def gui(self, ui):
        ui.text(f"Wall time: {time.time() - self.sim_start_time}")
        ui.text(f"Sim time: {self.sim_time}")
        ui.text(f"Latest volume [cm^3]: {self.log_volumes[-1] * 100**3}")
        ui.text(f"Poke state: {self.poke_state}")
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
        now = datetime.now()
        now_str = now.strftime("%d-%m-%Y_%H:%M:%S")
        df_out = pd.DataFrame({
            "sim_times_s": self.log_sim_times,
            "volumes_m3": self.log_volumes,
            "pressures_atm": self.log_pressures,
            "i_poke": self.log_pokes
        })
        df_out.to_csv(f"./logs/{self.args['name']}sim-outputs_{now_str}.csv")
        

if __name__ == "__main__":
    viewer = newton.viewer.ViewerGL(headless=False)
    # Default camera pose: 
    #  - pos: [10.0, 0.0, 2.0]
    #  - pitch: 0.0
    #  - yaw: -180

    verbose = False
    thighpad_poke_test = ThighpadPokeTest(viewer, False)
    thighpad_poke_test.run()

    viewer.close()
