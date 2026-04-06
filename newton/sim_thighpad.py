import numpy as np
from scipy.spatial.transform import Rotation as rot

import meshio

# Used for mesh pre-processing
import pandas as pd
from scipy.spatial import KDTree

import warp as wp
import newton
import newton.examples
from pxr import Usd, UsdGeom, Vt

import matplotlib.pyplot as plt
import argparse

class ThighpadPokeTest:
    def __init__(self, viewer, verbose=False):
        # Setup simulation parameters
        self.fps = 60
        self.frame = 0
        self.sim_time = 0.0
        self.frame_dt = 1.0/self.fps                    # dt of each macro-step
        self.sim_steps = 60                             # # of macrosteps in sim
        self.sim_substeps = 10                          # # of substeps per macrostep
        self.iterations = 5 
        self.sim_dt = self.frame_dt / self.sim_substeps # dt of each substep

        self.particle_radius = 0.0005 #1mm diameter

        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)

        self.verbose = verbose
        self.viewer = viewer

        self.args = self.parse_args()

        # Create scene
        self.model = self.create_model()

        # Initialize camera
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3([-0.15, 0.0, 0.03]), -5, 0.0)
        self.viewer._cam_speed = 0.1

        # Initialize both solvers
        self.solver_vbd = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,
            # particle_enable_tile_solve=False,
            # particle_enable_self_contact=False,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.0002,
            particle_self_contact_margin=0.0004,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            particle_collision_detection_interval=-1,
            particle_enable_tile_solve=False,
            rigid_contact_k_start=1.0e5,
            rigid_avbd_beta=1.0e6,
        )

        self.solver_rigid = None
        if not self.args["no_poke"]:
            self.solver_rigid = newton.solvers.SolverMuJoCo(self.model)

        # Preallocate variables for trajectory, control, and contacts
        self.state_now = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()

        # TODO - look into what collision_pipeline options do
        # self.model.collide(self.state_now, self.contacts)
        self.contacts = self.model.contacts()

        self.graph = None
        self.capture()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-poke", action="store_true", help="Skip loading the poke fixture")
        parser.add_argument("--debug_particles", action="store_true", help="Show debug particles")
        return vars(parser.parse_args())

    def create_model(self):
        """
            Create the simulation scene
        """
        # Initialize the scene
        self.scene = newton.ModelBuilder()
        self.scene.default_particle_radius = self.particle_radius

        ## ======= Add thigh pad =============
        self.create_thighpad(self.scene)

        ## ======= Add poke fixture if not disabled =============
        if not self.args["no_poke"]:
            self.create_poker(self.scene)

        ## ======== Add ground plane =======
        # TODO - understand the meaning of these numbers by looking at semi-implicit docs
        # ke = 100
        ke = 2e6
        kf = 1
        # kd = 0.0
        kd = 1e-7
        mu = 1.5
        # builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu))
        self.scene.add_ground_plane()
 
        ## ======== Finalize and export the model =======
        self.scene.color()
        model = self.scene.finalize()
        model.soft_contact_ke = ke
        model.soft_contact_kf = kf
        model.soft_contact_kd = kd
        model.soft_contact_mu = mu

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
        # modelpath_thighpad = "../Assets/Thigh-pad/tets_finer/model.usda"
        stage_thighpad = Usd.Stage.Open(modelpath_thighpad)
        prim_thighpad = stage_thighpad.GetPrimAtPath("/root/Model/TetMesh")
        tetmesh_thighpad = newton.TetMesh.create_from_usd(prim_thighpad)

        # Get Lame parameters from Youngs modulus and Poisson's ratio
        E = 1e6 # Youngs modulus (Pa)
        nu = 0.45 # Poisson's ratio
        k_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
        k_mu = E / (2 * (1 + nu))
        rho = 1.0e3 # kg/m^3
        scale = 1.0

        print(f"k_lambda: {k_lambda}, k_mu: {k_mu}")

        quat_initial = wp.quat_from_axis_angle(wp.vec3([1, 0, 0]), np.pi/2)
        # quat_initial = wp.quat_identity()
        start_particle_idx = len(scene.particle_q)
        scene.add_soft_mesh(
            pos         = wp.vec3(0.0, 0.0, 0.000),
            rot         = quat_initial,
            scale       = scale,
            vel         = wp.vec3(0.0, 0.0, 0.0),
            mesh        = tetmesh_thighpad,
            density     = rho,
            k_mu        = k_mu,
            k_lambda    = k_lambda,
            k_damp      = 1e-4,
            tri_ke      = 0.0,
            tri_ka      = 0.0,
            tri_kd      = 0.0,
            tri_drag    = 0.0,
            tri_lift    = 0.0,
        )

        # Read surface selection csv to find nodes that we want to fix in place
        col_names = ["id_surf", "v_x", "v_y", "v_z"]
        # df_surf_select = pd.read_csv("../Assets/Thigh-pad/tets_coarse/surface_selections.txt", names=col_names, sep="\s+")
        df_surf_select = pd.read_csv("../Assets/Thigh-pad/tets_fine/surface_selections.txt", names=col_names, sep="\s+")
        # df_surf_select = pd.read_csv("../Assets/Thigh-pad/tets_finer/surface_selections.txt", names=col_names, sep="\s+")
        surf_id_bottom = 1
        df_verts_bottom = df_surf_select[df_surf_select["id_surf"] == surf_id_bottom][["v_x", "v_y", "v_z"]]
        np_verts_bottom = df_verts_bottom.to_numpy(dtype=np.int64)
        ids_verts_bottom = np.unique(np_verts_bottom)

        # Fix bottom surface particles in place — must zero mass AND clear ACTIVE flag,
        # matching what add_cloth_grid does (builder.py:7156-7160).
        # The VBD solver has kernels that check each condition independently.
        self._fixed_particle_ids = [] # TODO - this should probably be a Set
        for vert_id in ids_verts_bottom:
            global_id = start_particle_idx + vert_id
            scene.particle_mass[global_id] = 0
            scene.particle_flags[global_id] = scene.particle_flags[global_id] & ~newton.ParticleFlags.ACTIVE
            self._fixed_particle_ids.append(global_id) # For debug viz

    def create_poker(self, scene):
        builder_poke_fixture = newton.ModelBuilder()
        path_poker = "./usd/thighpad_poke_fixture_onshape.usd"
        builder_poke_fixture.add_usd(
            path_poker,
            # For some reason the origin of the poke fixture usd is at Poke 6 (-x, +y corner)
            xform=wp.transform((0.025117, -0.020982, 0.1), wp.quat_identity()),
            enable_self_collisions=False,
            force_show_colliders=True,
        )
        scene.add_world(builder_poke_fixture)

    def setup_debug_viz(self, model):
        # Build per-particle debug color array: blue for fixed, gray for free
        debug_radius = self.particle_radius
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
            time_period = 0.25
            press_start_dist = 0.000
            press_end_dist = -0.005
            dist_to_top_surf = 0.087    
            stroke_depth = press_start_dist - press_end_dist
            offset = dist_to_top_surf - press_start_dist

            z = offset + stroke_depth/2 * np.cos(self.sim_time* 2*np.pi / time_period + np.pi)
            v_joint_target_pos_np = z * np.ones(9)
            wp.copy(self.control.joint_target_pos, wp.array(v_joint_target_pos_np, dtype=wp.float32))

        if self.graph:
            wp.capture_launch(self.graph)
            self.sim_time += self.sim_dt
        else:
            self.simulate()


    def run(self):
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
