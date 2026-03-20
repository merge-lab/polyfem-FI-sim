import numpy as np
from scipy.spatial.transform import Rotation as rot

import meshio

import warp as wp
import newton
import newton.examples
from pxr import Usd, UsdGeom, Vt

import matplotlib.pyplot as plt

class CubeExample:
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

        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)

        self.verbose = verbose
        self.viewer = viewer

        # Create FEM model
        self.model = self.create_model()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3([-0.5, 0.0, 0.25]), -20, 0.0)

        # ToLearn: what do you pass in here if you have multiple models?
        # or does the "model" object actually hold many models? since we "added"
        # the soft grid to it.
        self.solver_vbd = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,
            # particle_enable_tile_solve=False,
            particle_enable_self_contact=False,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            particle_collision_detection_interval=-1,
        )
        self.solver_featherstone = newton.solvers.SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)

        # Preallocate variables for trajectory, control, and contacts
        # ToLearn - do we always need control?
        self.state_now = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()

        # TODO - look into what collision_pipeline options do
        # self.model.collide(self.state_now, self.contacts)
        self.contacts = self.model.contacts()

        self.graph = None
        self.capture()

    def create_model(self):
        """
            Create the FEM mesh model
            TODO - is this a tet mesh? Hex mesh?
        """
        # Initialize the scene
        self.scene = newton.ModelBuilder()
        self.scene.default_particle_radius = 0.005

        ## ======= Add thigh pad =============
        # Fetching thighpad asset using the USD ecosystem
        modelpath_thighpad = "../Assets/Thigh-pad/tets_coarse/model.usda"
        stage_thighpad = Usd.Stage.Open(modelpath_thighpad)
        prim_thighpad = stage_thighpad.GetPrimAtPath("/root/Model/TetMesh")
        tetmesh_thighpad = newton.TetMesh.create_from_usd(prim_thighpad)

        # Get Lame parameters from Youngs modulus and Poisson's ratio
        E = 1e6 # Youngs modulus (Pa)
        nu = 0.4 # Poisson's ratio
        k_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
        k_mu = E / (2 * (1 + nu))
        rho = 1e3
        # k_lambda = 1e5
        # k_mu = 1e5
        scale = 1.0

        print(f"k_lambda: {k_lambda}, k_mu: {k_mu}")

        quat_initial = wp.quat_from_axis_angle(wp.vec3([1, 0, 0]), np.pi/2)
        # quat_initial = wp.quat_identity()
        self.scene.add_soft_mesh(
            pos         = wp.vec3(0.0, 0.0, 0.05),
            rot         = quat_initial,
            scale       = scale,
            vel         = wp.vec3(0.0, 0.0, 0.0),
            mesh        = tetmesh_thighpad,
            density     = rho,
            k_mu        = k_mu,
            k_lambda    = k_lambda,
            k_damp      = 1e-3,
            tri_ke      = 0.0,
            tri_ka      = 1e-8,
            tri_kd      = 1e-4,
            tri_drag    = 0.0,
            tri_lift    = 0.0,
        )

        ## ======= Add poke test fixture =======
        builder_poke_fixture = newton.ModelBuilder()
        self.create_poker(builder_poke_fixture)
        self.scene.add_world(builder_poke_fixture)

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
        return model

    def create_poker(self, builder):
        path_poker = "./usd/thighpad_poke_fixture_onshape.usd"
        builder.add_usd(
            path_poker,
            # For some reason the origin of the poke fixture usd is at Poke 6 (-x, +y corner)
            xform=wp.transform((0.025117, -0.020982, 0.1), wp.quat_identity()),
            enable_self_collisions=False,
            force_show_colliders=True,
        )

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

            # Featherstone as kinematic integrator (disable particles + gravity)
            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0

            # # self.state_now.joint_qd.assign(self.target_joint_qd) # TODO - move joints
            self.solver_featherstone.step(self.state_now, self.state_next, self.control, None, self.sim_dt)

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

            print(f"State now: {self.state_now.__dict__}")
            self.sim_time += self.sim_dt


    def step(self):
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
        self.viewer.end_frame()

        

if __name__ == "__main__":
    viewer = newton.viewer.ViewerGL(headless=False)
    # Default camera pose: 
    #  - pos: [10.0, 0.0, 2.0]
    #  - pitch: 0.0
    #  - yaw: -180

    verbose = False
    cube_example = CubeExample(viewer, False)
    cube_example.run()

    viewer.close()
