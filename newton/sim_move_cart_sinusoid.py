import numpy as np
import warp as wp

import newton
import newton.examples

class Example:
    def __init__(self, viewer, verbose=False):
        # Setup simulation parameters
        self.fps = 60
        self.frame = 0
        self.sim_time = 0.0
        self.frame_dt = 1.0/self.fps                    # dt of each macro-step
        self.sim_steps = 60                             # # of macrosteps in sim
        self.sim_substeps = 16                          # # of substeps per macrostep
        self.sim_dt = self.frame_dt / self.sim_substeps # dt of each substep

        self.verbose = verbose
        self.viewer = viewer

        # Create FEM model
        self.model = self.create_model()
        self.viewer.set_model(self.model)
        # self.viewer.set_camera(wp.vec3([-0.5, 0.0, 0.25]), -20, 0.0)

        # ToLearn: what do you pass in here if you have multiple models?
        # or does the "model" object actually hold many models? since we "added"
        # the soft grid to it.
        # self.solver = newton.solvers.SolverSemiImplicit(self.model)
        # self.solver = newton.solvers.SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.solver = newton.solvers.SolverMuJoCo(self.model)

        # Preallocate variables for trajectory, control, and contacts
        # ToLearn - do we always need control?
        self.state_now = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()


        ## Move 1: set the initial joint position
        wp.copy(self.control.joint_target_pos, wp.array([-2,], dtype=wp.float32))

        self.graph = None
        self.capture()

    def create_model(self):
        """
            Create the FEM mesh model
            TODO - is this a tet mesh? Hex mesh?
        """
        self.scene = newton.ModelBuilder()
        self.scene.default_particle_radius = 0.0005

        # Create the poker
        builder_poke_fixture = newton.ModelBuilder()
        self.create_poker(builder_poke_fixture)
        self.scene.add_world(builder_poke_fixture)

        # Add ground plane
        # TODO - understand the meaning of these numbers by looking at semi-implicit docs
        ke = 1000
        kf = 0.0
        kd = 10
        mu = 0.2
        # self.scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu))

        # Finalize and export the model
        model = self.scene.finalize()
        model.soft_contact_ke = ke
        model.soft_contact_kf = kf
        model.soft_contact_kd = kd
        model.soft_contact_mu = mu
        return model

    def create_poker(self, builder):
        path = "./usd/cart_on_rail.usda"
        builder.add_usd(
            path,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )
        # set initial joint positions
        builder.joint_q = [0.0]

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
            Go through all of the sub-steps, corresponding to one macro step...??
            Not quite sure how this makes sense
        """

        for i in range(self.sim_substeps):
            # Reset forces on the current state
            self.state_now.clear_forces()
            self.viewer.apply_forces(self.state_now)

            # TODO - look into what collision pipelines do
            self.contacts = self.model.collide(self.state_now)
            self.solver.step(self.state_now, self.state_next, self.control, self.contacts, self.sim_dt)

            # Swap the states (update state_now to be state_next)
            # We can swap because state_next can really be anything
            # TODO - maybe it can be None? or does that break things
            self.state_now, self.state_next = self.state_next, self.state_now

    def step(self):
        ## Move 2: dynamically update the joint position as a sinusoid.
        wp.copy(self.control.joint_target_pos, wp.array([2. * np.sin(self.sim_time* 2*np.pi / 10 - np.pi/2)], dtype=wp.float32))

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def run(self):
        while self.viewer.is_running():
            if not self.viewer.is_paused():
                print("Stepping simulation")
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

    verbose = False
    example = Example(viewer, False)
    example.run()

    viewer.close()
