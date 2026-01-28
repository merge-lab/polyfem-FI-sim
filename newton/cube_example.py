import numpy as np
import warp as wp

import newton
import newton.examples

class CubeExample:
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

        # ToLearn: what do you pass in here if you have multiple models?
        # or does the "model" object actually hold many models? since we "added"
        # the soft grid to it.
        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        # Preallocate variables for trajectory, control, and contacts
        # ToLearn - do we always need control?
        self.state_now = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()

        # TODO - look into what collision_pipeline options do
        self.contacts = self.model.collide(self.state_now, soft_contact_margin=0.001)

        self.graph = None
        self.capture()

    def create_model(self):
        """
            Create the FEM mesh model
            TODO - is this a tet mesh? Hex mesh?
        """
        builder = newton.ModelBuilder()
        builder.default_particle_radiu = 0.0005

        cells_per_side = 2
        cell_size = 0.1
        total_mass = 0.2

        # Compute particle density
        num_particles = cells_per_side ** 3
        particle_mass = total_mass / num_particles
        particle_density = particle_mass / (cell_size ** 3)

        # Get Lame parameters from Youngs modulus and Poisson's ratio
        E = 1e5 # Youngs modulus (Pa)
        nu = 0.4 # Poisson's ratio
        k_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
        k_mu = E / (2 * (1 + nu))

        # Add the soft grid using the model builder
        builder.add_soft_grid(
            pos=wp.vec3(0, 0, 0.5),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=cells_per_side,
            dim_y=cells_per_side,
            dim_z=cells_per_side,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=particle_density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=0.0,
            tri_ke=1e-4, # TODO - what is this?
            tri_ka=1e-4, # TODO - what is this?
            tri_kd=1e-4, # TODO - what is this?
            tri_drag=0.0,
            tri_lift=0.0,
            fix_bottom=False,
        )

        # Add ground plane
        ke = 1.0e3
        kf = 0.0
        kd = 1.0e0
        mu = 0.2
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu))

        # Finalize and export the model
        model = builder.finalize()
        model.soft_contact_ke = ke
        model.soft_contact_kf = kf
        model.soft_contact_kd = kd
        model.soft_contact_mu = mu
        return model

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
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

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

    verbose = False
    cube_example = CubeExample(viewer, False)
    cube_example.run()

    viewer.close()
