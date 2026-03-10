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
        # self.solver = newton.solvers.SolverFeatherstone(self.model)

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

        ## Load the vertex positions and tet indices from the gmsh file
        # filepath = "../PolyFEM/Thigh-pad/six-channel/meshes/ThighpadRemake - Part_fine.msh"
        filepath = "../PolyFEM/Thigh-pad/six-channel/meshes/ThighpadRemake - Part_coarsened.msh"
        # filepath = "../PolyFEM/Thigh-pad/six-channel/meshes/ThighpadRemake - Part_coarsened_ascii.msh"
        # filepath = "../PolyFEM/Examples/Pneumatics-examples/Hollow Sphere Cavity/meshes/HollowSphere - part.msh"
        # filepath = "../PolyFEM/Examples/Pneumatics-examples/Cube Inflation/cube2x2_hollow.msh"
        # filepath = "../PolyFEM/Examples/Pneumatics-examples/Cube Inflation/cube2x2_hole.msh"
        # filepath = "../PolyFEM/Examples/Pneumatics-examples/Cube Inflation/cube.msh"
        # filepath = "cube_ftetwild.msh"
        # filepath = "cube_ftetwild_reindexed.msh"
        # filepath = "cube_ftetwild_manif.msh"
        # filepath = "cube_gmsh.msh"
        # filepath = "cube_breaks.msh"
        msh_thighpad = meshio.read(filepath)

        # verts_thighpad = 0.1 * msh_thighpad.points
        verts_thighpad = 500.0*msh_thighpad.points
        tet_indices = msh_thighpad.cells_dict["tetra"]

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.scatter(verts_thighpad[:, 0], verts_thighpad[:, 1], verts_thighpad[:, 2])
        # plt.show()

        print(f"Verts dtype: {verts_thighpad.dtype}")
        print("Verts:")
        print(verts_thighpad)
        print("Verts range:")
        print(np.array([np.min(verts_thighpad, axis=0), np.max(verts_thighpad, axis=0)]))
        print(f"Tet dtype: {tet_indices.dtype}")

        mesh_pts_wp = [wp.vec3(vert) for vert in verts_thighpad.astype(np.float32)]
        # tet_indices_wp = Vt.IntArray(tet_indices.flatten().astype(np.int64).tolist())
        tet_indices_wp = Vt.IntArray.FromNumpy(tet_indices.flatten().astype(np.int64))
        # tet_indices_wp = Vt.IntArray.FromNumpy(np.ones(tet_indices.shape).flatten().astype(np.int64))

        ## Build the mesh in Newton's model builder
        builder = newton.ModelBuilder()
        builder.default_particle_radius = 0.05

        # Get Lame parameters from Youngs modulus and Poisson's ratio
        E = 1e4 # Youngs modulus (Pa)
        nu = 0.3 # Poisson's ratio
        k_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
        k_mu = E / (2 * (1 + nu))

        print(f"k_lambda: {k_lambda}, k_mu: {k_mu}")

        # quat_initial = rot.as_quat(rot.from_euler("xyz", [np.pi/2, 0, 0]))
        quat_initial = wp.quat_from_axis_angle(wp.vec3([1, 0, 0]), np.pi/2)
        builder.add_soft_mesh(
            pos         = wp.vec3(-40, 0.0, 10),
            rot         = quat_initial,
            scale       = 1.0,
            vel         = wp.vec3(0.0, 0.0, 0.0),
            vertices    = mesh_pts_wp,
            indices     = tet_indices_wp,
            density     = 1.0,
            k_mu        = k_mu,
            k_lambda    = k_lambda,
            k_damp      = 2.0,
            tri_ke      = 0.0,
            tri_ka      = 1e-8,
            tri_kd      = 1e-4,
            tri_drag    = 0.0,
            tri_lift    = 0.0,
        )

        # Add ground plane
        # TODO - understand the meaning of these numbers by looking at semi-implicit docs
        ke = 1000
        kf = 0.0
        kd = 10
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
