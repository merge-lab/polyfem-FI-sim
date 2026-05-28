# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody Hanging
#
# This simulation demonstrates volumetric soft bodies (tetrahedral grids) hanging
# from fixed particles on the left side. Four grids with different damping values
# (1e-1 to 1e-4) showcase the effect of damping on Neo-Hookean elastic behavior.
#
# Command: uv run -m newton.examples softbody.example_softbody_hanging
#
###########################################################################

import time

import warp as wp

import newton
import newton.examples


LARGE = False
class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_type = args.solver
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.iterations = 150
        self.sim_dt = self.frame_dt / self.sim_substeps

        if self.solver_type != "vbd":
            raise ValueError("The hanging softbody example only supports the VBD solver.")

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        
        self.add_softbodies(builder)

        # Color the mesh for VBD solver
        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 0
        self.model.soft_contact_mu = 1.0

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=not LARGE, # When simulating the small scale bars, the large sag deformations causes element inversion if there's no particle self contact.
            particle_self_contact_radius = 0.00005,
            particle_self_contact_margin = 0.00015,
            particle_enable_tile_solve=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3([4.5, 2, 1.36]), -1.0, 180.0)

        self.capture()

        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="side")
        self.sim_start_time = time.time()

    def add_softbodies(self, builder):
        # Grid dimensions
        if LARGE:
            h = 12*0.1
            l = 4*0.1
        else:
            h = 0.155
            l = 0.045
        base_cell_size = 0.2
        base_dim_x = int(h/base_cell_size)
        base_dim_y = int(l/base_cell_size)
        base_dim_z = base_dim_y

        # Material parameters
        material_E = 1.35e6            # Young's modulus [N/m^2]
        material_nu = 0.45            # Poisson's ratio [unitless]
        # material_rho = 65 / 65754 * 1000**3 / 1000  # 65g / 65754mm^3, converted from g/mm3 to kg/m3, = 988.533 kg/m3
        k_mu = material_E / (2 * (1 + material_nu))
        k_lambda = material_E * material_nu / ((1 + material_nu) * (1 - 2 * material_nu))
        k_damp = 1e-6

        # Create base configuration dict
        dict_baseparams = dict(
            pos=wp.vec3(0.0, 1.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=base_dim_x,
            dim_y=base_dim_y,
            dim_z=base_dim_z,
            cell_x=base_cell_size,
            cell_y=base_cell_size,
            cell_z=base_cell_size,
            tri_ke=material_E,
            density=1.0e3,
            k_mu = k_mu,
            k_lambda = k_lambda,
            k_damp=k_damp,
            fix_left=True,
        )

        spacing = l*1.5     # Space between grids along Y-axis
        if LARGE:
            cell_sizes = [0.044, 0.026, 0.021, 0.0182] # Corresponding to [11k, 51k, 100k, 143k] tets
        else:
            # cell_sizes = [0.0044, 0.0026] Scaled down version, i just naively 0.1x'd all the cell sizes.
            cell_sizes = [0.0051, 0.0031] # Corresponds to [9.6k, 49k] tets. Omitting higher resolutions bc turning on self-collision makes them too slow.

        for i, cs in enumerate(cell_sizes):
            # Modify each configuration dict with its individual parameters
            config_i = dict_baseparams.copy()
            config_i["pos"] = wp.vec3(0.0, 1.0 + i*spacing, 1.0)
            config_i["dim_x"], config_i["dim_y"], config_i["dim_z"] = int(h/cs), int(l/cs), int(l/cs)
            config_i["cell_x"], config_i["cell_y"], config_i["cell_z"] = cs, cs, cs

            builder.add_soft_grid(**config_i)

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # Test that particles are in a reasonable range (soft body may settle or deform)
        # We check that they haven't exploded or collapsed completely
        # 4 grids, each roughly 1.2 x 0.4 x 0.4 in size, positioned along Y-axis
        # Initial positions: Y from 1.0 to ~3.2, X from 0 to 1.2, Z around 1.0 to 1.4
        # With fix_left=True, grids hang and sag significantly towards the ground
        p_lower = wp.vec3(-1.0, -0.5, 0.0)
        p_upper = wp.vec3(3.0, 4.0, 3.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"Wall time: {time.time() - self.sim_start_time}")
        ui.text(f"Sim time: {self.sim_time}")
        ui.separator()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            help="Type of solver (only 'vbd' supports volumetric soft bodies)",
            type=str,
            choices=["vbd"],
            default="vbd",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
