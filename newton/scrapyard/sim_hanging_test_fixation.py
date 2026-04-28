# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

import numpy as np

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_type = args.solver
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        if self.solver_type != "vbd":
            raise ValueError("The hanging softbody example only supports the VBD solver.")

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Grid dimensions
        dim_x = 12
        dim_y = 4
        dim_z = 4
        cell_size = 0.1

        # Create 4 grids with different damping values
        damping_values = [1e-1, 1e-2, 1e-3, 1e-4]
        spacing = 0.6  # Space between grids along Y-axis

        self._fixed_particle_ids = [] # TODO - this should probably be a Set
        for i, k_damp in enumerate(damping_values):
            idx_start_particle_i = len(builder.particle_q)

            y_offset = i * spacing
            builder.add_soft_grid(
                pos=wp.vec3(0.0, 1.0, 0.5 + i*cell_size*dim_z*1.25),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=dim_x,
                dim_y=dim_y,
                dim_z=dim_z,
                cell_x=cell_size,
                cell_y=cell_size,
                cell_z=cell_size,
                density=1.0e3,
                k_mu=1.0e5,
                k_lambda=1.0e5,
                k_damp=k_damp,
                # fix_left=True,
            )

            fix_left = False
            if fix_left:
                ## Fix the left-most nodes in our mesh using our method rather than the function parameter
                # First find the indices of all the left-most particles
                mesh_particles_i = builder.particle_q[idx_start_particle_i : ]
                verts_i = np.array(mesh_particles_i)
                x_fix = 0.5
                i_fix = np.nonzero(verts_i[:, 0] == x_fix)[0] # Fix left side (same as Newton/buildrer.py)
                
                # Now set those particles' masses to zero

                # Fix bottom surface particles in place — must zero mass AND clear ACTIVE flag,
                # matching what add_cloth_grid does (builder.py:7156-7160).
                # The VBD solver has kernels that check each condition independently.
                for vert_id in i_fix:
                    global_id = idx_start_particle_i + vert_id
                    builder.particle_mass[global_id] = 0.0
                    # builder.particle_flags[global_id] = builder.particle_flags[global_id] # Try leaving it unchanged and see what happens
                    # builder.particle_flags[global_id] = builder.particle_flags[global_id] & ~newton.ParticleFlags.ACTIVE
                    self._fixed_particle_ids.append(global_id) # For debug viz
                

        # Color the mesh for VBD solver
        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 0
        self.model.soft_contact_mu = 1.0

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            # particle_enable_self_contact=False,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.04,
            particle_self_contact_margin=0.06,
            particle_enable_tile_solve=False,
        )

        self.setup_debug_viz(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        self.capture()

    def setup_debug_viz(self, model):
        # Build per-particle debug color array: blue for fixed, gray for free
        debug_radius = 0.02
        particle_color_default = [0.6, 0.6, 0.6] # Gray
        particle_color_fixed = [0.0, 0.0, 1.0] # Blue

        colors_np = np.tile(particle_color_default, (model.particle_count, 1)).astype(np.float32)
        for idx in self._fixed_particle_ids:
            colors_np[idx] = particle_color_fixed
        self.particle_debug_colors = wp.array(colors_np, dtype=wp.vec3)
        self.particle_debug_radii = wp.full(model.particle_count, debug_radius, dtype=wp.float32)


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
        self.viewer.log_points(
            name="/debug/fixed_particles",
            points=self.state_0.particle_q,
            radii=self.particle_debug_radii,
            colors=self.particle_debug_colors,
        )

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
