# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# **Sag issue demo - fingers**
# Compare the effect of internal channels on the sagging of a mesh
# The FI sensorized simple-finger mesh has 65k tets, and sags heavily in VBD. We have seen in the
# sim_sag_issue_demo_bars.py script that a bar with 100k tets hardly sags with 50
# VBD solve iterations.
# 
# Therefore, we need to check whether the presence of air channels (and the small tet elements they necessitate)
# uniquely contributes towards sagging.
#
# In this example script, we compare the behavior of a simple finger with air channels, versus a near identical
# but solid mesh, meshed with a similar tet count.
#
# This script can be re-run with different VBD solve iteration counts to examine how that affects behavior.
#
###########################################################################

import time

import warp as wp
import numpy as np
import pandas as pd

import newton
import newton.examples
from pxr import Usd


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
            # particle_enable_self_contact= False,
            particle_enable_self_contact=True, # When simulating the small scale bars, the large sag deformations causes element inversion if there's no particle self contact.
            particle_self_contact_radius = 0.00005,
            particle_self_contact_margin = 0.00015,
            particle_enable_tile_solve=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        # self.viewer.set_camera(wp.vec3([4.5, 2, 1.36]), -1.0, 180.0)
        self.viewer.set_camera(wp.vec3([0.75, 0.0, 0.28]), -8, -180)

        self.capture()

        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="side")
        self.sim_start_time = time.time()

    def _load_surf_selection(self, file_surf_select):
        col_names = ["id_surf", "v_1", "v_2", "v_3"]
        df_surf_select = pd.read_csv(file_surf_select, names=col_names, sep="\s+")

        map_surf_select: map[int, np.array] = {}
        for surf_id_i in np.unique(df_surf_select["id_surf"]):
            surf_selections_i = df_surf_select[df_surf_select["id_surf"] == surf_id_i]
            map_surf_select[surf_id_i] = surf_selections_i[col_names[1:]].to_numpy(dtype=np.int64) # Drop the id_surf column, pick out unique vertex ids, and convert to np int array.
        
        return map_surf_select

    def add_softbodies(self, builder):
        # SCALE = 10.0
        SCALE = 1.0

        # Material parameters
        material_E = 1.35e6            # Young's modulus [N/m^2]
        material_nu = 0.45            # Poisson's ratio [unitless]
        # material_rho = 65 / 65754 * 1000**3 / 1000  # 65g / 65754mm^3, converted from g/mm3 to kg/m3, = 988.533 kg/m3
        k_mu = material_E / (2 * (1 + material_nu))
        k_lambda = material_E * material_nu / ((1 + material_nu) * (1 - 2 * material_nu))
        k_damp = 1e-6

        assetdir = "../../Assets/BasicFinger/mesh_l=0.05_e=2e-3/"
        modelpath_FORTE = assetdir + "model.usda"
        stage_FORTE = Usd.Stage.Open(modelpath_FORTE)
        prim_FORTE = stage_FORTE.GetPrimAtPath("/root/Model/TetMesh")
        tetmesh_FORTE = newton.TetMesh.create_from_usd(prim_FORTE)

        # Load surface selection as a map: surf_id -> vertex ids
        file_surf_select = assetdir + "surface_selections.txt"
        map_selected_surfs_FI_finger = self._load_surf_selection(file_surf_select)

        # quat_l_finger = wp.quat_from_axis_angle(wp.vec3([1, 0, 0]), 0.)
        r_fingercntr_finger = wp.vec3(0.016, 0, 0)
        quat_FI_finger = wp.quat_from_euler(wp.vec3(np.pi, 0, -np.pi/2), 0, 1, 2)
        posn_FI_finger = wp.vec3(0.203, -0.007, 0.17) # Btwn screw hole posn should be like <0.21, -0.02, 0.18> # TODO - where is origin of finger mesh?
        self.FI_finger_start_idx = len(builder.particle_q)
        builder.add_soft_mesh(
            pos         = posn_FI_finger * SCALE,
            rot         = quat_FI_finger,
            scale       = SCALE,
            vel         = wp.vec3(0.0, 0.0, 0.0),
            mesh        = tetmesh_FORTE,
            density     = 1e3,
            k_mu        = k_mu,
            k_lambda    = k_lambda,
            k_damp      = k_damp,
            tri_ke      = 0.0,
            tri_ka      = 0.0,
            tri_kd      = 0.0,
            tri_drag    = 0.0,
            tri_lift    = 0.0,
        )
        
        # Apply dirichlet BC's for the FI finger
        id_fixed = 6
        vert_ids_fix = np.unique(map_selected_surfs_FI_finger[id_fixed])
        self._fixed_particle_ids = []
        for vert_id in vert_ids_fix:
            global_id = self.FI_finger_start_idx + vert_id
            builder.particle_mass[global_id] = 0
            builder.particle_flags[global_id] = builder.particle_flags[global_id] & ~newton.ParticleFlags.ACTIVE
            self._fixed_particle_ids.append(global_id) # For debug viz


        # Now add the channelless finger
        assetdir = "../../Assets/BasicFinger-Channelless/mesh_l=0.018_e=2e-3/"
        modelpath_FORTE = assetdir + "model.usda"
        stage_FORTE = Usd.Stage.Open(modelpath_FORTE)
        prim_FORTE = stage_FORTE.GetPrimAtPath("/root/Model/TetMesh")
        tetmesh_FORTE = newton.TetMesh.create_from_usd(prim_FORTE)
        file_surf_select = assetdir + "surface_selections.txt"
        map_selected_surfs_FI_finger = self._load_surf_selection(file_surf_select)

        # quat_channelless_finger = wp.quat_from_axis_angle(wp.vec3([0, 0, 1]), wp.PI/2)
        posn_channelless_finger = wp.vec3(0.203, 0.05, 0.21) # Inter-screw-hole posn should be like <0.21, 0.02, 0.18
        self.channelless_finger_start_idx = len(builder.particle_q)
        builder.add_soft_mesh(
            pos         = posn_channelless_finger * SCALE,
            # rot         = quat_channelless_finger,
            rot         = quat_FI_finger,
            scale       = SCALE,
            vel         = wp.vec3(0.0, 0.0, 0.0),
            mesh        = tetmesh_FORTE,
            density     = 1e3,
            k_mu        = k_mu,
            k_lambda    = k_lambda,
            k_damp      = k_damp,
            tri_ke      = 0.0,
            tri_ka      = 0.0,
            tri_kd      = 0.0,
            tri_drag    = 0.0,
            tri_lift    = 0.0,
        )

        # Apply dirichlet BCs for the channelless finger
        id_fixed = 6
        vert_ids_fix = np.unique(map_selected_surfs_FI_finger[id_fixed])
        self._fixed_particle_ids = []
        for vert_id in vert_ids_fix:
            global_id = self.channelless_finger_start_idx + vert_id
            builder.particle_mass[global_id] = 0
            builder.particle_flags[global_id] = builder.particle_flags[global_id] & ~newton.ParticleFlags.ACTIVE
            self._fixed_particle_ids.append(global_id) # For debug viz


        # Now add the scaled up channelless finger
        # assetdir = "../../Assets/BasicFinger-Channelless-10x/mesh_l=0.018_e=2e-3/"
        SCALE = 10.0
        assetdir = "../../Assets/BasicFinger-Channelless/mesh_l=0.018_e=2e-3/"
        modelpath_FORTE = assetdir + "model.usda"
        stage_FORTE = Usd.Stage.Open(modelpath_FORTE)
        prim_FORTE = stage_FORTE.GetPrimAtPath("/root/Model/TetMesh")
        tetmesh_FORTE = newton.TetMesh.create_from_usd(prim_FORTE)
        file_surf_select = assetdir + "surface_selections.txt"
        map_selected_surfs_FI_finger = self._load_surf_selection(file_surf_select)

        # quat_channelless_finger = wp.quat_from_axis_angle(wp.vec3([0, 0, 1]), wp.PI/2)
        posn_channelless_finger = wp.vec3(0.203, 0.05, 1) # Inter-screw-hole posn should be like <0.21, 0.02, 0.18
        self.channelless_finger_start_idx = len(builder.particle_q)
        builder.add_soft_mesh(
            pos         = posn_channelless_finger,
            # rot         = quat_channelless_finger,
            rot         = quat_FI_finger,
            scale       = SCALE,
            vel         = wp.vec3(0.0, 0.0, 0.0),
            mesh        = tetmesh_FORTE,
            density     = 1e3,
            k_mu        = k_mu,
            k_lambda    = k_lambda,
            k_damp      = k_damp,
            tri_ke      = 0.0,
            tri_ka      = 0.0,
            tri_kd      = 0.0,
            tri_drag    = 0.0,
            tri_lift    = 0.0,
        )

        # Apply dirichlet BCs for the channelless finger
        id_fixed = 6
        vert_ids_fix = np.unique(map_selected_surfs_FI_finger[id_fixed])
        self._fixed_particle_ids = []
        for vert_id in vert_ids_fix:
            global_id = self.channelless_finger_start_idx + vert_id
            builder.particle_mass[global_id] = 0
            builder.particle_flags[global_id] = builder.particle_flags[global_id] & ~newton.ParticleFlags.ACTIVE
            self._fixed_particle_ids.append(global_id) # For debug viz

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
