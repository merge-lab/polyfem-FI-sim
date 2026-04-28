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

class YamFORTESim:
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
        # TODO - what is robo
        self.model = self.create_model()

        # Initialize camera
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3([0.6, 0.4, 0.35]), -22, -124)
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
        if self.args["use_mujoco"]:
            self.solver_rigid = newton.solvers.SolverMuJoCo(self.model)

        # Preallocate variables for trajectory, control, and contacts
        self.state_now = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()

        # Initialize configuration of poke fixture
        # self.control.joint_target_pos = init_qs # TODO - figure out what robot init_qs should be

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
    
    ### ===================== Boilerplate-y functions =====================
    def create_model(self, init_qs=None):
        """
            Create the simulation scene
        """
        # Initialize the scene
        self.scene = newton.ModelBuilder()
        self.scene.default_particle_radius = self.sim_params.particle_radius
        self.create_scene() # This is where we define the scene creation logic

        ## ========= Make world replicas
        replicator = newton.ModelBuilder()
        n_worlds = 1
        replicator.replicate(self.scene, n_worlds, spacing=(0.2, 0.2, 0.0))
        replicator.color()
 
        ## ======== Finalize and export the model =======
        model = replicator.finalize()

        newton.eval_fk(model, model.joint_q, model.joint_qd, model)

        model.soft_contact_ke = self.sim_params.soft_contact_ke
        model.soft_contact_kd = self.sim_params.soft_contact_kd
        model.soft_contact_mu = self.sim_params.soft_contact_mu

        self.setup_debug_viz(model)
        return model

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

            if self.args["use_mujoco"]:
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
        self._control_robot()

        if self.graph:
            wp.capture_launch(self.graph)
            self.sim_time += self.sim_dt
        else:
            self.simulate()
        
        self._log_states()

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


    ### ===================== Specific simulation scene functions =====================
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--use_mujoco", action="store_true", default=False)
        parser.add_argument("--debug_particles", action="store_true", help="Show debug particles")
        parser.add_argument("--name", default="", type=str, help="Prefix to filename")
        return vars(parser.parse_args())
    
    def create_scene(self):
        ## ======== Add yam arm =======
        self.create_yam_arm(self.scene)

        # self.create_FORTE_loadcell(self.scene)

        # TODO - add breadboard

        ## ======== Add ground plane =======
        self.scene.add_ground_plane()

        self.create_thighpad(self.scene)

    
    def create_yam_arm(self, scene):
        yam_urdf_path = "../Assets/Yam-arm/yam_model_i2rt/yam_st_urdf_with_linear_gripper_gilbert.urdf"
        scene.add_urdf(
            yam_urdf_path,
            xform=wp.transform((0, 0, 0.0), wp.quat_identity()),
            floating=False,
            scale=1.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )

    def create_FORTE_fingers(self, scene):
        pass

    def create_FORTE_loadcell(self, scene):
        pass

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

        print(f"k_mu: {self.sim_params.material_k_mu}, k_lambda: {self.sim_params.material_k_lambda}")

        quat_initial = wp.quat_from_axis_angle(wp.vec3([1, 0, 0]), np.pi/2)
        # quat_initial = wp.quat_identity()
        self.pad_start_particle_idx = len(scene.particle_q)
        scene.add_soft_mesh(
            pos         = wp.vec3(0.4, 0.0, 0.000),
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

    def setup_debug_viz(self, model):
        # Build per-particle debug color array: blue for fixed, gray for free

        pass
        # debug_radius = self.sim_params.particle_radius
        # particle_color_default = [0.6, 0.6, 0.6] # Gray
        # particle_color_fixed = [0.0, 0.0, 1.0] # Blue

        # colors_np = np.tile(particle_color_default, (model.particle_count, 1)).astype(np.float32)
        # for idx in self._fixed_particle_ids:
        #     colors_np[idx] = particle_color_fixed
        # self.particle_debug_colors = wp.array(colors_np, dtype=wp.vec3)
        # self.particle_debug_radii = wp.full(model.particle_count, debug_radius, dtype=wp.float32)


    def _control_robot(self):
        pass

    def _log_states(self):
        self.log_sim_times.append(self.sim_time)
        return

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
        return
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
    sim = YamFORTESim(viewer, False)
    sim.run()

    viewer.close()
