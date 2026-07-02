from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd     # Used for mesh pre-processing

import newton
import warp as wp
from pxr import Usd

@dataclass
class SimParams:

    ### Solver parameters
    # Time resolution and solver iterations
    fps = 60
    sim_substeps = 10
    iterations = 5
    @property
    def frame_dt(self):
        return 1/self.fps
    @property
    def sim_dt(self):
        return self.frame_dt / self.sim_substeps

    # Solver particle contact radii
    particle_self_contact_radius = 0.0001
    particle_self_contact_margin = 0.0003
    # particle_radius = 0.00005 # 0.1mm diameter
    particle_radius = 0.0005 # 1mm diameter

    # SOlver contact behavior parameters
    soft_contact_kd = 1e-7      # Soft contact param #TODO better docs
    soft_contact_ke = 1e8       # Soft contact param
    soft_contact_mu = 2.5       # Soft contact param
    rigid_contact_k_start = 1.0e5       # For avbd rigid-rigid contacts
    rigid_avbd_beta = 1.0e8             # For avbd rigid-rigid contacts

    ### World parameters
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



def create_yam_arm(scene: newton.ModelBuilder, xform: wp.transform):
    yam_urdf_path = "../Assets/Yam-arm/yam_model_i2rt/yam_gilbert_no_finray.urdf"
    # yam_urdf_path = "../Assets/Yam-arm/yam_model_i2rt/yam_gilbert.urdf"
    scene.add_urdf(
        yam_urdf_path,
        xform=xform,
        floating=False,
        scale=1.0,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
    )


def create_pad(
        filepath: str,
        scene: newton.ModelBuilder, pos: wp.vec3, rot: wp.quat, 
        rho: float, k_mu: float, k_lambda: float, k_damp: float,
):
    """
        Load the soft pad and place it in the simulation scene
    """
    # Fetching thighpad asset using the USD ecosystem
    asset_dir = Path(filepath)
    modelpath = asset_dir / "model.usda"
    stage = Usd.Stage.Open(str(modelpath))
    prim = stage.GetPrimAtPath("/root/Model/TetMesh")
    tetmesh = newton.TetMesh.create_from_usd(prim)

    pad_start_particle_idx = len(scene.particle_q)
    quat_initial = wp.quat_from_axis_angle(wp.vec3([1, 0, 0]), np.pi/2) # Changes from y-up to z-up
    scene.add_soft_mesh(
        pos         = pos,
        rot         = rot * quat_initial,
        scale       = 1.0,
        vel         = wp.vec3(0.0, 0.0, 0.0),
        mesh        = tetmesh,
        density     = rho,
        k_mu        = k_mu,
        k_lambda    = k_lambda,
        k_damp      = k_damp,
        tri_ke      = 0.0,
        tri_ka      = 0.0,
        tri_kd      = 0.0,
        tri_drag    = 0.0,
        tri_lift    = 0.0,
    )


    file_surf_select = asset_dir / "surface_selections.txt"
    dict_surf_select = load_surf_selection(str(file_surf_select))

    surf_ids_fix = [0] # channel id for bottom surface of thighpad
    fixed_particle_ids = fix_surfaces_in_mesh(
        scene, pad_start_particle_idx, dict_surf_select, surf_ids_fix
    )

    return fixed_particle_ids, dict_surf_select

def load_surf_selection(file_surf_select):
    """
        Load a surface selection CSV into a dictionary
        The dictionary maps: surf_id -> vert_ids where the verts are in the selected surface
        
        Args:
            file_surf_select: filename of surface selection csv. Follows Gjoka et al 2024's convention of <id, v_1, v_2, v3>
        
        Retruns:
            Dictionary (int -> list[int]) mapping surface ids to a list of ids of vertices that constitute the surface. 
    """

    col_names = ["id_surf", "v_1", "v_2", "v_3"]
    df_surf_select = pd.read_csv(file_surf_select, names=col_names, sep="\s+")

    map_surf_select: map[int, np.array] = {}
    for surf_id_i in np.unique(df_surf_select["id_surf"]):
        surf_selections_i = df_surf_select[df_surf_select["id_surf"] == surf_id_i]
        map_surf_select[surf_id_i] = surf_selections_i[col_names[1:]].to_numpy(dtype=np.int64) # Drop the id_surf column, pick out unique vertex ids, and convert to np int array.
    
    return map_surf_select

def fix_surfaces_in_mesh(scene: newton.ModelBuilder, 
    i_mesh_start_particle: int, map_surf_select: dict, i_surfs_to_fix: list[int]
):
    """
    Fixes the vertices in selected surfaces of a mesh. 

    Args:
        i_mesh_start_particle: index of the first particle of the mesh. This is typically calculated as
            len(scene.particle_q) prior to adding the given mesh.

        map_surf_select: surface selection dictionary created by load_surf_selection()

        i_surfs_to_fix: list of surface indices that we actually want to fix
           (since we may not want to fix all surfaces where a selection is defined)
        
    Returns:
        The list of global particle ids that have been fixed
    """
    # Get the complete list of ids to be fixed
    vert_ids_fix = [map_surf_select[id_fixed_i] for id_fixed_i in i_surfs_to_fix]
    vert_ids_fix = np.unique(vert_ids_fix)

    fixed_particle_ids = []
    for vert_id in vert_ids_fix:
        global_id = i_mesh_start_particle + vert_id
        scene.particle_mass[global_id] = 0
        scene.particle_flags[global_id] = scene.particle_flags[global_id] & ~newton.ParticleFlags.ACTIVE
        fixed_particle_ids.append(global_id) # For debug viz
    
    return fixed_particle_ids
    