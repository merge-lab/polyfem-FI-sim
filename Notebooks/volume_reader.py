import pyvista as pv
import pandas as pd
import meshio
import numpy as np

def get_mb_at_time(reader, t):
    reader.set_active_time_value(t)
    multiblock = reader.read()[0]
    return multiblock

def get_volume_at_time(reader, t, id_interior):
    # t must be an element within reader.time_values
    multiblock = get_mb_at_time(reader, t)

    smesh_t = multiblock[1][0]

    # Todo - we only need to compute this mesh once, could cache for efficiency if needed
    mask_mesh_interior = smesh_t["sidesets"] == id_interior
    i_mask_interior = np.nonzero(mask_mesh_interior)
    
    smesh_warped_t = smesh_t.warp_by_vector("solution")
    mesh_interior_t = smesh_warped_t.extract_points(i_mask_interior, include_cells=True)
    volume_t = mesh_interior_t.extract_surface().volume

    # mesh_interior_t.plot()

    return volume_t