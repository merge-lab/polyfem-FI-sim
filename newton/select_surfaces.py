import argparse

import yaml
import numpy as np
import pandas as pd
import meshio
import trimesh
import igl

from utils_funcs import _extract_boundary_tris


def parse_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="View tet mesh boundary with surface selection coloring and slice plane."
    )
    parser.add_argument("config", help="Path to surface selection config file")
    parser.add_argument("--incl", "-i", type=str, default="all", 
        help="'all' or 'any', to determine if counting a triangle " \
        "face as 'included' requires any or all vertices being bounded. Defaults to 'all'."
    )
    parser.add_argument("--thresh", "-t", type=float, default=0.1, help="Winding number " \
        "threshold for a point to be considered 'in' a mesh. Defaults to 0.1." \
    )
    parser.add_argument("--out", "-o", type=str, default="surface_selections.txt", help="Output file name")
    args = parser.parse_args()
    config = parse_config(args.config)

    return args, config

def _get_faces_in_selector(tetmesh_verts: np.array, boundary_tris: np.array, selection: dict, args):
    # Load surface mesh from selector obj file
    # selector_mesh = meshio.read(selection["mesh"])
    selector_mesh = trimesh.load_mesh(selection["mesh"])
    
    # CORE: use the fast winding number for soups algorithm implemented in the igl library
    # See this paper: https://www.dgp.toronto.edu/projects/fast-winding-numbers/fast-winding-numbers-for-soups-and-clouds-siggraph-2018-barill-et-al.pdf
    # Compute winding number of all tetmesh vertices w.r.t the selector mesh
    # In theory, winding numbers should be either 0 (not within mesh) or 1 (within mesh). In practice they are floats due to inverse trig errors
    wind_nums = igl.fast_winding_number(selector_mesh.vertices, selector_mesh.faces, tetmesh_verts)
    i_verts_in = np.nonzero(wind_nums > args.thresh)[0]
    face_verts_in_selection = np.isin(boundary_tris, i_verts_in) # Elementwise checks if face vert indices are in the set of bounded verts

    if args.incl == "any":
        mask_faces_selected = np.any(face_verts_in_selection, axis=1)
    elif args.incl == "all":
        mask_faces_selected = np.all(face_verts_in_selection, axis=1)
    else:
        raise ValueError("Invalid argument value for --incl")
    
    faces_selected = boundary_tris[mask_faces_selected, :]
    return faces_selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args, cfg = _parse_args()

    print(f"Selecting surfaces from {cfg['tetMesh']}")
    mesh = meshio.read(cfg["tetMesh"])
    verts, boundary_tris = _extract_boundary_tris(mesh)
    
    selections: dict[int, list] = {}
    dfs = []
    for selection in cfg["selections"]:
        id_selection = selection["surface_id"]
        selections[id_selection] = _get_faces_in_selector(verts, boundary_tris, selection, args)
        
        # Build dataframe for i-th selection
        # Format: <surf_id, id_v0, id_v1, id_v2>
        df_i = pd.DataFrame(selections[id_selection])
        df_i.insert(0, "id", id_selection)
        dfs.append(df_i)
    
    df_out = pd.concat(dfs)
    df_out.to_csv(args.out, header=False, index=False, sep=" ")

if __name__ == "__main__":
    main()