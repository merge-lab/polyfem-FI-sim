"""
Convert a gmsh tetrahedral mesh (.msh) and surface mesh (.obj/.stl) to a USD file.

The output USD scene has two prims:
  /Model    - surface mesh (UsdGeomMesh)
  /TetModel - tetrahedral mesh (UsdGeom.TetMesh)

Usage:
    python msh_to_usd.py <tet_mesh.msh> <surface_mesh.obj|stl> <output.usd>

Dependencies:
    pip install meshio pxr-usd   (or install via omniverse / isaac sim python env)
"""

import argparse
import sys
import numpy as np
import meshio
import trimesh
from pxr import Usd, UsdGeom, Vt, Gf


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_tet_mesh(msh_path: str):
    """
    Load a gmsh .msh file and return (points, tets).

    points : (N, 3) float array
    tets   : (M, 4) int array  – vertex indices for each tetrahedron
    """
    mesh = meshio.read(msh_path)

    # meshio stores elements by type; grab tetrahedra
    tets = mesh.cells_dict["tetra"]
    # tets = None
    # for cell_block in mesh.cells:
    #     if cell_block.type == "tetra":
    #         tets = cell_block.data
    #         break
    #     elif cell_block.type == "tetra10":          # quadratic tets – use corner nodes only
    #         tets = cell_block.data[:, :4]
    #         break

    if tets is None:
        raise ValueError(
            f"No tetrahedral elements found in '{msh_path}'. "
            "Make sure the mesh was generated with 3-D tetrahedra."
        )

    points = mesh.points.astype(np.float32)
    # if points.shape[1] == 2:
    #     # Promote 2-D mesh to 3-D (shouldn't happen for tets, but just in case)
    #     points = np.column_stack([points, np.zeros(len(points), dtype=np.float32)])

    print(f"[tet mesh]  {len(points)} vertices, {len(tets)} tetrahedra  ({msh_path})")
    return points, tets.astype(np.int32)


def load_surface_mesh(surf_path: str):
    """
    Load an .obj or .stl surface mesh and return (points, triangles).

    points    : (N, 3) float array
    triangles : (F, 3) int array
    """
    # trimesh handles OBJ files with split vertex/normal indices correctly,
    # unlike meshio which errors when len(vertices) != len(normals).
    mesh = trimesh.load(surf_path, force="mesh", process=False)

    points    = np.array(mesh.vertices, dtype=np.float32)
    triangles = np.array(mesh.faces,    dtype=np.int32)

    print(f"[surface]   {len(points)} vertices, {len(triangles)} triangles  ({surf_path})")
    return points, triangles


# ---------------------------------------------------------------------------
# USD writers
# ---------------------------------------------------------------------------

def write_surface_mesh(stage: Usd.Stage, prim_path: str, points: np.ndarray, triangles: np.ndarray):
    """Create a UsdGeomMesh prim at prim_path."""
    mesh = UsdGeom.Mesh.Define(stage, prim_path)

    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))

    # Every face has 3 vertices
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(triangles)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(triangles.flatten().tolist()))

    mesh.GetOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    print(f"[USD]  wrote surface mesh  -> {prim_path}")
    return mesh


def write_tet_mesh(stage: Usd.Stage, prim_path: str, points: np.ndarray, tets: np.ndarray):
    """Create a UsdGeom.TetMesh prim at prim_path."""
    tet_mesh = UsdGeom.TetMesh.Define(stage, prim_path)

    tet_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))

    tet_mesh.GetTetVertexIndicesAttr().Set(
        Vt.Vec4iArray([Gf.Vec4i(int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in tets])
    )

    # Derive the boundary surface from the tet topology
    surface_faces = UsdGeom.TetMesh.ComputeSurfaceFaces(tet_mesh)
    tet_mesh.GetSurfaceFaceVertexIndicesAttr().Set(surface_faces)

    tet_mesh.GetOrientationAttr().Set(UsdGeom.Tokens.rightHanded)

    print(f"[USD]  wrote tet mesh      -> {prim_path}")
    return tet_mesh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a gmsh .msh tet mesh + surface mesh to a USD file."
    )
    parser.add_argument("tet_mesh",     help="Input gmsh .msh file (tetrahedral mesh)")
    parser.add_argument("surface_mesh", help="Input .obj or .stl surface mesh")
    parser.add_argument("output_usd",   help="Output .usd / .usda / .usdc file")
    args = parser.parse_args()

    # -- load inputs ---------------------------------------------------------
    tet_points, tets       = load_tet_mesh(args.tet_mesh)
    surf_points, triangles = load_surface_mesh(args.surface_mesh)

    # -- build USD stage -----------------------------------------------------
    stage = Usd.Stage.CreateNew(args.output_usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    write_surface_mesh(stage, "/Model",    surf_points, triangles)
    write_tet_mesh(stage,     "/TetModel", tet_points,  tets)

    stage.Save()
    print(f"\nSaved USD scene -> {args.output_usd}")


if __name__ == "__main__":
    main()
