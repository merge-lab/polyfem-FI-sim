import argparse
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime
from pathlib import Path

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
    Compute the volume of a closed triangular mesh.

    Args:
        vertices: Vertex positions (3D coordinates), as a wp.array or numpy-like (N, 3) array.
        indices: Triangle indices (each triangle is defined by 3 vertex indices).

    Returns:
        The (unsigned) volume of the mesh.
    """

    indices = np.array(indices).flatten()
    num_tris = len(indices) // 3
    vol_warp = wp.zeros(1, dtype=float) # Preallocate output

    verts_np = vertices.numpy() if isinstance(vertices, wp.array) else np.asarray(vertices)
    verts_np = verts_np.reshape(-1, 3)
    # Center the mesh on its vertex mean before summing origin-tetrahedra:
    # their magnitudes grow with the mesh's distance from the origin while the
    # volume itself doesn't, so a far-from-origin mesh loses most of float32's
    # precision to cancellation. Volume is translation-invariant.
    verts_np = verts_np - verts_np.mean(axis=0, dtype=np.float64)

    wp_vertices = wp.array(verts_np, dtype=wp.vec3)
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
