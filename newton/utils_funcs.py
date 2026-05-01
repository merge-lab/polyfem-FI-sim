from collections import defaultdict

import meshio
import numpy as np

def _extract_boundary_tris(mesh: meshio.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, boundary_triangles Nx3) from a meshio Mesh."""
    verts = mesh.points

    tets = None
    # Pull tets data from the appropriate data block
    for block in mesh.cells:
        if block.type == "tetra":
            tets = block.data
            break
    if tets is None:
        raise ValueError("Mesh contains neither triangles nor tetrahedra.")

    # Standard outward-facing face orderings for a Gmsh tetrahedron
    tet_face_slots = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]

    face_count: dict[tuple, int] = defaultdict(int)
    faces_ordered: dict[tuple, tuple] = {}

    for tet in tets:
        for fi, fj, fk in tet_face_slots:
            # For each face in the tet, store the global indices of its vertices as a sorted tuple
            # Then use the sorted tuple as a hash to track how many occurences each face has seen.
            face_idxs = (int(tet[fi]), int(tet[fj]), int(tet[fk]))
            face_idxs_sorted = tuple(sorted(face_idxs))
            face_count[face_idxs_sorted] += 1

            # Store correctly ordered version of each face for future reference 
            # (we'll need the correct orientation in the end!)
            if face_idxs_sorted not in faces_ordered:
                faces_ordered[face_idxs_sorted] = face_idxs

    # Boundary faces are those that only appear in one tet
    boundary = [faces_ordered[k] for k, cnt in face_count.items() if cnt == 1]
    return verts, np.array(boundary, dtype=np.int32)
