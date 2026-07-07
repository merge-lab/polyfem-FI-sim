"""Tests for compute_volume_mesh() against solids with closed-form volumes,
plus direct comparisons against trimesh's built-in volume.

Expected volumes come from two sources per test:
  - a closed-form expression for the (discrete) solid, and
  - trimesh's exact signed-volume oracle (mesh.volume) for the same mesh.
The warp kernel accumulates in float32, so comparisons use a relative
tolerance of 1e-5.

The `test_matches_trimesh_*` tests mirror the debug comparison in
sim_franka_poke_pad._calc_vols_and_pres(): both paths get the identical
float32 vertex array (like state.particle_q) and face indices, ours via
compute_volume_mesh() and the reference via trimesh.Trimesh(...).volume
with trimesh's default processing, exactly as the sim constructs it.
"""

import math

import numpy as np
import pytest
import trimesh
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

from utils_FI import compute_volume_mesh

REL_TOL = 1e-5


def assert_volume(mesh: trimesh.Trimesh, expected: float, rel: float = REL_TOL):
    assert mesh.is_watertight, "test fixture mesh must be watertight"
    computed = compute_volume_mesh(mesh.vertices, mesh.faces)
    assert computed == pytest.approx(expected, rel=rel)
    assert computed == pytest.approx(mesh.volume, rel=REL_TOL)


@pytest.mark.parametrize(
    "extents",
    [
        (1.0, 1.0, 1.0),
        (1.0, 2.0, 3.0),
        (0.05, 0.2, 0.01),
    ],
)
def test_box(extents):
    mesh = trimesh.creation.box(extents=extents)
    assert_volume(mesh, math.prod(extents))


def test_box_translated_from_origin():
    # All tetrahedra in the kernel share the origin as apex, so a mesh far
    # from the origin only gets the right volume if the signed parts cancel.
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    mesh.apply_translation([10.0, -7.0, 5.0])
    assert_volume(mesh, 6.0)


def test_box_rotated():
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    rotation = trimesh.transformations.rotation_matrix(
        angle=0.7, direction=[1.0, 2.0, -0.5], point=[0.3, 0.0, 1.0]
    )
    mesh.apply_transform(rotation)
    assert_volume(mesh, 6.0)


def test_cylinder():
    radius, height, sections = 0.5, 2.0, 64
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    # The discrete cylinder is an n-gon prism inscribed in the true cylinder.
    discrete = 0.5 * sections * radius**2 * math.sin(2.0 * math.pi / sections) * height
    assert_volume(mesh, discrete)
    # At 64 sections the analytic cylinder volume is within ~0.2%.
    assert compute_volume_mesh(mesh.vertices, mesh.faces) == pytest.approx(
        math.pi * radius**2 * height, rel=5e-3
    )


def test_icosphere():
    radius = 0.75
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    assert_volume(mesh, 4.0 / 3.0 * math.pi * radius**3, rel=1e-2)


def test_nonconvex_union_of_prisms():
    # Union of three overlapping rectangles extruded to a single watertight
    # non-convex prism: a 4x1 bar, a 1x4 bar, and a 1x3 bar overlapping them.
    height = 2.0
    footprint = unary_union(
        [
            shapely_box(0.0, 0.0, 4.0, 1.0),
            shapely_box(0.0, 0.0, 1.0, 4.0),
            shapely_box(1.5, 0.0, 2.5, 3.0),
        ]
    )
    assert footprint.area < footprint.convex_hull.area, "footprint must be non-convex"
    mesh = trimesh.creation.extrude_polygon(footprint, height=height)
    assert_volume(mesh, footprint.area * height)


def test_inward_winding():
    # Inverted faces make the signed sum negative; the function must still
    # return the positive volume.
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    mesh.invert()
    assert compute_volume_mesh(mesh.vertices, mesh.faces) == pytest.approx(6.0, rel=REL_TOL)


# ---------------------------------------------------------------------------
# Direct comparison against trimesh's built-in volume, sim-style
# ---------------------------------------------------------------------------


def sim_style_volumes(mesh: trimesh.Trimesh) -> tuple[float, float]:
    """Compute the volume both ways as sim_franka_poke_pad does: identical
    float32 vertices (particle_q is float32) and faces into each path."""
    verts32 = np.asarray(mesh.vertices, dtype=np.float32)
    ours = compute_volume_mesh(verts32, mesh.faces)
    reference = trimesh.Trimesh(vertices=verts32, faces=mesh.faces).volume
    return ours, reference


def channel_tube(dented: bool = False) -> trimesh.Trimesh:
    """Channel-scale tube (r=2mm, L=40mm, V~5e-7 m^3), optionally with a
    poke-like dent pushed into its side."""
    mesh = trimesh.creation.cylinder(radius=0.002, height=0.040, sections=48)
    if dented:
        verts = mesh.vertices.copy()
        bump = np.exp(-(verts[:, 2] ** 2) / (2 * 0.008**2))
        verts[:, 0] -= 0.0015 * bump * (verts[:, 0] > 0)
        mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)
    return mesh


@pytest.mark.parametrize("dented", [False, True], ids=["straight", "dented"])
def test_matches_trimesh_channel_at_origin(dented):
    ours, reference = sim_style_volumes(channel_tube(dented))
    assert ours == pytest.approx(reference, rel=REL_TOL)


@pytest.mark.parametrize(
    "offset",
    [(0.5, 0.3, -0.1), (2.0, 1.2, -0.4)],
    ids=["offset_0.5m", "offset_2m"],
)
@pytest.mark.parametrize("dented", [False, True], ids=["straight", "dented"])
def test_matches_trimesh_channel_offset_from_origin(offset, dented):
    # Regression test: compute_volume_mesh used to sum origin-tetrahedra of
    # the raw coordinates, so a ~5e-7 m^3 channel placed away from the world
    # origin lost most of float32's precision to cancellation (rel. error
    # ~2.6e-3 at 0.5m, ~50% at 2m vs trimesh's float64 integral). Centering
    # the vertices on their mean inside compute_volume_mesh fixed this.
    mesh = channel_tube(dented)
    mesh.apply_translation(offset)
    ours, reference = sim_style_volumes(mesh)
    assert ours == pytest.approx(reference, rel=REL_TOL)


def test_accepts_warp_array_vertices():
    # The sims pass state.particle_q, a warp GPU array of vec3f — not numpy.
    import warp as wp

    mesh = channel_tube(dented=True)
    mesh.apply_translation((2.0, 1.2, -0.4))
    verts32 = np.asarray(mesh.vertices, dtype=np.float32)
    reference = trimesh.Trimesh(vertices=verts32, faces=mesh.faces).volume
    ours = compute_volume_mesh(wp.array(verts32, dtype=wp.vec3), mesh.faces)
    assert ours == pytest.approx(reference, rel=REL_TOL)


def test_trimesh_volume_is_signed_ours_is_absolute():
    # For an inward-wound mesh trimesh.volume goes negative while
    # compute_volume_mesh returns the absolute volume — a sign the two can
    # "disagree" in the sim printout without any numerical error at all.
    mesh = trimesh.creation.box(extents=(0.02, 0.03, 0.04))
    mesh.invert()
    ours, reference = sim_style_volumes(mesh)
    assert reference < 0
    assert ours == pytest.approx(-reference, rel=REL_TOL)
