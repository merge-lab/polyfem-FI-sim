#!/usr/bin/env python3
"""
Mesh surface viewer.

Loads a tetrahedral .msh file, extracts boundary triangles, colors them
by surface selection IDs (from surface_selections.txt), and provides an
interactive slice plane via ImGui sliders.

Usage:
    uv run python newton/view_surface_selection.py <mesh.msh> [-s selections.txt]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import meshio
import numpy as np
import pandas as pd
from imgui_bundle import ImVec2, ImVec4, hello_imgui, imgui, immapp, implot3d

from utils_funcs import _extract_boundary_tris

UNSELECTED_COLOR = ImVec4(0.55, 0.55, 0.55, 0.85)


# ---------------------------------------------------------------------------
# Mesh loading and boundary extraction
# ---------------------------------------------------------------------------


def _load_selections(sel_path: str) -> dict[frozenset, int]:
    """Parse surface_selections.txt → {frozenset(v0,v1,v2): selection_id}."""
    data = np.loadtxt(sel_path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Build a map of the surface ids of all selected vertices
    sel_map: dict[frozenset, int] = {}
    for row in data:
        id_i = int(row[0])
        verts_i = frozenset(row[1:4].tolist())
        sel_map[verts_i] = id_i
    return sel_map


def _make_colors(unique_ids: list[int]) -> dict[int, ImVec4]:
    """Assign a distinct HSV-derived color to each selection ID."""
    import colorsys
    n = max(len(unique_ids), 1)
    return {
        sid: ImVec4(*colorsys.hsv_to_rgb(i / n, 0.75, 0.92), 1.0)
        for i, sid in enumerate(sorted(unique_ids))
    }


# ---------------------------------------------------------------------------
# Per-group data container
# ---------------------------------------------------------------------------

class GroupData:
    """Pre-processes one selection group for fast per-frame slice + draw."""

    def __init__(self, faces: np.ndarray, verts: np.ndarray, all_points: list):
        # Compact vertex set: only vertices used by this group
        flat = faces.flatten()
        unique_vids = np.unique(flat)

        remap = np.empty(int(unique_vids.max()) + 1, dtype=np.int32)
        remap[unique_vids] = np.arange(len(unique_vids), dtype=np.int32)

        self.points = [all_points[int(v)] for v in unique_vids]
        self.local_faces = remap[faces]             # (N, 3) local indices
        self.face_verts = verts[faces]              # (N, 3, 3) positions
        self.full_idx: list[int] = self.local_faces.flatten().tolist()
        self.n_faces = len(faces)

    def get_mesh(self, plane: tuple[float, float, float, float] | None) -> "implot3d.Mesh | None":
        """Build an implot3d.Mesh, optionally filtered by slice plane (a,b,c,d)."""
        if plane is None:
            return implot3d.Mesh(points=self.points, idx=self.full_idx)

        a, b, c, d = plane
        # face_verts: (N, 3, 3)  →  dots: (N, 3)
        dots = self.face_verts @ np.array([a, b, c], dtype=np.float64)
        mask = np.all(dots >= d, axis=1)
        if not mask.any():
            return None
        idx = self.local_faces[mask].flatten().tolist()
        return implot3d.Mesh(points=self.points, idx=idx)

def parse_args():
    parser = argparse.ArgumentParser(
        description="View tet mesh boundary with surface selection coloring and slice plane."
    )
    parser.add_argument("msh_file", help="Path to .msh tetrahedral mesh")
    parser.add_argument(
        "--selections", "-s",
        help="Path to surface_selections.txt (auto-detected from mesh directory if omitted)",
    )
    args = parser.parse_args()

    return args

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ### --- Parse arguments ---
    args = parse_args()
    msh_path = Path(args.msh_file)
    sel_path = args.selections

    # If selection path is not explicitly provided, assume selections are in same dir as mesh
    if sel_path is None:
        candidate = msh_path.parent / "surface_selections.txt"
        if candidate.exists():
            sel_path = str(candidate)

    ### --- Load mesh ---
    print(f"Loading {msh_path} …")
    mesh = meshio.read(str(msh_path))
    verts, boundary_tris = _extract_boundary_tris(mesh)
    print(f"  {len(verts)} vertices, {len(boundary_tris)} boundary triangles")

    # Pre-build implot3d Point list once (shared across all GroupData objects)
    all_points = [
        implot3d.Point(float(x), float(y), float(z))
        for x, y, z in verts
    ]

    ### --- Assign faces to groups ---
    group_faces_raw: dict[int, list] = defaultdict(list)
    unselected_raw: list = []

    if sel_path:
        sel_map = _load_selections(sel_path)
        for face in boundary_tris:
            sid = sel_map.get(frozenset(face.tolist()))
            if sid is not None:
                group_faces_raw[sid].append(face)
            else:
                unselected_raw.append(face)
        print(f"  Selection groups: {sorted(group_faces_raw.keys())}")
        print(f"  Unselected boundary faces: {len(unselected_raw)}")
    else:
        unselected_raw = list(boundary_tris)

    group_ids = sorted(group_faces_raw.keys())
    colors = _make_colors(group_ids)

    groups: dict[int, GroupData] = {
        sid: GroupData(np.array(group_faces_raw[sid], dtype=np.int32), verts, all_points)
        for sid in group_ids
    }
    unsel_data: GroupData | None = (
        GroupData(np.array(unselected_raw, dtype=np.int32), verts, all_points)
        if unselected_raw else None
    )

    # --- Boundary vertices (unique vertices on the surface) ---
    bv_ids = np.unique(boundary_tris)          # sorted unique vertex indices
    bv_pos = verts[bv_ids]                     # (M, 3)  — constant, never rebuilt
    bv_xs = np.ascontiguousarray(bv_pos[:, 0])
    bv_ys = np.ascontiguousarray(bv_pos[:, 1])
    bv_zs = np.ascontiguousarray(bv_pos[:, 2])

    # --- Bounding box (used for initial view and equal-scale box) ---
    bb_min = verts.min(axis=0)
    bb_max = verts.max(axis=0)
    bb_center = (bb_min + bb_max) * 0.5
    bb_ranges = bb_max - bb_min
    bb_max_range = float(bb_ranges.max())
    # 110 % of each axis extent → 5 % padding on each side
    bb_half = bb_ranges * 0.55
    bb_box_scale = (bb_ranges / bb_max_range).tolist()  # for equal visual scale

    # --- Slice plane state ---
    # Precompute projection range for the initial normal (Y axis)
    _init_projs = verts[:, 1]

    class State:
        slice_enabled: bool = False
        a: float = 0.0
        b: float = 1.0
        c: float = 0.0
        d: float = float(_init_projs.min())
        d_min: float = float(_init_projs.min())
        d_max: float = float(_init_projs.max())
        show_vertices: bool = True

    state = State()

    def _update_d_range() -> None:
        normal = np.array([state.a, state.b, state.c])
        nlen = np.linalg.norm(normal)
        if nlen < 1e-9:
            state.d_min, state.d_max = 0.0, 1.0
        else:
            projs = verts @ normal
            state.d_min, state.d_max = float(projs.min()), float(projs.max())
        state.d = float(np.clip(state.d, state.d_min, state.d_max))

    def _active_plane() -> tuple | None:
        return (state.a, state.b, state.c, state.d) if state.slice_enabled else None

    # --- ImGui/implot3d GUI ---
    _style_applied = [False]

    def gui() -> None:
        if not _style_applied[0]:
            style = implot3d.get_style()
            style.plot_padding = ImVec2(0, 0)    # default ~10 — tightens outer margins
            style.label_padding = ImVec2(0, 0)   # default ~5  — pulls tick labels closer
            _style_applied[0] = True

        # Control panel (fixed left side)
        imgui.set_next_window_size(ImVec2(290, 0), imgui.Cond_.first_use_ever)
        imgui.set_next_window_pos(ImVec2(8, 8), imgui.Cond_.first_use_ever)
        imgui.begin("Controls##panel")

        imgui.separator_text("Slice Plane   ax + by + cz ≥ d")
        _, state.slice_enabled = imgui.checkbox("Enable slice", state.slice_enabled)

        if state.slice_enabled:
            ca, state.a = imgui.slider_float("a", state.a, -1.0, 1.0, "%.2f")
            cb, state.b = imgui.slider_float("b", state.b, -1.0, 1.0, "%.2f")
            cc, state.c = imgui.slider_float("c", state.c, -1.0, 1.0, "%.2f")
            if ca or cb or cc:
                _update_d_range()
            _, state.d = imgui.slider_float(
                "d", state.d, state.d_min, state.d_max, "%.4g"
            )

        imgui.separator_text("Legend")
        for sid in group_ids:
            imgui.color_button(
                f"##c{sid}", colors[sid],
                imgui.ColorEditFlags_.no_tooltip, ImVec2(14, 14)
            )
            imgui.same_line()
            imgui.text(f"Group {sid}  ({groups[sid].n_faces} faces)")

        if unsel_data is not None:
            imgui.color_button(
                "##cu", UNSELECTED_COLOR,
                imgui.ColorEditFlags_.no_tooltip, ImVec2(14, 14)
            )
            imgui.same_line()
            imgui.text(f"Unselected  ({unsel_data.n_faces} faces)")

        imgui.separator_text("Vertices")
        _, state.show_vertices = imgui.checkbox(
            f"Show boundary vertices  ({len(bv_ids)})", state.show_vertices
        )

        imgui.separator()
        imgui.text_disabled(f"{len(verts)} vertices · {len(boundary_tris)} tris")
        imgui.end()

        # 3-D viewport (fills remaining space)
        if implot3d.begin_plot("##mesh3d", ImVec2(-1.0, -1.0)):
            implot3d.setup_axes("X", "Y", "Z", implot3d.AxisFlags_.lock, implot3d.AxisFlags_.lock, implot3d.AxisFlags_.lock)
            implot3d.setup_axes_limits(
                bb_center[0] - bb_half[0], bb_center[0] + bb_half[0],
                bb_center[1] - bb_half[1], bb_center[1] + bb_half[1],
                bb_center[2] - bb_half[2], bb_center[2] + bb_half[2],
                implot3d.Cond_.once,
            )
            implot3d.setup_box_scale(*bb_box_scale)

            plane = _active_plane()
            for sid in group_ids:
                mesh = groups[sid].get_mesh(plane)
                if mesh is not None:
                    implot3d.set_next_fill_style(colors[sid])
                    implot3d.plot_mesh(
                        f"Group {sid}", mesh,
                        implot3d.MeshFlags_.no_legend  # legend is in our custom panel
                    )

            if unsel_data is not None:
                mesh = unsel_data.get_mesh(plane)
                if mesh is not None:
                    implot3d.set_next_fill_style(UNSELECTED_COLOR)
                    implot3d.plot_mesh(
                        "Unselected", mesh,
                        implot3d.MeshFlags_.no_legend
                    )

            if state.show_vertices:
                plane = _active_plane()
                if plane is not None:
                    a, b, c, d = plane
                    dots = bv_pos @ np.array([a, b, c])
                    mask = dots >= d
                    xs, ys, zs = bv_xs[mask], bv_ys[mask], bv_zs[mask]
                else:
                    xs, ys, zs = bv_xs, bv_ys, bv_zs
                if len(xs):
                    implot3d.set_next_marker_style(
                        implot3d.Marker_.circle, size=2.0,
                        fill=ImVec4(1.0, 1.0, 1.0, 0.8),
                        outline=ImVec4(0.0, 0.0, 0.0, 0.0),
                    )
                    implot3d.plot_scatter(
                        "##bv", xs, ys, zs,
                        implot3d.ScatterFlags_.no_legend,
                    )

            implot3d.end_plot()

    runner = hello_imgui.RunnerParams()
    runner.app_window_params.window_title = f"Surface Viewer — {msh_path.name}"
    runner.app_window_params.window_geometry.size = (1400, 900)
    runner.callbacks.show_gui = gui

    addons = immapp.AddOnsParams()
    addons.with_implot3d = True

    immapp.run(runner, addons)


if __name__ == "__main__":
    main()
