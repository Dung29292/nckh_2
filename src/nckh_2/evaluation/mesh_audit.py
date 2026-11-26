"""Stage 1 & 5: Comprehensive 3D mesh diagnostics, 6-side breakdown, reference deviation & visual audit.

Provides scene-agnostic quantitative metrics for arbitrary .obj meshes:
  - Topology health: vertices, triangles, spatially-welded connected components,
    non-manifold edges/vertices, degenerate triangles, watertightness, Euler characteristic.
  - Triangle quality: aspect ratio (AR), minimum interior angle, bad-triangle percentages.
  - 6-Side canonical breakdown (+Z Top, -Z Bottom, -X Left, +X Right, -Y Front, +Y Back):
    per-side triangle count, surface area, density (tris/area), density CV across sides,
    and per-side surface deviation against an optional reference mesh or point cloud.
  - Visual inspection: 6-view shaded orthographic figure + per-side density/error bar charts.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from nckh_2.backends.alignment import compute_pose_skew
from nckh_2.backends.geometry import _spatial_connected_components
from nckh_2.backends.smoothing import _compute_face_geometry
from nckh_2.evaluation.metrics import (
    evaluate_detail_preservation,
)


SIDE_AXES: tuple[tuple[str, np.ndarray], ...] = (
    ("top_+Z", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
    ("bottom_-Z", np.array([0.0, 0.0, -1.0], dtype=np.float64)),
    ("left_-X", np.array([-1.0, 0.0, 0.0], dtype=np.float64)),
    ("right_+X", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
    ("front_-Y", np.array([0.0, -1.0, 0.0], dtype=np.float64)),
    ("back_+Y", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
)


@dataclasses.dataclass
class SideMetrics:
    side: str
    triangle_count: int
    triangle_share_pct: float
    surface_area: float
    area_share_pct: float
    density_tris_per_area: float
    mean_aspect_ratio: float
    bad_ar_pct: float
    dev_mean: float | None = None
    dev_p95: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class MeshAuditReport:
    stage_label: str
    num_vertices: int
    num_triangles: int
    num_components: int
    non_manifold_edges: int
    non_manifold_vertices: int
    degenerate_triangles: int
    is_watertight: bool
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    bbox_extents: tuple[float, float, float]
    bbox_diag: float
    surface_area: float
    mean_aspect_ratio: float
    p95_aspect_ratio: float
    bad_aspect_ratio_pct: float
    min_angle_p05_deg: float
    bad_min_angle_pct: float
    density_cv_6sides: float
    max_axis_skew_deg: float
    obb_aabb_volume_ratio: float
    is_skewed: bool
    ref_dev_mean: float | None
    ref_dev_p95: float | None
    ref_dev_p99: float | None
    sides: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compute_triangle_quality(
    verts: np.ndarray,
    tris: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-triangle (aspect_ratio, min_angle_deg, areas) in vectorized NumPy.

    Aspect ratio AR = (a*b*c) / (8*(s-a)*(s-b)*(s-c)) normalized so equilateral = 1.0.
    Degenerate triangles receive AR = inf and min_angle = 0.0.
    """
    n_t = len(tris)
    if n_t == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    a = np.linalg.norm(e0, axis=1)
    b = np.linalg.norm(e1, axis=1)
    c = np.linalg.norm(e2, axis=1)

    cross = np.cross(e0, -e2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    s = 0.5 * (a + b + c)
    denom = 8.0 * np.maximum(s - a, 0.0) * np.maximum(s - b, 0.0) * np.maximum(s - c, 0.0)
    ar = np.where(denom > 1e-18, (a * b * c) / np.maximum(denom, 1e-18), np.inf)

    # Interior angles via law of cosines
    def _angle(u: np.ndarray, v: np.ndarray, lu: np.ndarray, lv: np.ndarray) -> np.ndarray:
        dot = np.sum(u * v, axis=1)
        cos_val = np.clip(dot / np.maximum(lu * lv, 1e-12), -1.0, 1.0)
        return np.degrees(np.arccos(cos_val))

    ang0 = _angle(e0, -e2, a, c)
    ang1 = _angle(-e0, e1, a, b)
    ang2 = np.maximum(0.0, 180.0 - ang0 - ang1)
    min_ang = np.minimum(np.minimum(ang0, ang1), ang2)
    min_ang = np.where(areas > 1e-14, min_ang, 0.0)

    return ar, min_ang, areas


def load_reference_points(ref_path: Path, n_samples: int = 60000) -> np.ndarray:
    """Load reference .obj/.ply mesh or point cloud into an (N, 3) surface point array."""
    suffix = ref_path.suffix.lower()
    if suffix in (".obj", ".stl", ".off", ".gltf", ".glb"):
        m = o3d.io.read_triangle_mesh(str(ref_path), enable_post_processing=False)
        if len(m.triangles) > 0:
            pc = m.sample_points_uniformly(number_of_points=n_samples)
            return np.asarray(pc.points, dtype=np.float64)
        return np.asarray(m.vertices, dtype=np.float64)
    pc = o3d.io.read_point_cloud(str(ref_path))
    pts = np.asarray(pc.points, dtype=np.float64)
    if len(pts) > n_samples:
        idx = np.linspace(0, len(pts) - 1, n_samples, dtype=np.int64)
        pts = pts[idx]
    return pts


def audit_mesh(
    mesh: o3d.geometry.TriangleMesh,
    stage_label: str = "input",
    ref_points: np.ndarray | None = None,
    aspect_ratio_bad_thr: float = 4.0,
    min_angle_bad_deg: float = 15.0,
) -> MeshAuditReport:
    """Run full topology, triangle-quality, 6-side density, and reference-deviation audit."""
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)
    if n_v == 0 or n_t == 0:
        zeros = (0.0, 0.0, 0.0)
        return MeshAuditReport(
            stage_label=stage_label,
            num_vertices=n_v,
            num_triangles=n_t,
            num_components=0,
            non_manifold_edges=0,
            non_manifold_vertices=0,
            degenerate_triangles=0,
            is_watertight=False,
            bbox_min=zeros,
            bbox_max=zeros,
            bbox_extents=zeros,
            bbox_diag=0.0,
            surface_area=0.0,
            mean_aspect_ratio=0.0,
            p95_aspect_ratio=0.0,
            bad_aspect_ratio_pct=0.0,
            min_angle_p05_deg=0.0,
            bad_min_angle_pct=0.0,
            density_cv_6sides=0.0,
            max_axis_skew_deg=0.0,
            obb_aabb_volume_ratio=1.0,
            is_skewed=False,
            ref_dev_mean=None,
            ref_dev_p95=None,
            ref_dev_p99=None,
            sides={},
        )

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)

    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    ext = bmax - bmin
    diag = float(np.linalg.norm(ext))

    _, tri_counts_comp, _ = _spatial_connected_components(verts, tris)
    num_comp = len(tri_counts_comp)

    nm_edges = len(mesh.get_non_manifold_edges(allow_boundary_edges=True))
    nm_verts = len(mesh.get_non_manifold_vertices())
    watertight = bool(mesh.is_watertight())

    ar, min_ang, areas = compute_triangle_quality(verts, tris)
    eps_area = max(1e-12, 1e-10 * diag * diag)
    degen_mask = (areas <= eps_area) | ~np.isfinite(ar)
    n_degen = int(np.sum(degen_mask))

    finite_ar = ar[np.isfinite(ar)]
    mean_ar = float(np.mean(finite_ar)) if len(finite_ar) > 0 else float("inf")
    p95_ar = float(np.percentile(finite_ar, 95)) if len(finite_ar) > 0 else float("inf")
    bad_ar_pct = 100.0 * float(np.mean(ar > aspect_ratio_bad_thr))
    min_ang_p05 = float(np.percentile(min_ang, 5))
    bad_ang_pct = 100.0 * float(np.mean(min_ang < min_angle_bad_deg))
    total_area = float(np.sum(areas))

    # Face centroids & normals for 6-side partitioning
    centroids, _, f_norms = _compute_face_geometry(verts, tris)

    # Reference distance per triangle centroid (if reference points supplied)
    tri_dev = None
    ref_mean = ref_p95 = ref_p99 = None
    if ref_points is not None and len(ref_points) > 0:
        tree = cKDTree(ref_points)
        tri_dev, _ = tree.query(centroids, k=1)
        ref_mean = round(float(np.mean(tri_dev)), 6)
        ref_p95 = round(float(np.percentile(tri_dev, 95)), 6)
        ref_p99 = round(float(np.percentile(tri_dev, 99)), 6)

    # Assign each triangle to its dominant canonical side (+Z, -Z, -X, +X, -Y, +Y)
    axes_mat = np.vstack([vec for _, vec in SIDE_AXES])  # (6, 3)
    dots = f_norms @ axes_mat.T  # (F, 6)
    side_idx = np.argmax(dots, axis=1)

    sides_dict: dict[str, dict[str, Any]] = {}
    densities: list[float] = []
    for s_i, (s_name, _) in enumerate(SIDE_AXES):
        smask = side_idx == s_i
        s_tris = int(np.sum(smask))
        s_area = float(np.sum(areas[smask])) if s_tris > 0 else 0.0
        s_dens = (s_tris / s_area) if s_area > 1e-12 else 0.0
        if s_tris > 0 and s_area > 1e-12:
            densities.append(s_dens)
        s_ar_vals = ar[smask]
        s_ar_fin = s_ar_vals[np.isfinite(s_ar_vals)]
        s_mean_ar = float(np.mean(s_ar_fin)) if len(s_ar_fin) > 0 else 0.0
        s_bad_ar = 100.0 * float(np.mean(s_ar_vals > aspect_ratio_bad_thr)) if s_tris > 0 else 0.0

        s_dev_mean = round(float(np.mean(tri_dev[smask])), 6) if (tri_dev is not None and s_tris > 0) else None
        s_dev_p95 = round(float(np.percentile(tri_dev[smask], 95)), 6) if (tri_dev is not None and s_tris > 0) else None

        sides_dict[s_name] = SideMetrics(
            side=s_name,
            triangle_count=s_tris,
            triangle_share_pct=round(100.0 * s_tris / max(n_t, 1), 2),
            surface_area=round(s_area, 5),
            area_share_pct=round(100.0 * s_area / max(total_area, 1e-12), 2),
            density_tris_per_area=round(s_dens, 2),
            mean_aspect_ratio=round(s_mean_ar, 3),
            bad_ar_pct=round(s_bad_ar, 2),
            dev_mean=s_dev_mean,
            dev_p95=s_dev_p95,
        ).to_dict()

    density_cv = (
        float(np.std(densities) / max(float(np.mean(densities)), 1e-12))
        if len(densities) >= 2
        else 0.0
    )
    pose_skew = compute_pose_skew(mesh)

    return MeshAuditReport(
        stage_label=stage_label,
        num_vertices=n_v,
        num_triangles=n_t,
        num_components=num_comp,
        non_manifold_edges=nm_edges,
        non_manifold_vertices=nm_verts,
        degenerate_triangles=n_degen,
        is_watertight=watertight,
        bbox_min=tuple(round(float(x), 5) for x in bmin),
        bbox_max=tuple(round(float(x), 5) for x in bmax),
        bbox_extents=tuple(round(float(x), 5) for x in ext),
        bbox_diag=round(diag, 5),
        surface_area=round(total_area, 5),
        mean_aspect_ratio=round(mean_ar, 4),
        p95_aspect_ratio=round(p95_ar, 4),
        bad_aspect_ratio_pct=round(bad_ar_pct, 2),
        min_angle_p05_deg=round(min_ang_p05, 2),
        bad_min_angle_pct=round(bad_ang_pct, 2),
        density_cv_6sides=round(density_cv, 4),
        max_axis_skew_deg=pose_skew.max_axis_skew_deg,
        obb_aabb_volume_ratio=pose_skew.obb_aabb_volume_ratio,
        is_skewed=pose_skew.is_skewed,
        ref_dev_mean=ref_mean,
        ref_dev_p95=ref_p95,
        ref_dev_p99=ref_p99,
        sides=sides_dict,
    )


def print_audit_summary(report: MeshAuditReport) -> None:
    """Print a compact human-readable audit summary to stdout."""
    print(f"\n=== Mesh Audit [{report.stage_label}] ===")
    print(
        f"  Topology : verts={report.num_vertices:,} | tris={report.num_triangles:,} | "
        f"comps={report.num_components} | non_manifold_e={report.non_manifold_edges} | "
        f"degen={report.degenerate_triangles} | watertight={report.is_watertight}"
    )
    print(
        f"  Quality  : mean_AR={report.mean_aspect_ratio:.2f} | P95_AR={report.p95_aspect_ratio:.2f} | "
        f"bad_AR(>4)={report.bad_aspect_ratio_pct:.2f}% | P05_min_angle={report.min_angle_p05_deg:.1f} deg | "
        f"6-side density CV={report.density_cv_6sides:.3f}"
    )
    print(
        f"  Geometry : extents={report.bbox_extents} | diag={report.bbox_diag:.4f} | "
        f"area={report.surface_area:.4f} | skew={report.max_axis_skew_deg:.1f} deg "
        f"(OBB/AABB={report.obb_aabb_volume_ratio:.2f}, skewed={report.is_skewed})"
    )
    if report.ref_dev_mean is not None:
        print(
            f"  Ref Dev  : mean={report.ref_dev_mean:.5f} | P95={report.ref_dev_p95:.5f} | P99={report.ref_dev_p99:.5f}"
        )
    print("  6-Side Breakdown:")
    for s_name, sd in report.sides.items():
        dev_str = f" | dev_P95={sd['dev_p95']:.5f}" if sd["dev_p95"] is not None else ""
        print(
            f"    {s_name:10s}: tris={sd['triangle_count']:7,} ({sd['triangle_share_pct']:5.1f}%) | "
            f"area={sd['surface_area']:7.3f} | dens={sd['density_tris_per_area']:8.1f} | "
            f"bad_AR={sd['bad_ar_pct']:5.1f}%{dev_str}"
        )


def save_audit_comparison_figure(
    before_mesh: o3d.geometry.TriangleMesh,
    after_mesh: o3d.geometry.TriangleMesh,
    before_rep: MeshAuditReport,
    after_rep: MeshAuditReport,
    out_png: Path,
    dpi: int = 150,
) -> None:
    """Save side-by-side shaded inspection + 6-side density & quality comparison chart."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 10))

    def _plot_shaded(ax, mesh: o3d.geometry.TriangleMesh, title: str, elev: float, azim: float):
        if len(mesh.triangles) == 0:
            ax.set_title(f"{title} (empty)")
            return
        m_vis = (
            mesh.simplify_quadric_decimation(target_number_of_triangles=20000)
            if len(mesh.triangles) > 20000
            else mesh
        )
        v = np.asarray(m_vis.vertices)
        t = np.asarray(m_vis.triangles)
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=t, color="lightgray", edgecolor="none", shade=True)
        ax.view_init(elev=elev, azim=azim)
        ext = np.maximum(np.ptp(v, axis=0), 1e-4)
        ax.set_box_aspect(ext)
        ax.set_title(title)

    _plot_shaded(fig.add_subplot(231, projection="3d"), before_mesh, f"Before: {before_rep.stage_label} (Iso)", 30, -55)
    _plot_shaded(fig.add_subplot(232, projection="3d"), after_mesh, f"After: {after_rep.stage_label} (Iso)", 30, -55)
    _plot_shaded(fig.add_subplot(233, projection="3d"), after_mesh, f"After: {after_rep.stage_label} (Top-Back)", 45, 125)

    # Bar chart 1: 6-side triangle density before vs after
    side_names = [s for s, _ in SIDE_AXES]
    x = np.arange(len(side_names))
    w = 0.36
    ax_d = fig.add_subplot(234)
    d_before = [before_rep.sides.get(s, {}).get("density_tris_per_area", 0.0) for s in side_names]
    d_after = [after_rep.sides.get(s, {}).get("density_tris_per_area", 0.0) for s in side_names]
    ax_d.bar(x - w / 2, d_before, w, label=f"Before (CV={before_rep.density_cv_6sides:.3f})", color="#64748b")
    ax_d.bar(x + w / 2, d_after, w, label=f"After (CV={after_rep.density_cv_6sides:.3f})", color="#2563eb")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(side_names, rotation=25, ha="right")
    ax_d.set_ylabel("Triangles / Area")
    ax_d.set_title("6-Side Triangle Density Balance")
    ax_d.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_d.legend()

    # Bar chart 2: 6-side bad aspect ratio % before vs after
    ax_q = fig.add_subplot(235)
    q_before = [before_rep.sides.get(s, {}).get("bad_ar_pct", 0.0) for s in side_names]
    q_after = [after_rep.sides.get(s, {}).get("bad_ar_pct", 0.0) for s in side_names]
    ax_q.bar(x - w / 2, q_before, w, label=f"Before ({before_rep.bad_aspect_ratio_pct:.1f}%)", color="#dc2626")
    ax_q.bar(x + w / 2, q_after, w, label=f"After ({after_rep.bad_aspect_ratio_pct:.1f}%)", color="#16a34a")
    ax_q.set_xticks(x)
    ax_q.set_xticklabels(side_names, rotation=25, ha="right")
    ax_q.set_ylabel("Bad Triangles (AR > 4) %")
    ax_q.set_title("6-Side Jagged/Sliver Triangle Ratio")
    ax_q.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_q.legend()

    # Bar chart 3: 6-side P95 deviation vs reference (if available)
    ax_e = fig.add_subplot(236)
    e_before = [before_rep.sides.get(s, {}).get("dev_p95") or 0.0 for s in side_names]
    e_after = [after_rep.sides.get(s, {}).get("dev_p95") or 0.0 for s in side_names]
    ax_e.bar(x - w / 2, e_before, w, label="Before P95 Dev", color="#f59e0b")
    ax_e.bar(x + w / 2, e_after, w, label="After P95 Dev", color="#0ea5e9")
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(side_names, rotation=25, ha="right")
    ax_e.set_ylabel("P95 Surface Deviation")
    ax_e.set_title("6-Side Reference Deviation (P95)")
    ax_e.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_e.legend()

    plt.tight_layout()
    fig.savefig(str(out_png), dpi=dpi)
    plt.close(fig)


def save_stage_ledger(
    reports: list[MeshAuditReport],
    out_json: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist structured JSON ledger of all stage diagnostics."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "stages": [r.to_dict() for r in reports],
    }
    if extra:
        payload["summary"] = extra
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
