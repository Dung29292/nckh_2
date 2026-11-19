"""Phase 5: Quantitative 3D reconstruction & detail-retention evaluation harness.

Measures:
  - Global & split Chamfer Distance (Observed vs Occluded regions)
  - Precision, Recall, and F-Score @ tau (1% and 2% of bounding diagonal)
  - Normal Consistency (NC)
  - Detail-Retention Metrics:
      * High-Curvature Chamfer (CD_edge) on top 10% discrete-curvature crease vertices
      * Curvature Preservation Ratio (CPR -> 1.0 ideal; < 0.6 indicates over-smoothing)
      * Flat-Panel Normal Variance (sigma2_flat -> lower means smoother flat panels)
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .detail_smooth import _build_face_adjacency, _compute_face_geometry, compute_discrete_curvature


@dataclasses.dataclass
class SurfaceMetrics:
    chamfer: float
    precision: float
    recall: float
    f_score_1pct: float
    f_score_2pct: float
    normal_consistency: float

    def to_dict(self) -> dict[str, float]:
        return {k: round(float(v), 5) for k, v in dataclasses.asdict(self).items()}


@dataclasses.dataclass
class DetailRetentionMetrics:
    cd_edge: float                # Chamfer error strictly on top-10% sharpest crease vertices
    curvature_preservation: float # CPR = mean(curv_after[edge]) / mean(curv_ref[edge])
    flat_normal_var_before: float # Noise variance on smooth/flat regions before smoothing
    flat_normal_var_after: float  # Noise variance on smooth/flat regions after smoothing
    noise_reduction_pct: float    # % reduction in flat-panel normal variance

    def to_dict(self) -> dict[str, float]:
        return {k: round(float(v), 5) for k, v in dataclasses.asdict(self).items()}


def _sample_surface(mesh: o3d.geometry.TriangleMesh, n_samples: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    m = o3d.geometry.TriangleMesh(mesh)
    m.compute_vertex_normals()
    m.compute_triangle_normals()
    pc = m.sample_points_uniformly(number_of_points=n_samples)
    pts = np.asarray(pc.points, dtype=np.float64)
    if not pc.has_normals():
        pc.estimate_normals()
    norms = np.asarray(pc.normals, dtype=np.float64)
    return pts, norms


def compute_surface_metrics(
    pts_pred: np.ndarray,
    norms_pred: np.ndarray,
    pts_gt: np.ndarray,
    norms_gt: np.ndarray,
    bbox_diag: float | None = None,
) -> SurfaceMetrics:
    """Compute Chamfer Distance, F-Score @ 1% & 2% diagonal, and Normal Consistency."""
    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return SurfaceMetrics(float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0)

    if bbox_diag is None:
        bbox_diag = float(np.linalg.norm(pts_gt.max(axis=0) - pts_gt.min(axis=0)))
    diag = max(bbox_diag, 1e-6)

    tree_gt = cKDTree(pts_gt)
    tree_pred = cKDTree(pts_pred)

    d_p2g, idx_p2g = tree_gt.query(pts_pred, k=1)
    d_g2p, idx_g2p = tree_pred.query(pts_gt, k=1)

    cd = 0.5 * (float(np.mean(d_p2g)) + float(np.mean(d_g2p)))

    def _fscore(tau: float) -> tuple[float, float, float]:
        prec = float(np.mean(d_p2g <= tau))
        rec = float(np.mean(d_g2p <= tau))
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 1e-9 else 0.0
        return prec, rec, f1

    prec1, rec1, f1_1 = _fscore(0.01 * diag)
    _, _, f1_2 = _fscore(0.02 * diag)

    # Normal Consistency (absolute cosine handles orientation-invariant comparison robustly)
    nc_p2g = float(np.mean(np.abs(np.sum(norms_pred * norms_gt[idx_p2g], axis=1))))
    nc_g2p = float(np.mean(np.abs(np.sum(norms_gt * norms_pred[idx_g2p], axis=1))))
    nc = 0.5 * (nc_p2g + nc_g2p)

    return SurfaceMetrics(
        chamfer=cd,
        precision=prec1,
        recall=rec1,
        f_score_1pct=f1_1,
        f_score_2pct=f1_2,
        normal_consistency=nc,
    )


def evaluate_reconstruction_split(
    mesh_pred: o3d.geometry.TriangleMesh,
    mesh_gt: o3d.geometry.TriangleMesh,
    occluded_fn: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 20000,
) -> dict[str, dict[str, float]]:
    """Evaluate predicted mesh against GT separately on All, Observed, and Occluded regions."""
    pts_p, n_p = _sample_surface(mesh_pred, n_samples)
    pts_g, n_g = _sample_surface(mesh_gt, n_samples)
    diag = float(np.linalg.norm(pts_g.max(axis=0) - pts_g.min(axis=0)))

    occ_g = occluded_fn(pts_g)
    occ_p = occluded_fn(pts_p)

    res_all = compute_surface_metrics(pts_p, n_p, pts_g, n_g, bbox_diag=diag)
    res_obs = compute_surface_metrics(pts_p[~occ_p], n_p[~occ_p], pts_g[~occ_g], n_g[~occ_g], bbox_diag=diag)
    res_occ = compute_surface_metrics(pts_p[occ_p], n_p[occ_p], pts_g[occ_g], n_g[occ_g], bbox_diag=diag)

    return {
        "all": res_all.to_dict(),
        "observed": res_obs.to_dict(),
        "occluded": res_occ.to_dict(),
    }


def _flat_panel_normal_variance(mesh: o3d.geometry.TriangleMesh, flat_face_mask: np.ndarray) -> float:
    """Measure area-weighted topological 1-ring normal variance across flat faces (lower = smoother)."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    if len(tris) == 0 or not np.any(flat_face_mask):
        return 0.0
    _, areas, fn = _compute_face_geometry(verts, tris)
    row, col = _build_face_adjacency(len(verts), tris)
    keep = flat_face_mask[row] & flat_face_mask[col] & (areas[row] > 1e-12) & (areas[col] > 1e-12)
    if not np.any(keep):
        return 0.0
    r, c = row[keep], col[keep]
    err = np.sum((fn[r] - fn[c]) ** 2, axis=1)
    w = areas[r] * areas[c]
    return float(np.sum(w * err) / max(float(np.sum(w)), 1e-12))


def evaluate_detail_preservation(
    mesh_noisy: o3d.geometry.TriangleMesh,
    mesh_smoothed: o3d.geometry.TriangleMesh,
    mesh_clean_gt: o3d.geometry.TriangleMesh | None = None,
) -> DetailRetentionMetrics:
    """Quantify sharpness retention (CPR & CD_edge) vs flat-panel noise removal."""
    ref_mesh = mesh_clean_gt if mesh_clean_gt is not None else mesh_noisy
    curv_ref = compute_discrete_curvature(ref_mesh)
    curv_after = compute_discrete_curvature(mesh_smoothed)

    v_ref = np.asarray(ref_mesh.vertices, dtype=np.float64)
    v_after = np.asarray(mesh_smoothed.vertices, dtype=np.float64)

    edge_thr = float(np.percentile(curv_ref, 90))
    edge_mask = curv_ref >= max(edge_thr, 1e-4)
    if not np.any(edge_mask):
        edge_mask = np.ones(len(v_ref), dtype=bool)

    # Map high-curvature reference vertices onto smoothed mesh (handles different vertex counts too)
    tree_after = cKDTree(v_after)
    d_edge, idx_edge = tree_after.query(v_ref[edge_mask], k=1)
    cd_edge = float(np.mean(d_edge))

    ref_edge_c = float(np.mean(curv_ref[edge_mask]))
    after_edge_c = float(np.mean(curv_after[idx_edge]))
    cpr = after_edge_c / max(ref_edge_c, 1e-8)

    # Flat-panel normal variance before vs after smoothing (on bottom 50% curvature faces)
    tris_n = np.asarray(mesh_noisy.triangles, dtype=np.int32)
    curv_noisy_v = compute_discrete_curvature(mesh_noisy)
    face_curv = curv_noisy_v[tris_n].mean(axis=1)
    flat_mask = face_curv <= float(np.percentile(face_curv, 50))

    var_before = _flat_panel_normal_variance(mesh_noisy, flat_mask)
    if len(mesh_smoothed.triangles) == len(tris_n):
        var_after = _flat_panel_normal_variance(mesh_smoothed, flat_mask)
    else:
        curv_sm_v = compute_discrete_curvature(mesh_smoothed)
        tris_s = np.asarray(mesh_smoothed.triangles, dtype=np.int32)
        flat_s = curv_sm_v[tris_s].mean(axis=1) <= float(np.percentile(curv_sm_v[tris_s].mean(axis=1), 50))
        var_after = _flat_panel_normal_variance(mesh_smoothed, flat_s)

    red_pct = 100.0 * (var_before - var_after) / max(var_before, 1e-12)
    return DetailRetentionMetrics(
        cd_edge=cd_edge,
        curvature_preservation=cpr,
        flat_normal_var_before=var_before,
        flat_normal_var_after=var_after,
        noise_reduction_pct=red_pct,
    )


def build_synthetic_detailed_box(subdiv_iters: int = 4) -> o3d.geometry.TriangleMesh:
    """Create a synthetic toolbox with sharp edges, protruding front latch & top handle for ablation tests."""
    body = o3d.geometry.TriangleMesh.create_box(width=2.0, height=1.3, depth=1.0)
    body.translate([-1.0, -0.65, 0.0])
    body = body.subdivide_midpoint(number_of_iterations=subdiv_iters)

    verts = np.asarray(body.vertices, dtype=np.float64).copy()
    # Sculpt protruding front latch on -Y face
    is_latch = (verts[:, 1] < -0.64) & (np.abs(verts[:, 0]) < 0.25) & (verts[:, 2] > 0.40) & (verts[:, 2] < 0.85)
    verts[is_latch, 1] -= 0.12
    # Sculpt raised top handle on +Z face
    is_handle = (verts[:, 2] > 0.99) & (np.abs(verts[:, 0]) < 0.45) & (np.abs(verts[:, 1]) < 0.14)
    verts[is_handle, 2] += 0.14
    # Sculpt lid overhang step around z > 0.72
    is_lid = (verts[:, 2] > 0.72) & ((np.abs(verts[:, 0]) > 0.99) | (np.abs(verts[:, 1]) > 0.64))
    verts[is_lid, 0] *= 1.04
    verts[is_lid, 1] *= 1.05

    body.vertices = o3d.utility.Vector3dVector(verts)
    body.compute_triangle_normals()
    body.compute_vertex_normals()
    return body
