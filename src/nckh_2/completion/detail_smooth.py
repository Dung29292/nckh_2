"""Phase 1: Detail-preserving anisotropic mesh smoothing (Bilateral Normal Filtering).

Implements Zheng et al.'s Bilateral Normal Filtering + Confidence-Anchored Least-Squares
Vertex Update in vectorized NumPy/SciPy, plus optional PyMeshLab two-step feature smoothing
and high-frequency observed displacement transfer.

Unlike isotropic Taubin/Laplacian filters (which uniformly round 90-degree corners, latches,
and handles), bilateral normal filtering smooths flat/curved MVS scanline ripples while
halting smoothing across sharp geometric creases and observed visual edges.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.spatial import cKDTree


def _compute_face_geometry(verts: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return face centroids (F, 3), face areas (F,), and unit normals (F, 3)."""
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    dbl_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * dbl_area
    normals = np.zeros_like(cross)
    valid = dbl_area > 1e-12
    normals[valid] = cross[valid] / dbl_area[valid, None]
    return centroids, areas, normals


def _build_face_adjacency(num_verts: int, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return directed 1-ring face adjacency index arrays (row, col) for vertex-sharing faces."""
    num_faces = len(tris)
    v_idx = tris.ravel()
    f_idx = np.repeat(np.arange(num_faces, dtype=np.int32), 3)
    vf = sparse.coo_matrix(
        (np.ones(len(v_idx), dtype=np.int8), (v_idx, f_idx)),
        shape=(num_verts, num_faces),
    ).tocsr()
    ff = (vf.T @ vf).tocoo()
    mask = ff.row != ff.col
    return ff.row[mask].astype(np.int32), ff.col[mask].astype(np.int32)


def _mean_edge_length(verts: np.ndarray, tris: np.ndarray) -> float:
    e0 = np.linalg.norm(verts[tris[:, 1]] - verts[tris[:, 0]], axis=1)
    e1 = np.linalg.norm(verts[tris[:, 2]] - verts[tris[:, 1]], axis=1)
    e2 = np.linalg.norm(verts[tris[:, 0]] - verts[tris[:, 2]], axis=1)
    return float(np.mean(np.concatenate([e0, e1, e2])))


def compute_discrete_curvature(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Compute per-vertex discrete normal-variation curvature proxy in [0, 2].

    Measures the mean angular normal difference across incident faces at each vertex.
    Sharp edges, latches, and handle creases score high; flat/smooth panels score near 0.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    n_v = len(verts)
    if n_v == 0 or len(tris) == 0:
        return np.zeros(n_v, dtype=np.float64)

    _, _, f_norms = _compute_face_geometry(verts, tris)
    v_flat = tris.ravel()
    f_flat = np.repeat(np.arange(len(tris), dtype=np.int32), 3)

    # Mean vertex normal from incident faces
    v_norms = np.zeros((n_v, 3), dtype=np.float64)
    for ax in range(3):
        v_norms[:, ax] = np.bincount(v_flat, weights=f_norms[f_flat, ax], minlength=n_v)
    vn_len = np.linalg.norm(v_norms, axis=1, keepdims=True)
    v_norms = np.where(vn_len > 1e-12, v_norms / np.maximum(vn_len, 1e-12), 0.0)

    # Per-(vertex, face) normal deviation -> average per vertex
    diffs = np.linalg.norm(f_norms[f_flat] - v_norms[v_flat], axis=1)
    deg = np.maximum(np.bincount(v_flat, minlength=n_v).astype(np.float64), 1.0)
    curv = np.bincount(v_flat, weights=diffs, minlength=n_v) / deg
    return curv


def interpolate_point_confidence_to_mesh(
    mesh: o3d.geometry.TriangleMesh,
    ref_points: np.ndarray,
    ref_confidence: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """Interpolate per-point confidence values w in [0, 1] onto mesh vertices via k-NN."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) == 0 or len(ref_points) == 0:
        return np.ones(len(verts), dtype=np.float64)

    tree = cKDTree(ref_points)
    k_eff = min(k, len(ref_points))
    dists, idxs = tree.query(verts, k=k_eff)
    if k_eff == 1:
        return np.clip(ref_confidence[idxs], 0.0, 1.0).astype(np.float64)
    w = 1.0 / (dists + 1e-6)
    w /= np.sum(w, axis=1, keepdims=True)
    v_conf = np.sum(ref_confidence[idxs] * w, axis=1)
    return np.clip(v_conf, 0.0, 1.0).astype(np.float64)


def bilateral_normal_smooth(
    mesh: o3d.geometry.TriangleMesh,
    normal_iter: int = 6,
    vertex_iter: int = 20,
    sigma_s_scale: float = 1.5,
    sigma_r: float = 0.38,
    vertex_confidence: np.ndarray | None = None,
    vertex_edge_strength: np.ndarray | None = None,
    anchor_weight: float = 0.06,
    step_size: float = 0.85,
) -> o3d.geometry.TriangleMesh:
    """Apply detail-preserving Bilateral Normal Filtering + confidence-anchored vertex update.

    Args:
        mesh: Input Open3D triangle mesh (not mutated in place).
        normal_iter: Number of bilateral face-normal filtering passes.
        vertex_iter: Number of vertex-position alignment passes per stage.
        sigma_s_scale: Spatial Gaussian width as multiple of mean edge length.
        sigma_r: Normal-similarity Gaussian width (smaller = stricter crease preservation).
        vertex_confidence: Optional per-vertex confidence in [0, 1] (1.0 = observed MVS truth,
            < 0.75 = synthesized/occluded seam). Observed regions preserve sharper edges and
            receive stronger positional anchoring.
        vertex_edge_strength: Optional per-vertex RGB edge score in [0, 1] to further protect
            observed visual creases.
        anchor_weight: Fidelity pull toward original observed positions weighted by confidence.
        step_size: Relaxation factor for vertex projection onto filtered tangent planes.

    Returns:
        Denoised copy of ``mesh`` with preserved vertex colors and sharp geometric features.
    """
    out = o3d.geometry.TriangleMesh(mesh)
    verts_orig = np.asarray(out.vertices, dtype=np.float64).copy()
    tris = np.asarray(out.triangles, dtype=np.int32)
    n_v, n_f = len(verts_orig), len(tris)
    if n_v == 0 or n_f == 0:
        return out

    has_colors = out.has_vertex_colors() and len(out.vertex_colors) == n_v
    v_colors = np.asarray(out.vertex_colors, dtype=np.float64).copy() if has_colors else None

    v_conf = (
        np.clip(np.asarray(vertex_confidence, dtype=np.float64), 0.0, 1.0)
        if vertex_confidence is not None and len(vertex_confidence) == n_v
        else np.ones(n_v, dtype=np.float64)
    )
    v_edge = (
        np.clip(np.asarray(vertex_edge_strength, dtype=np.float64), 0.0, 1.0)
        if vertex_edge_strength is not None and len(vertex_edge_strength) == n_v
        else np.zeros(n_v, dtype=np.float64)
    )

    face_conf = v_conf[tris].mean(axis=1)
    face_edge = v_edge[tris].mean(axis=1)

    # Adaptive per-face range sigma: tighter on observed/high-edge faces, softer on synthesized seams
    sigma_r_face = np.full(n_f, float(sigma_r), dtype=np.float64)
    sigma_r_face *= np.where(face_conf >= 0.85, 0.85, 1.0 + 0.50 * (1.0 - face_conf))
    sigma_r_face *= np.clip(1.0 - 0.45 * face_edge, 0.45, 1.0)
    sigma_r_face = np.maximum(sigma_r_face, 0.08)

    centroids, areas, normals = _compute_face_geometry(verts_orig, tris)
    mean_edge = max(_mean_edge_length(verts_orig, tris), 1e-6)
    sigma_s = max(sigma_s_scale * mean_edge, 1e-6)

    row, col = _build_face_adjacency(n_v, tris)
    if len(row) == 0:
        return out

    dist_s2 = np.sum((centroids[row] - centroids[col]) ** 2, axis=1)
    w_spatial = areas[col] * np.exp(-dist_s2 / (2.0 * sigma_s * sigma_s))

    # Stage 1: Iterative Bilateral Face Normal Filtering
    n_filt = normals.copy()
    inv_2sr2 = 1.0 / (2.0 * sigma_r_face[row] ** 2)
    self_weight = (areas * 0.25)[:, None]

    for _ in range(max(0, int(normal_iter))):
        dist_n2 = np.sum((n_filt[row] - n_filt[col]) ** 2, axis=1)
        w = w_spatial * np.exp(-dist_n2 * inv_2sr2)

        n_next = np.zeros_like(n_filt)
        for ax in range(3):
            n_next[:, ax] = np.bincount(row, weights=w * n_filt[col, ax], minlength=n_f)
        n_next += self_weight * n_filt
        norms = np.linalg.norm(n_next, axis=1, keepdims=True)
        n_filt = np.where(norms > 1e-12, n_next / np.maximum(norms, 1e-12), n_filt)

    # Stage 2: Area-Weighted Least-Squares Vertex Update aligned to filtered face normals
    verts_cur = verts_orig.copy()
    v_flat = tris.ravel()
    f_flat = np.repeat(np.arange(n_f, dtype=np.int32), 3)
    # Ignore degenerate zero-area marching-cubes slivers so they do not destabilize neighbor vertices
    med_area = max(float(np.median(areas)), 1e-12)
    valid_face_w = np.where(areas > 1e-4 * med_area, areas, 0.0)
    w_face = valid_face_w[f_flat]
    area_sum = np.maximum(np.bincount(v_flat, weights=w_face, minlength=n_v), 1e-12)
    max_step = 0.30 * mean_edge

    # Crease-weighted anchor: locks sharp observed features while letting noisy flat panels relax
    curv0 = compute_discrete_curvature(out)
    curv_scale = max(float(np.percentile(curv0, 90)), 1e-4)
    crease_factor = np.clip(np.maximum(curv0 / curv_scale, v_edge), 0.0, 1.0)
    eff_anchor = (anchor_weight * v_conf * (0.10 + 0.90 * crease_factor))[:, None]

    for _ in range(max(0, int(vertex_iter))):
        c_cur = verts_cur[tris].mean(axis=1)
        diff = c_cur[f_flat] - verts_cur[v_flat]
        proj = np.sum(n_filt[f_flat] * diff, axis=1) * w_face

        disp = np.zeros_like(verts_cur)
        for ax in range(3):
            disp[:, ax] = np.bincount(v_flat, weights=proj * n_filt[f_flat, ax], minlength=n_v)
        disp /= area_sum[:, None]

        disp_len = np.linalg.norm(disp, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_step / np.maximum(disp_len, 1e-12))
        verts_cur = verts_cur + (step_size * scale) * disp

        if anchor_weight > 0.0:
            verts_cur += eff_anchor * (verts_orig - verts_cur)

    out.vertices = o3d.utility.Vector3dVector(verts_cur)
    out.remove_degenerate_triangles()
    out.compute_triangle_normals()
    out.compute_vertex_normals()
    if v_colors is not None and len(v_colors) == len(out.vertices):
        out.vertex_colors = o3d.utility.Vector3dVector(v_colors)
    return out


def transfer_observed_detail_residuals(
    smoothed_mesh: o3d.geometry.TriangleMesh,
    observed_points: np.ndarray,
    vertex_confidence: np.ndarray | None = None,
    strength: float = 0.55,
    max_disp_frac: float = 0.015,
) -> o3d.geometry.TriangleMesh:
    """Re-inject spatially coherent micro-details from observed MVS points onto smoothed mesh.

    Computes signed normal residuals strictly on high-confidence observed vertices, applies
    a 1-ring median filter to suppress isolated MVS noise spikes, and adds coherent relief
    (grooves, shallow ribs, latch edges) back along vertex normals.
    """
    out = o3d.geometry.TriangleMesh(smoothed_mesh)
    verts = np.asarray(out.vertices, dtype=np.float64).copy()
    tris = np.asarray(out.triangles, dtype=np.int32)
    n_v = len(verts)
    if n_v == 0 or len(tris) == 0 or len(observed_points) < 10:
        return out

    out.compute_vertex_normals()
    v_norms = np.asarray(out.vertex_normals, dtype=np.float64)

    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    max_disp = max_disp_frac * max(diag, 1e-4)

    tree = cKDTree(observed_points)
    dists, idxs = tree.query(verts, k=1)
    near_obs = observed_points[idxs]
    raw_res = np.sum((near_obs - verts) * v_norms, axis=1)

    # Zero out residual where vertex is far from observed points or has low confidence
    valid_obs = dists < (2.5 * max_disp)
    if vertex_confidence is not None and len(vertex_confidence) == n_v:
        valid_obs &= np.asarray(vertex_confidence) >= 0.85
    raw_res = np.where(valid_obs, np.clip(raw_res, -max_disp, max_disp), 0.0)

    # Reject isolated single-vertex spikes using 1-ring neighbor median check
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    adj = np.vstack([edges, edges[:, ::-1]])
    deg = np.maximum(np.bincount(adj[:, 0], minlength=n_v).astype(np.float64), 1.0)
    nbr_mean = np.bincount(adj[:, 0], weights=raw_res[adj[:, 1]], minlength=n_v) / deg
    # Keep only spatially coherent residual supported by neighbors (suppresses salt-and-pepper spikes)
    coherent_res = np.where(
        np.sign(raw_res) == np.sign(nbr_mean),
        0.5 * (raw_res + nbr_mean),
        0.25 * nbr_mean,
    )

    verts += (strength * coherent_res)[:, None] * v_norms
    out.vertices = o3d.utility.Vector3dVector(verts)
    out.compute_triangle_normals()
    out.compute_vertex_normals()
    return out


def pymeshlab_two_step_smooth(
    mesh: o3d.geometry.TriangleMesh,
    steps: int = 3,
    normal_thr_deg: float = 45.0,
    normal_steps: int = 12,
    vertex_steps: int = 6,
) -> o3d.geometry.TriangleMesh:
    """Feature-preserving Two-Step Smoothing backed by PyMeshLab (fallback to bilateral_normal)."""
    try:
        import pymeshlab
    except ImportError:
        return bilateral_normal_smooth(mesh)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    has_colors = mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(verts)
    v_cols = np.asarray(mesh.vertex_colors, dtype=np.float64) if has_colors else None

    ms = pymeshlab.MeshSet()
    if has_colors:
        rgba = np.column_stack([np.clip(v_cols, 0.0, 1.0), np.ones(len(verts))])
        pm = pymeshlab.Mesh(vertex_matrix=verts, face_matrix=tris, v_color_matrix=rgba)
    else:
        pm = pymeshlab.Mesh(vertex_matrix=verts, face_matrix=tris)
    ms.add_mesh(pm)

    ms.apply_coord_two_steps_smoothing(
        stepsmoothnum=int(steps),
        normalthr=float(normal_thr_deg),
        stepnormalnum=int(normal_steps),
        stepfitnum=int(vertex_steps),
    )
    res = ms.current_mesh()
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(res.vertex_matrix())
    out.triangles = o3d.utility.Vector3iVector(res.face_matrix())
    if has_colors and len(res.vertex_matrix()) == len(verts):
        out.vertex_colors = o3d.utility.Vector3dVector(v_cols)
    out.remove_degenerate_triangles()
    out.compute_triangle_normals()
    out.compute_vertex_normals()
    return out
