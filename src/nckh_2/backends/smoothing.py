"""Bilateral normal filtering, PyMeshLab two-step smoothing, va detail residual transfer."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.spatial import cKDTree


def _compute_face_geometry(verts: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    """Compute per-vertex discrete normal-variation curvature proxy in [0, 2]."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    n_v = len(verts)
    if n_v == 0 or len(tris) == 0:
        return np.zeros(n_v, dtype=np.float64)

    _, _, f_norms = _compute_face_geometry(verts, tris)
    v_flat = tris.ravel()
    f_flat = np.repeat(np.arange(len(tris), dtype=np.int32), 3)

    v_norms = np.zeros((n_v, 3), dtype=np.float64)
    for ax in range(3):
        v_norms[:, ax] = np.bincount(v_flat, weights=f_norms[f_flat, ax], minlength=n_v)
    vn_len = np.linalg.norm(v_norms, axis=1, keepdims=True)
    v_norms = np.where(vn_len > 1e-12, v_norms / np.maximum(vn_len, 1e-12), 0.0)

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
    """Apply detail-preserving Bilateral Normal Filtering + confidence-anchored vertex update."""
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

    verts_cur = verts_orig.copy()
    v_flat = tris.ravel()
    f_flat = np.repeat(np.arange(n_f, dtype=np.int32), 3)
    med_area = max(float(np.median(areas)), 1e-12)
    valid_face_w = np.where(areas > 1e-4 * med_area, areas, 0.0)
    w_face = valid_face_w[f_flat]
    area_sum = np.maximum(np.bincount(v_flat, weights=w_face, minlength=n_v), 1e-12)
    max_step = 0.30 * mean_edge

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
    out.compute_triangle_normals()
    out.compute_vertex_normals()
    if v_colors is not None and len(v_colors) == len(out.vertices):
        out.vertex_colors = o3d.utility.Vector3dVector(v_colors)
    if mesh.has_triangle_uvs() and len(mesh.triangle_uvs) == 3 * len(out.triangles):
        out.triangle_uvs = o3d.utility.Vector2dVector(np.asarray(mesh.triangle_uvs))
    if mesh.has_triangle_material_ids() and len(mesh.triangle_material_ids) == len(out.triangles):
        out.triangle_material_ids = o3d.utility.IntVector(list(mesh.triangle_material_ids))
    return out


def suppress_mesh_spikes(
    mesh: o3d.geometry.TriangleMesh,
    sigma_threshold: float = 3.5,
    pull_strength: float = 0.85,
) -> tuple[o3d.geometry.TriangleMesh, int]:
    """Triet tieu cac dinh gai nhon dot bien (isolated spikes / needle artifacts) bang loc 1-ring MAD.

    Chi dieu chinh cac dinh ngoai lai co do lech so voi tam lang gieng 1-ring vuot qua
    ``sigma_threshold * MAD`` (va >= 1.5x do dai canh trung binh cuc bo), giu nguyen 100%
    canh sac, goc vuong va hinh hoc hop le cua toan canh.
    """
    out = o3d.geometry.TriangleMesh(mesh)
    verts = np.asarray(out.vertices, dtype=np.float64).copy()
    tris = np.asarray(out.triangles, dtype=np.int32)
    n_v = len(verts)
    if n_v < 4 or len(tris) == 0:
        return out, 0

    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    adj = np.vstack([edges, edges[:, ::-1]])
    src, dst = adj[:, 0], adj[:, 1]

    deg = np.bincount(src, minlength=n_v).astype(np.float64)
    valid = deg >= 3.0
    deg_safe = np.maximum(deg, 1.0)

    # Tinh tam 1-ring cua moi dinh (vectorized)
    nbr_centroid = np.zeros_like(verts)
    for ax in range(3):
        nbr_centroid[:, ax] = np.bincount(src, weights=verts[dst, ax], minlength=n_v) / deg_safe

    # Do dai canh trung vi toan cuc de phan biet gai nhon thuc su voi nhieu vi mo
    edge_lens = np.linalg.norm(verts[src] - verts[dst], axis=1)
    med_edge = max(float(np.median(edge_lens)), 1e-9)

    # Do lech Laplacian tu dinh den tam 1-ring
    disp = nbr_centroid - verts
    disp_norm = np.linalg.norm(disp, axis=1)
    med_disp = float(np.median(disp_norm[valid])) if np.any(valid) else 0.0
    mad = float(np.median(np.abs(disp_norm[valid] - med_disp))) if np.any(valid) else 0.0
    robust_sigma = max(1.4826 * mad, 1e-9)

    # Chi xem la gai nhon khi do lech vuot xa phan phoi toan cuc VA lon hon 1.5x do dai canh trung vi
    global_thr = med_disp + float(sigma_threshold) * robust_sigma
    spike_mask = valid & (disp_norm > global_thr) & (disp_norm > 1.5 * med_edge)
    n_spikes = int(np.sum(spike_mask))

    if n_spikes > 0:
        alpha = float(np.clip(pull_strength, 0.0, 1.0))
        verts[spike_mask] += alpha * disp[spike_mask]
        out.vertices = o3d.utility.Vector3dVector(verts)
        out.compute_triangle_normals()
        out.compute_vertex_normals()

    return out, n_spikes


def transfer_observed_detail_residuals(
    smoothed_mesh: o3d.geometry.TriangleMesh,
    observed_points: np.ndarray,
    vertex_confidence: np.ndarray | None = None,
    strength: float = 0.55,
    max_disp_frac: float = 0.015,
) -> o3d.geometry.TriangleMesh:
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

    valid_obs = dists < (2.5 * max_disp)
    if vertex_confidence is not None and len(vertex_confidence) == n_v:
        valid_obs &= np.asarray(vertex_confidence) >= 0.85
    raw_res = np.where(valid_obs, np.clip(raw_res, -max_disp, max_disp), 0.0)

    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    adj = np.vstack([edges, edges[:, ::-1]])
    deg = np.maximum(np.bincount(adj[:, 0], minlength=n_v).astype(np.float64), 1.0)
    nbr_mean = np.bincount(adj[:, 0], weights=raw_res[adj[:, 1]], minlength=n_v) / deg
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


def taubin_volume_preserving_smooth(
    mesh: o3d.geometry.TriangleMesh,
    iterations: int = 10,
    lambda_pos: float = 0.35,
    mu_neg: float = -0.37,
) -> o3d.geometry.TriangleMesh:
    """Volume-preserving two-step alternating Taubin smoothing (Section 3.4 of nckh.pdf).

    Alternates a shrinking pass (lambda > 0) with an opposing inflation pass (mu < -lambda):
        Step 1: V^(t + 1/2) = V^(t)     + lambda * L_umbrella * V^(t)
        Step 2: V^(t + 1)   = V^(t+1/2) + mu     * L_umbrella * V^(t+1/2)
    Spectral transfer function f(k) = (1 - lambda*k)*(1 - mu*k) suppresses high-frequency
    Marching Cubes staircase noise while preserving global ergonomic volume without shrinkage.
    """
    if lambda_pos <= 0.0 or mu_neg >= -lambda_pos:
        raise ValueError(
            f"Taubin smoothing requires lambda > 0 and mu < -lambda, got lambda={lambda_pos}, mu={mu_neg}"
        )

    out = o3d.geometry.TriangleMesh(mesh)
    verts = np.asarray(out.vertices, dtype=np.float64).copy()
    tris = np.asarray(out.triangles, dtype=np.int32)
    n_v = len(verts)
    if iterations <= 0 or n_v == 0 or len(tris) == 0:
        return out

    has_colors = bool(out.has_vertex_colors() and len(out.vertex_colors) == n_v)
    v_colors = np.asarray(out.vertex_colors, dtype=np.float64).copy() if has_colors else None
    has_uvs = bool(out.has_triangle_uvs() and len(out.triangle_uvs) == 3 * len(tris))
    tri_uvs = np.asarray(out.triangle_uvs, dtype=np.float64).copy() if has_uvs else None
    has_mat = bool(out.has_triangle_material_ids() and len(out.triangle_material_ids) == len(tris))
    mat_ids = list(out.triangle_material_ids) if has_mat else None

    # Build sparse random-walk umbrella Laplacian operator L_rw = D^(-1) A - I
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    undir = np.vstack([edges, edges[:, ::-1]])
    max_v = n_v + 1
    keys = undir[:, 0].astype(np.int64) * max_v + undir[:, 1].astype(np.int64)
    _, uniq_idx = np.unique(keys, return_index=True)
    src, dst = undir[uniq_idx, 0], undir[uniq_idx, 1]

    adj = sparse.coo_matrix((np.ones(len(src), dtype=np.float64), (src, dst)), shape=(n_v, n_v)).tocsr()
    deg = np.maximum(np.asarray(adj.sum(axis=1)).ravel(), 1.0)
    inv_deg = sparse.diags(1.0 / deg, format="csr")
    lap_umbrella = inv_deg @ adj - sparse.eye(n_v, format="csr")

    lam = float(lambda_pos)
    mu = float(mu_neg)
    for _ in range(int(iterations)):
        # Pass 1: shrinking filter (lambda > 0)
        verts = verts + lam * (lap_umbrella @ verts)
        # Pass 2: anti-shrinking inflation filter (mu < -lambda)
        verts = verts + mu * (lap_umbrella @ verts)

    out.vertices = o3d.utility.Vector3dVector(verts)
    if v_colors is not None:
        out.vertex_colors = o3d.utility.Vector3dVector(v_colors)
    if tri_uvs is not None:
        out.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)
    if mat_ids is not None:
        out.triangle_material_ids = o3d.utility.IntVector(mat_ids)
    out.compute_triangle_normals()
    out.compute_vertex_normals()
    return out


def drift_monitored_smooth(
    mesh: o3d.geometry.TriangleMesh,
    mode: str = "taubin",
    batch_iters: int = 2,
    max_batches: int = 5,
    max_drift_frac: float = 0.004,
    taubin_lambda: float = 0.35,
    taubin_mu: float = -0.37,
    bilateral_sigma_s_scale: float = 1.5,
    bilateral_sigma_r: float = 0.25,
    bilateral_anchor_weight: float = 0.08,
    vertex_confidence: np.ndarray | None = None,
) -> tuple[o3d.geometry.TriangleMesh, dict[str, object]]:
    """Apply volume-preserving smoothing in small micro-batches while bounding P95 geometric drift.

    Stops early if P95 surface drift against the pre-smoothed input mesh exceeds
    ``max_drift_frac * bbox_diag``, preventing over-smoothing and silhouette shrinkage.
    """
    out = o3d.geometry.TriangleMesh(mesh)
    verts_ref = np.asarray(out.vertices, dtype=np.float64).copy()
    if len(verts_ref) == 0 or len(out.triangles) == 0 or mode == "none":
        return out, {"mode": mode, "batches_run": 0, "stopped_reason": "noop", "p95_drift": 0.0, "p95_drift_frac": 0.0}

    diag = max(float(np.linalg.norm(verts_ref.max(axis=0) - verts_ref.min(axis=0))), 1e-6)
    max_drift = float(max_drift_frac) * diag
    tree_ref = cKDTree(verts_ref)

    has_colors = bool(out.has_vertex_colors() and len(out.vertex_colors) == len(verts_ref))
    v_colors = np.asarray(out.vertex_colors, dtype=np.float64).copy() if has_colors else None
    has_uvs = bool(out.has_triangle_uvs() and len(out.triangle_uvs) == 3 * len(out.triangles))
    tri_uvs = np.asarray(out.triangle_uvs, dtype=np.float64).copy() if has_uvs else None

    history: list[dict[str, float]] = []
    batches_run = 0
    stopped_reason = "max_batches_reached"
    p95_drift = 0.0

    for b_idx in range(1, max(1, int(max_batches)) + 1):
        prev_verts = np.asarray(out.vertices, dtype=np.float64).copy()
        if mode == "bilateral_normal":
            cand = bilateral_normal_smooth(
                out,
                normal_iter=max(1, int(batch_iters)),
                vertex_iter=max(2, int(batch_iters) * 3),
                sigma_s_scale=bilateral_sigma_s_scale,
                sigma_r=bilateral_sigma_r,
                vertex_confidence=vertex_confidence,
                anchor_weight=bilateral_anchor_weight,
            )
        elif mode == "hybrid":
            # Cong thuc tong quat cho canh hon hop (vat cong + vat canh sac):
            # 1) Taubin nhe khu gon song tren be mat cong (khong teo the tich, an toan voi dinh co lap)
            # 2) Bilateral normal lam min tiep va giu sac canh goc >= 32 do (neo chat vung w=1.0)
            cand = taubin_volume_preserving_smooth(
                out,
                iterations=max(1, int(batch_iters) // 2),
                lambda_pos=float(taubin_lambda),
                mu_neg=float(taubin_mu),
            )
            cand = bilateral_normal_smooth(
                cand,
                normal_iter=max(1, int(batch_iters)),
                vertex_iter=max(2, int(batch_iters) * 2),
                sigma_s_scale=bilateral_sigma_s_scale,
                sigma_r=bilateral_sigma_r,
                vertex_confidence=vertex_confidence,
                anchor_weight=bilateral_anchor_weight,
            )
        else:
            cand = taubin_volume_preserving_smooth(
                out,
                iterations=max(1, int(batch_iters)),
                lambda_pos=float(taubin_lambda),
                mu_neg=float(taubin_mu),
            )

        cand_verts = np.asarray(cand.vertices, dtype=np.float64)
        dists, _ = tree_ref.query(cand_verts, k=1)
        cand_p95 = float(np.percentile(dists, 95))
        cand_mean = float(np.mean(dists))

        if cand_p95 > max_drift and batches_run > 0:
            # Revert to previous safe batch so drift stays strictly within budget
            out.vertices = o3d.utility.Vector3dVector(prev_verts)
            stopped_reason = f"drift_budget_exceeded_at_batch_{b_idx}"
            break

        out = cand
        batches_run = b_idx
        p95_drift = cand_p95
        history.append({
            "batch": b_idx,
            "mean_drift": round(cand_mean, 6),
            "p95_drift": round(cand_p95, 6),
            "p95_drift_frac": round(cand_p95 / diag, 6),
        })

        if cand_p95 > max_drift:
            stopped_reason = f"drift_budget_reached_at_batch_{b_idx}"
            break

    if v_colors is not None and len(v_colors) == len(out.vertices):
        out.vertex_colors = o3d.utility.Vector3dVector(v_colors)
    if tri_uvs is not None and len(tri_uvs) == 3 * len(out.triangles):
        out.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)
    out.compute_triangle_normals()
    out.compute_vertex_normals()

    return out, {
        "mode": mode,
        "batches_run": batches_run,
        "total_iters": batches_run * int(batch_iters),
        "stopped_reason": stopped_reason,
        "p95_drift": round(p95_drift, 6),
        "p95_drift_frac": round(p95_drift / diag, 6),
        "history": history,
    }
