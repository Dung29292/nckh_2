"""Unified Region-Wise Surface Model Selection (BIC/GRIC) & Data-Driven Void Completion.

Implements Requirement A (one mathematical formulation for all object shapes):
  - Normal-continuity region growing + co-planar fragment unification
  - Nested model hierarchy per region: Plane (k=3) < Sphere (k=4) < Cylinder (k=5) < Freeform MLS
  - Huber-BIC scoring with optional soft Bayesian `shape_hint` prior (data always wins)
  - Universal Visual-Hull-gated void completion inside validated analytic regions
  - Data-derived bilateral normal bandwidth sigma_r from planar normal dispersion
"""
from __future__ import annotations

from collections import deque
import dataclasses
import math
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from nckh_2.backends.adaptive import AdaptivePointStats
from nckh_2.backends.surfaces import CircularCylinder, HessianPlane


def huber_loss(u: np.ndarray, k: float = 1.345) -> np.ndarray:
    """Evaluate robust Huber M-estimator loss rho_H(u)."""
    abs_u = np.abs(u)
    quad = 0.5 * abs_u * abs_u
    lin = k * (abs_u - 0.5 * k)
    return np.where(abs_u <= k, quad, lin)


@dataclasses.dataclass
class RegionFit:
    """BIC-selected geometric model for a single surface region."""

    region_id: int
    indices: np.ndarray
    model_type: str  # "plane" | "cylinder" | "sphere" | "freeform"
    bic_score: float
    all_scores: dict[str, float]
    rms_residual: float
    plane: HessianPlane | None = None
    cylinder: CircularCylinder | None = None
    sphere_center: np.ndarray | None = None
    sphere_radius: float = 0.0
    normal_dispersion: float = 0.0
    prior_rejected: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "region_id": int(self.region_id),
            "num_points": int(len(self.indices)),
            "model_type": str(self.model_type),
            "bic_score": round(float(self.bic_score), 3),
            "rms_residual": round(float(self.rms_residual), 6),
            "normal_dispersion": round(float(self.normal_dispersion), 4),
            "prior_rejected": bool(self.prior_rejected),
        }
        if self.plane is not None:
            out["plane"] = self.plane.to_dict()
        if self.cylinder is not None:
            out["cylinder"] = {
                "axis": [round(float(v), 6) for v in self.cylinder.axis],
                "anchor": [round(float(v), 6) for v in self.cylinder.anchor],
                "radius": round(float(self.cylinder.radius), 6),
            }
        return out


def _fit_sphere_algebraic(pts: np.ndarray, norm_eps: float = 1.0e-12) -> tuple[np.ndarray, float] | None:
    """Linear least-squares sphere fit: x^2+y^2+z^2 = 2*cx*x + 2*cy*y + 2*cz*z + (R^2 - ||c||^2)."""
    if len(pts) < 6:
        return None
    A = np.column_stack([2.0 * pts, np.ones(len(pts), dtype=np.float64)])
    b = np.sum(pts * pts, axis=1)
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center = sol[:3]
    r2 = float(sol[3] + float(np.dot(center, center)))
    if not np.isfinite(r2) or r2 <= norm_eps:
        return None
    return center, math.sqrt(r2)


def _local_freeform_residuals(pts: np.ndarray, k: int, norm_eps: float = 1.0e-12) -> np.ndarray:
    """Estimate local quadratic-surface residual per point for the freeform BIC baseline."""
    n = len(pts)
    if n < 6:
        return np.zeros(n, dtype=np.float64)
    k_eff = min(max(int(k), 6), n - 1)
    tree = cKDTree(pts)
    sample_n = min(n, 1500)
    if sample_n < n:
        idx = np.linspace(0, n - 1, sample_n, dtype=int)
        q_pts = pts[idx]
    else:
        q_pts = pts
    _, nbrs = tree.query(q_pts, k=k_eff + 1, workers=-1)
    patches = pts[nbrs]
    centered = patches - np.mean(patches, axis=1, keepdims=True)
    covs = np.einsum("mki,mkj->mij", centered, centered) / float(max(k_eff, 1))
    eigvals = np.linalg.eigvalsh(covs)
    res_sub = np.sqrt(np.maximum(eigvals[:, 0], 0.0))
    if sample_n < n:
        # Scale up to full region count deterministically
        return np.repeat(res_sub, int(math.ceil(n / float(sample_n))))[:n]
    return res_sub


def _fit_region_bic(
    region_id: int,
    indices: np.ndarray,
    all_pts: np.ndarray,
    all_nrms: np.ndarray,
    stats: AdaptivePointStats,
    cfg_adapt: dict[str, Any],
    num_cfg: dict[str, Any],
    seed: int = 42,
) -> RegionFit:
    """Evaluate Plane, Sphere, Cylinder, and Freeform models on one region and select by BIC."""
    pts = all_pts[indices]
    nrms = all_nrms[indices]
    n = len(pts)
    norm_eps = float(num_cfg["norm_eps"])
    sigma = max(stats.noise_sigma, norm_eps)
    huber_k = float(cfg_adapt["huber_k"])
    bic_lam = float(cfg_adapt["bic_lambda"])
    shape_hint = str(cfg_adapt.get("shape_hint", "auto")).strip().lower()
    prior_bonus = float(cfg_adapt.get("shape_hint_prior_bonus", 2.0))
    ln_n = math.log(max(float(n), 2.0))

    def _bic(residuals: np.ndarray, k_params: float, family: str) -> float:
        data_term = float(np.sum(huber_loss(residuals / sigma, k=huber_k)))
        penalty = bic_lam * float(k_params) * ln_n
        bonus = 0.0
        if (shape_hint == "box" and family == "plane") or (
            shape_hint == "cylinder" and family == "cylinder"
        ) or (shape_hint == "freeform" and family == "freeform"):
            bonus = prior_bonus * ln_n
        return data_term + penalty - bonus

    scores: dict[str, float] = {}

    # 1. Plane (k = 3)
    plane_obj: HessianPlane | None = None
    plane_rms = float("inf")
    if n >= 6:
        plane_obj = HessianPlane.fit_svd(pts, norm_eps=norm_eps, canonical_sign=False)
        # Orient plane normal to match mean region normal
        if float(np.dot(plane_obj.normal, np.mean(nrms, axis=0))) < 0.0:
            plane_obj = HessianPlane(normal=-plane_obj.normal, d=-plane_obj.d, norm_eps=norm_eps)
        r_plane = plane_obj.distance(pts)
        plane_rms = float(np.sqrt(np.mean(r_plane * r_plane)))
        scores["plane"] = _bic(r_plane, 3.0, "plane")

    # 2. Sphere (k = 4)
    sphere_fit = _fit_sphere_algebraic(pts, norm_eps=norm_eps)
    sphere_center: np.ndarray | None = None
    sphere_radius = 0.0
    sphere_rms = float("inf")
    if sphere_fit is not None:
        sc, sr = sphere_fit
        if 0.5 * stats.point_spacing < sr < 5.0 * stats.diag:
            r_sph = np.abs(np.linalg.norm(pts - sc[None, :], axis=1) - sr)
            sphere_rms = float(np.sqrt(np.mean(r_sph * r_sph)))
            sphere_center, sphere_radius = sc, sr
            scores["sphere"] = _bic(r_sph, 4.0, "sphere")

    # 3. Circular Cylinder (k = 5)
    cyl_obj: CircularCylinder | None = None
    cyl_rms = float("inf")
    if n >= 20:
        sub_n = min(n, 2000)
        if sub_n < n:
            s_idx = np.linspace(0, n - 1, sub_n, dtype=int)
            c_pts, c_nrms = pts[s_idx], nrms[s_idx]
        else:
            c_pts, c_nrms = pts, nrms
        quad_cfg = {
            "min_inliers": max(12, int(0.35 * len(c_pts))),
            "eps_cylinder_m": float(stats.inlier_tol),
            "cylinder_radius_range_m": [float(max(2.0 * stats.point_spacing, norm_eps)), float(0.75 * stats.diag)],
            "ransac_iterations": 250,
            "normal_axis_ortho_max_cos": 0.35,
            "vertical_column_min_abs_az": 0.0,
            "gn_max_iterations": 20,
            "gn_tol": 1.0e-5,
        }
        try:
            cyl_cand, cyl_inl = CircularCylinder.fit_ransac_with_normals(
                c_pts, c_nrms, quad_cfg=quad_cfg, num_cfg=num_cfg, constrain_vertical=False, seed=seed
            )
            if cyl_cand is not None and len(cyl_inl) >= max(12, int(0.45 * len(c_pts))):
                # Geometric validity: a true cylinder must subtend a real angular arc (>= 60 deg) around its axis
                ax = cyl_cand.axis
                ref_c = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                if abs(float(np.dot(ref_c, ax))) > 0.9:
                    ref_c = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                e1 = np.cross(ax, ref_c)
                e1 /= max(float(np.linalg.norm(e1)), norm_eps)
                e2 = np.cross(ax, e1)
                rel_c = pts - cyl_cand.anchor[None, :]
                ang_c = np.sort(np.mod(np.degrees(np.arctan2(rel_c @ e2, rel_c @ e1)), 360.0))
                gaps_c = np.empty(len(ang_c), dtype=np.float64)
                gaps_c[:-1] = np.diff(ang_c)
                gaps_c[-1] = (ang_c[0] + 360.0) - ang_c[-1]
                arc_span_deg = 360.0 - float(np.max(gaps_c))
                if arc_span_deg >= 60.0:
                    r_cyl = cyl_cand.residual(pts)
                    cyl_rms = float(np.sqrt(np.mean(r_cyl * r_cyl)))
                    cyl_obj = cyl_cand
                    scores["cylinder"] = _bic(r_cyl, 5.0, "cylinder")
        except Exception:
            pass

    # 4. Freeform local Monge MLS baseline (effective DoF scales with number of independent local patches)
    r_free = _local_freeform_residuals(pts, k=stats.k_neighbors, norm_eps=norm_eps)
    free_rms = float(np.sqrt(np.mean(r_free * r_free)))
    k_free = max(9.0, 6.0 * (float(n) / float(max(stats.k_neighbors, 8))))
    scores["freeform"] = _bic(r_free, k_free, "freeform")

    best_model = min(scores, key=lambda m: scores[m])
    rms_map = {"plane": plane_rms, "sphere": sphere_rms, "cylinder": cyl_rms, "freeform": free_rms}

    # Compute normal dispersion (1 - ||mean_n||) for bilateral sigma_r estimation
    mean_n = np.mean(nrms, axis=0)
    n_disp = max(0.0, 1.0 - float(np.linalg.norm(mean_n)))

    prior_rejected = False
    if shape_hint == "box" and best_model != "plane":
        prior_rejected = True
    elif shape_hint == "cylinder" and best_model != "cylinder":
        prior_rejected = True
    elif shape_hint == "freeform" and best_model != "freeform":
        prior_rejected = True

    return RegionFit(
        region_id=region_id,
        indices=indices,
        model_type=best_model,
        bic_score=float(scores[best_model]),
        all_scores={k: round(float(v), 3) for k, v in scores.items()},
        rms_residual=float(rms_map[best_model]),
        plane=plane_obj if best_model == "plane" else None,
        cylinder=cyl_obj if best_model == "cylinder" else None,
        sphere_center=sphere_center if best_model == "sphere" else None,
        sphere_radius=float(sphere_radius) if best_model == "sphere" else 0.0,
        normal_dispersion=n_disp,
        prior_rejected=prior_rejected,
    )


def segment_and_select_regions(
    points: np.ndarray,
    normals: np.ndarray,
    stats: AdaptivePointStats,
    cfg_adapt: dict[str, Any],
    num_cfg: dict[str, Any],
    seed: int = 42,
) -> list[RegionFit]:
    """Segment point cloud via normal-continuity region growing + co-planar unification, then score by BIC."""
    pts = np.asarray(points, dtype=np.float64)
    nrms = np.asarray(normals, dtype=np.float64)
    n = len(pts)
    if n < 30 or len(nrms) != n:
        return []

    norm_eps = float(num_cfg["norm_eps"])
    n_len = np.maximum(np.linalg.norm(nrms, axis=1, keepdims=True), norm_eps)
    nrms = nrms / n_len

    # Subsample for fast region growing if cloud is very dense, then assign all points to nearest region
    max_grow_pts = 25000
    if n > max_grow_pts:
        rng = np.random.default_rng(seed)
        sub_idx = np.sort(rng.choice(n, size=max_grow_pts, replace=False))
        g_pts = pts[sub_idx]
        g_nrms = nrms[sub_idx]
    else:
        sub_idx = np.arange(n, dtype=np.int64)
        g_pts = pts
        g_nrms = nrms

    m = len(g_pts)
    k_grow = min(max(stats.k_neighbors, 12), m - 1)
    tree = cKDTree(g_pts)
    dists, nbrs = tree.query(g_pts, k=k_grow + 1, workers=-1)
    nbr_idx = nbrs[:, 1:]
    nbr_dists = dists[:, 1:]
    max_step = max(3.5 * stats.point_spacing * (math.sqrt(float(n) / float(m))), stats.gap_tol)

    seed_angle_deg = float(cfg_adapt.get("region_normal_seed_angle_deg", 22.0))
    cos_thr = math.cos(math.radians(seed_angle_deg))
    # Data-adaptive smoothness-constrained crease barrier (Rabbani et al., 2006):
    # uses the lower of cos(0.55 * seed_angle) and the cloud's own 20th-percentile smoothness
    # so sharp 90-deg creases block BFS while noisy real MVS flat walls still grow cleanly.
    nbr_dots = np.abs(np.einsum("mi,mki->mk", g_nrms, g_nrms[nbr_idx]))
    smoothness = np.mean(nbr_dots, axis=1)
    q35_smooth = float(np.quantile(smoothness, 0.35))
    crease_cos = min(math.cos(math.radians(0.55 * seed_angle_deg)), max(q35_smooth, 0.90))
    seed_order = np.argsort(-smoothness)

    visited = np.zeros(m, dtype=bool)
    min_reg_g = max(20, int(round(float(cfg_adapt.get("region_min_points_frac", 0.015)) * m)))
    max_regions = int(cfg_adapt.get("region_max_regions", 24))
    raw_regions: list[np.ndarray] = []

    for s in seed_order:
        if visited[s] or float(smoothness[s]) < crease_cos:
            continue
        q: deque[int] = deque([int(s)])
        visited[s] = True
        comp: list[int] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            cn = g_nrms[cur]
            for nb, nd in zip(nbr_idx[cur], nbr_dists[cur]):
                nb_i = int(nb)
                if visited[nb_i] or nd > max_step:
                    continue
                if float(np.dot(cn, g_nrms[nb_i])) >= cos_thr:
                    visited[nb_i] = True
                    if float(smoothness[nb_i]) >= crease_cos:
                        q.append(nb_i)
                    else:
                        comp.append(nb_i)
        if len(comp) >= min_reg_g:
            raw_regions.append(np.asarray(comp, dtype=np.int64))
            if len(raw_regions) >= max_regions * 2:
                break

    if not raw_regions:
        return []

    # Co-planar fragment unification: merges disconnected borders of a hollow flat wall (e.g. box_01)
    merged_groups: list[list[int]] = []
    used = [False] * len(raw_regions)
    planes_cand: list[HessianPlane | None] = []
    for reg in raw_regions:
        pl = HessianPlane.fit_svd(g_pts[reg], norm_eps=norm_eps)
        if float(np.dot(pl.normal, np.mean(g_nrms[reg], axis=0))) < 0.0:
            pl = HessianPlane(normal=-pl.normal, d=-pl.d, norm_eps=norm_eps)
        rms = float(np.sqrt(np.mean(pl.distance(g_pts[reg]) ** 2)))
        planes_cand.append(pl if rms <= stats.inlier_tol else None)

    for i in range(len(raw_regions)):
        if used[i]:
            continue
        used[i] = True
        grp = [i]
        pi = planes_cand[i]
        if pi is not None:
            for j in range(i + 1, len(raw_regions)):
                if used[j] or planes_cand[j] is None:
                    continue
                pj = planes_cand[j]
                if float(np.dot(pi.normal, pj.normal)) >= cos_thr:
                    comb_idx = np.concatenate([raw_regions[k] for k in grp] + [raw_regions[j]])
                    p_joint = HessianPlane.fit_svd(g_pts[comb_idx], norm_eps=norm_eps)
                    rms_j = float(np.sqrt(np.mean(p_joint.distance(g_pts[comb_idx]) ** 2)))
                    if rms_j <= stats.inlier_tol:
                        used[j] = True
                        grp.append(j)
                        pi = p_joint
        merged_groups.append(grp)

    # Map subsampled regions back to full point cloud indices
    if n > max_grow_pts:
        g_labels = np.full(m, -1, dtype=np.int32)
        for gid, grp in enumerate(merged_groups[:max_regions]):
            for r_i in grp:
                g_labels[raw_regions[r_i]] = gid
        _, nn_g = tree.query(pts, k=1, workers=-1)
        full_labels = g_labels[nn_g]
        final_index_lists = [
            np.nonzero(full_labels == gid)[0]
            for gid in range(min(len(merged_groups), max_regions))
        ]
    else:
        final_index_lists = [
            np.concatenate([raw_regions[r_i] for r_i in grp])
            for grp in merged_groups[:max_regions]
        ]

    min_reg_full = max(20, int(round(float(cfg_adapt.get("region_min_points_frac", 0.015)) * n)))
    results: list[RegionFit] = []
    for rid, idxs in enumerate(final_index_lists):
        if len(idxs) < min_reg_full:
            continue
        fit = _fit_region_bic(
            region_id=rid,
            indices=idxs,
            all_pts=pts,
            all_nrms=nrms,
            stats=stats,
            cfg_adapt=cfg_adapt,
            num_cfg=num_cfg,
            seed=seed + rid,
        )
        results.append(fit)

    results.sort(key=lambda r: len(r.indices), reverse=True)
    return results


def fill_analytic_region_voids(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray | None,
    regions: list[RegionFit],
    stats: AdaptivePointStats,
    visual_hull: Any | None,
    cfg_adapt: dict[str, Any],
    norm_eps: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, dict[str, Any]]:
    """Fill genuine MVS voids inside BIC-validated planar or cylindrical regions (Visual Hull gated).

    Never overwrites observed MVS points (confidence = 1.0). Added points carry confidence = 0.70.
    """
    pts = np.asarray(points, dtype=np.float64)
    nrms = np.asarray(normals, dtype=np.float64)
    n_obs = len(pts)
    conf_obs = np.ones(n_obs, dtype=np.float64)

    if not bool(cfg_adapt.get("region_void_fill_enable", True)) or not regions or n_obs < 30:
        return pts, nrms, colors, conf_obs, {"added_points": 0, "filled_regions": 0}

    obs_tree = cKDTree(pts)
    step = max(1.25 * stats.point_spacing, stats.diag / 250.0, norm_eps)
    gap_thr = stats.gap_tol

    add_pts_list: list[np.ndarray] = []
    add_nrms_list: list[np.ndarray] = []
    add_cols_list: list[np.ndarray] = []
    filled_regions = 0

    for reg in regions:
        r_pts = pts[reg.indices]
        if reg.model_type == "plane" and reg.plane is not None and len(r_pts) >= 20:
            n_pl = reg.plane.normal
            # Orthonormal 2D basis (u, v) on plane
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(ref, n_pl))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            u_ax = np.cross(n_pl, ref)
            u_ax /= max(float(np.linalg.norm(u_ax)), norm_eps)
            v_ax = np.cross(n_pl, u_ax)

            proj_pts = reg.plane.project_points(r_pts)
            origin = np.mean(proj_pts, axis=0)
            rel = proj_pts - origin[None, :]
            uv = np.column_stack([rel @ u_ax, rel @ v_ax])

            u_min, v_min = float(uv[:, 0].min()), float(uv[:, 1].min())
            u_max, v_max = float(uv[:, 0].max()), float(uv[:, 1].max())
            nu = int(np.clip(math.ceil((u_max - u_min) / step), 4, 160))
            nv = int(np.clip(math.ceil((v_max - v_min) / step), 4, 160))
            if nu * nv < 16:
                continue

            # Rasterize 2D convex hull of region inliers onto the (nu, nv) grid
            hull_2d = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
            if len(hull_2d) < 3:
                continue
            gu = np.linspace(u_min, u_max, nu)
            gv = np.linspace(v_min, v_max, nv)
            uu, vv = np.meshgrid(gu, gv, indexing="ij")
            cand_uv = np.column_stack([uu.ravel(), vv.ravel()])

            # Fast 2D point-in-convex-polygon test via OpenCV raster mask
            poly_px = np.column_stack([
                np.round((hull_2d[:, 0] - u_min) / max(u_max - u_min, norm_eps) * (nu - 1)),
                np.round((hull_2d[:, 1] - v_min) / max(v_max - v_min, norm_eps) * (nv - 1)),
            ]).astype(np.int32)
            mask_2d = np.zeros((nv, nu), dtype=np.uint8)
            cv2.fillConvexPoly(mask_2d, poly_px, 1)
            inside_hull = mask_2d.T.ravel() > 0
            if not np.any(inside_hull):
                continue

            sel_uv = cand_uv[inside_hull]
            cand_3d = origin[None, :] + sel_uv[:, 0:1] * u_ax[None, :] + sel_uv[:, 1:2] * v_ax[None, :]

            # Dual gate: genuine MVS void AND inside Visual Hull
            d_obs, nn_idx = obs_tree.query(cand_3d, k=1, workers=-1)
            keep = d_obs > gap_thr
            if visual_hull is not None and np.any(keep):
                keep[keep] &= visual_hull.contains_points(cand_3d[keep], dilate_voxels=1)

            n_keep = int(np.sum(keep))
            if n_keep > 0:
                add_pts_list.append(cand_3d[keep])
                add_nrms_list.append(np.repeat(n_pl[None, :], n_keep, axis=0))
                if colors is not None:
                    add_cols_list.append(colors[nn_idx[keep]])
                filled_regions += 1

        elif reg.model_type == "cylinder" and reg.cylinder is not None and len(r_pts) >= 30:
            ax = reg.cylinder.axis
            c0 = reg.cylinder.anchor
            rad = reg.cylinder.radius
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(ref, ax))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            e1 = np.cross(ax, ref)
            e1 /= max(float(np.linalg.norm(e1)), norm_eps)
            e2 = np.cross(ax, e1)

            rel = r_pts - c0[None, :]
            z_vals = rel @ ax
            z_lo, z_hi = float(np.percentile(z_vals, 2.0)), float(np.percentile(z_vals, 98.0))
            if z_hi - z_lo <= step:
                continue
            nz = int(np.clip(math.ceil((z_hi - z_lo) / step), 6, 140))
            n_phi = int(np.clip(math.ceil(2.0 * math.pi * rad / step), 24, 180))
            zs = np.linspace(z_lo, z_hi, nz)
            phis = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
            zz, pp = np.meshgrid(zs, phis, indexing="ij")
            rad_dir = np.cos(pp.ravel())[:, None] * e1[None, :] + np.sin(pp.ravel())[:, None] * e2[None, :]
            cand_3d = c0[None, :] + zz.ravel()[:, None] * ax[None, :] + rad * rad_dir

            d_obs, nn_idx = obs_tree.query(cand_3d, k=1, workers=-1)
            keep = d_obs > gap_thr
            if visual_hull is not None and np.any(keep):
                keep[keep] &= visual_hull.contains_points(cand_3d[keep], dilate_voxels=1)
            elif visual_hull is None:
                # Without Visual Hull, only fill small local voids inside observed angular span
                keep &= d_obs < (4.0 * gap_thr)

            n_keep = int(np.sum(keep))
            if n_keep > 0:
                add_pts_list.append(cand_3d[keep])
                add_nrms_list.append(rad_dir[keep])
                if colors is not None:
                    add_cols_list.append(colors[nn_idx[keep]])
                filled_regions += 1

    if not add_pts_list:
        return pts, nrms, colors, conf_obs, {"added_points": 0, "filled_regions": 0}

    add_pts = np.vstack(add_pts_list)
    add_nrms = np.vstack(add_nrms_list)
    out_pts = np.vstack([pts, add_pts])
    out_nrms = np.vstack([nrms, add_nrms])
    out_cols = np.vstack([colors, np.vstack(add_cols_list)]) if (colors is not None and add_cols_list) else colors
    out_conf = np.concatenate([conf_obs, np.full(len(add_pts), 0.70, dtype=np.float64)])

    return out_pts, out_nrms, out_cols, out_conf, {
        "added_points": int(len(add_pts)),
        "filled_regions": int(filled_regions),
    }


def estimate_bilateral_sigma_r(regions: list[RegionFit], fallback: float = 0.25) -> float:
    """Derive bilateral normal range bandwidth sigma_r from normal dispersion inside planar regions."""
    planar_disps = [r.normal_dispersion for r in regions if r.model_type == "plane"]
    if not planar_disps:
        return float(fallback)
    med_disp = float(np.median(planar_disps))
    # Map dispersion (1 - cos theta) to chord-length normal bandwidth ~ 2 * sqrt(2 * disp)
    est = 2.0 * math.sqrt(max(2.0 * med_disp, 1.0e-4))
    return float(np.clip(est, 0.14, 0.40))
