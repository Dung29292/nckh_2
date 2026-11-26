"""Adaptive Geometric Estimation, Clutter Reclamation & Manhattan Regularization (Section 3 of nckh.pdf).

Strictly implements:
  - Section 3.1 & Eq. (10): Iterative 3-point local RANSAC with Dual Validation
    (Orthogonal distance < eps_plane AND Normal compatibility cos(theta) = |n_plane . n_Q|, theta < 15 deg)
    followed by seed-discarded Global SVD Refit.
  - Section 3.2: RandLA-Net Clutter Reclamation + 2D Wall Occupancy Morphological Closing.
  - Section 3.3 & Eq. (11): Manhattan World dominant wall orientation theta_0 = atan2(ny, nx) mod (pi/2),
    scene rotation R(-theta_0), rectilinear regularization (|dx| >= |dy| horizontal else vertical),
    and exact analytical 90-deg line-line corner intersections.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_closing
from scipy.spatial import cKDTree

from nckh_2.backends.surfaces import HessianPlane


@dataclasses.dataclass
class DetectedPlane:
    """Validated architectural plane instance extracted by Dual-Validated Iterative RANSAC."""

    plane: HessianPlane
    inlier_indices: np.ndarray
    seed_indices: tuple[int, int, int]
    orientation_class: str
    reclaimed_clutter_indices: np.ndarray = dataclasses.field(default_factory=lambda: np.empty(0, dtype=np.int64))
    gap_closed_cells: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plane": self.plane.to_dict(),
            "orientation_class": self.orientation_class,
            "num_inliers": int(len(self.inlier_indices)),
            "num_reclaimed_clutter": int(len(self.reclaimed_clutter_indices)),
            "gap_closed_cells": int(self.gap_closed_cells),
        }


@dataclasses.dataclass
class WallSegment2D:
    """2D floorplan wall segment before and after Manhattan rectilinear regularization."""

    p_start: np.ndarray          # shape (2,)
    p_end: np.ndarray            # shape (2,)
    normal_2d: np.ndarray        # unit 2D normal (nx, ny)
    offset_c: float              # 2D Hessian line nx * x + ny * y + c = 0
    axis_lock: str               # "horizontal" | "vertical" | "diagonal"
    source_plane_idx: int

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p_end - self.p_start))


def _ensure_unit_normals(
    points: np.ndarray,
    normals: np.ndarray | None,
    ransac_cfg: dict[str, Any],
    num_cfg: dict[str, Any],
) -> np.ndarray:
    norm_eps = float(num_cfg["norm_eps"])
    if normals is not None and len(normals) == len(points):
        nrm = np.asarray(normals, dtype=np.float64).copy()
        lens = np.linalg.norm(nrm, axis=1, keepdims=True)
        valid = lens[:, 0] > norm_eps
        if np.all(valid):
            return nrm / lens

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(ransac_cfg["normal_estimation_radius_m"]),
            max_nn=int(ransac_cfg["normal_estimation_max_nn"]),
        )
    )
    nrm = np.asarray(pcd.normals, dtype=np.float64)
    lens = np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), norm_eps)
    return nrm / lens


# =============================================================================
# SECTION 3.1: ITERATIVE RANSAC WITH DUAL VALIDATION & GLOBAL SVD REFIT (Eq. 10)
# =============================================================================

def iterative_ransac_planes(
    points: np.ndarray,
    normals: np.ndarray | None,
    floorplan_cfg: dict[str, Any],
) -> tuple[list[DetectedPlane], np.ndarray]:
    """Extract architectural planes sequentially via Dual-Validated RANSAC + Global SVD Refit.

    Implements Section 3.1 step-by-step:
      1. Minimal 3-Point Local Sample Set: n = (P2 - P1) x (P3 - P1).
      2. Validation 1 (Orthogonal Distance): |a*xj + b*yj + c*zj + d| < eps_plane.
      3. Validation 2 (Normal Compatibility, Eq. 10): cos(theta) = |n_plane . n_Q| > cos(theta_max).
      4. Global Refit: Discard the 3 seed points and solve exact SVD Total Least Squares over all inliers.
    """
    pts = np.asarray(points, dtype=np.float64)
    r_cfg = floorplan_cfg["adaptive_ransac"]
    p_cfg = floorplan_cfg["hessian_plane"]
    n_cfg = floorplan_cfg["numerical"]

    nrms = _ensure_unit_normals(pts, normals, r_cfg, n_cfg)
    eps_plane = float(r_cfg["eps_plane_m"])
    cos_thr = float(np.cos(np.radians(float(r_cfg["normal_max_angle_deg"]))))
    max_planes = int(r_cfg["max_planes"])
    max_iters = int(r_cfg["ransac_iterations"])
    conf_p = float(r_cfg["confidence"])
    min_inliers = int(r_cfg["min_inliers"])
    min_rem = int(r_cfg["min_remaining_points"])
    seed_radius = float(r_cfg["local_seed_radius_m"])
    discard_seeds = bool(r_cfg["discard_seed_on_refit"])
    collinear_eps = float(n_cfg["collinear_cross_eps"])
    norm_eps = float(n_cfg["norm_eps"])
    canon_sign = bool(p_cfg["canonical_sign"])

    rng = np.random.default_rng(int(r_cfg["deterministic_seed"]))
    rem_global_idx = np.arange(len(pts), dtype=np.int64)
    scene_z_split = float(np.quantile(pts[:, 2], float(p_cfg["floor_ceiling_split_quantile"]))) if len(pts) else 0.0
    scene_z_bounds = (float(np.quantile(pts[:, 2], 0.01)), float(np.quantile(pts[:, 2], 0.99))) if len(pts) else (0.0, 1.0)

    detected: list[DetectedPlane] = []

    for _ in range(max_planes):
        n_rem = len(rem_global_idx)
        if n_rem < max(min_inliers, min_rem, 4):
            break

        rem_pts = pts[rem_global_idx]
        rem_nrms = nrms[rem_global_idx]
        tree = cKDTree(rem_pts)

        best_plane: HessianPlane | None = None
        best_local_inliers = np.empty(0, dtype=np.int64)
        best_local_seeds = (0, 1, 2)
        adaptive_limit = max_iters
        trial = 0

        while trial < adaptive_limit:
            trial += 1
            # Sample anchor P1 and draw P2, P3 from local neighborhood to prevent cross-room outlier bridges
            s1 = int(rng.integers(0, n_rem))
            local_nbrs = tree.query_ball_point(rem_pts[s1], r=seed_radius) if seed_radius > 0.0 else []
            if len(local_nbrs) >= 3:
                chosen = rng.choice(local_nbrs, size=3, replace=False)
                s1, s2, s3 = int(chosen[0]), int(chosen[1]), int(chosen[2])
            else:
                chosen = rng.choice(n_rem, size=3, replace=False)
                s1, s2, s3 = int(chosen[0]), int(chosen[1]), int(chosen[2])

            cand = HessianPlane.from_three_points(
                rem_pts[s1],
                rem_pts[s2],
                rem_pts[s3],
                collinear_eps=collinear_eps,
                norm_eps=norm_eps,
                canonical_sign=canon_sign,
            )
            if cand is None:
                continue

            # Filter 1: Orthogonal distance |a*xj + b*yj + c*zj + d| < eps_plane (Eq. 2)
            dist_ok = cand.distance(rem_pts) < eps_plane
            # Filter 2: Normal compatibility cos(theta) = |n_plane . n_Q| > cos(theta_max) (Eq. 10)
            cos_theta = np.abs(rem_nrms @ cand.normal)
            norm_ok = cos_theta > cos_thr

            inliers_local = np.nonzero(dist_ok & norm_ok)[0]
            if len(inliers_local) > len(best_local_inliers):
                best_local_inliers = inliers_local
                best_plane = cand
                best_local_seeds = (s1, s2, s3)
                # Standard adaptive RANSAC stopping bound: K = log(1 - p) / log(1 - w^3)
                w_ratio = len(inliers_local) / float(n_rem)
                if 0.0 < w_ratio < 1.0:
                    denom = np.log(max(1.0 - w_ratio ** 3, 1.0e-12))
                    if denom < 0.0:
                        k_est = int(np.ceil(np.log(max(1.0 - conf_p, 1.0e-12)) / denom))
                        adaptive_limit = min(max_iters, max(k_est, 25))

        if best_plane is None or len(best_local_inliers) < min_inliers:
            break

        # Global Refit (Section 3.1): Discard the 3 initial seed points and refit via SVD on all remaining inliers
        if discard_seeds:
            refit_local = np.setdiff1d(best_local_inliers, np.asarray(best_local_seeds, dtype=np.int64), assume_unique=True)
            if len(refit_local) < 3:
                refit_local = best_local_inliers
        else:
            refit_local = best_local_inliers

        refined_plane = HessianPlane.fit_svd(
            rem_pts[refit_local],
            norm_eps=norm_eps,
            canonical_sign=canon_sign,
        )

        # Re-evaluate final dual-validated inlier consensus on the optimal SVD plane
        final_dist_ok = refined_plane.distance(rem_pts) < eps_plane
        final_norm_ok = np.abs(rem_nrms @ refined_plane.normal) > cos_thr
        final_local = np.nonzero(final_dist_ok & final_norm_ok)[0]
        if len(final_local) < min_inliers:
            final_local = best_local_inliers

        global_inliers = rem_global_idx[final_local]
        global_seeds = (
            int(rem_global_idx[best_local_seeds[0]]),
            int(rem_global_idx[best_local_seeds[1]]),
            int(rem_global_idx[best_local_seeds[2]]),
        )
        z_med = float(np.median(pts[global_inliers, 2]))
        orient_cls = refined_plane.classify_orientation(
            p_cfg,
            inlier_z_median=z_med,
            scene_z_split=scene_z_split,
            scene_z_bounds=scene_z_bounds,
        )

        detected.append(
            DetectedPlane(
                plane=refined_plane,
                inlier_indices=global_inliers,
                seed_indices=global_seeds,
                orientation_class=orient_cls,
            )
        )

        keep_rem_mask = np.ones(n_rem, dtype=bool)
        keep_rem_mask[final_local] = False
        rem_global_idx = rem_global_idx[keep_rem_mask]

    return detected, rem_global_idx


# =============================================================================
# SECTION 3.2: CLUTTER RECLAMATION & MORPHOLOGICAL STRUCTURAL GAP CLOSING
# =============================================================================

def reclaim_clutter_and_close_gaps(
    all_points: np.ndarray,
    all_normals: np.ndarray,
    semantic_labels: np.ndarray | None,
    detected_planes: list[DetectedPlane],
    floorplan_cfg: dict[str, Any],
    explicit_clutter_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Reclaim RandLA-Net misclassified wall points from `clutter` & close occluded wall cavities (Section 3.2)."""
    pts = np.asarray(all_points, dtype=np.float64)
    nrms = np.asarray(all_normals, dtype=np.float64)
    c_cfg = floorplan_cfg["clutter_recovery"]
    n_cfg = floorplan_cfg["numerical"]
    p_cfg = floorplan_cfg["hessian_plane"]

    if explicit_clutter_mask is not None:
        clutter_pool = np.nonzero(np.asarray(explicit_clutter_mask, dtype=bool))[0]
    elif semantic_labels is not None and len(semantic_labels) == len(pts):
        clutter_ids = [int(x) for x in c_cfg["clutter_semantic_labels"]]
        clutter_pool = np.nonzero(np.isin(np.asarray(semantic_labels, dtype=np.int64), clutter_ids))[0]
    else:
        clutter_pool = np.empty(0, dtype=np.int64)

    # Exclude points already assigned to any structural plane
    assigned = (
        np.unique(np.concatenate([dp.inlier_indices for dp in detected_planes]))
        if detected_planes
        else np.empty(0, dtype=np.int64)
    )
    clutter_pool = np.setdiff1d(clutter_pool, assigned, assume_unique=False)

    # Estimate vertical room bounds [z_floor, z_ceil] from floor/ceiling slabs AND wall spans (never clipped by desks)
    margin_z = float(c_cfg["vertical_margin_m"])
    floors = [float(np.median(pts[dp.inlier_indices, 2])) for dp in detected_planes if dp.orientation_class == "horizontal_floor"]
    ceils = [float(np.median(pts[dp.inlier_indices, 2])) for dp in detected_planes if dp.orientation_class == "horizontal_ceiling"]
    wall_z_mins = [float(pts[dp.inlier_indices, 2].min()) for dp in detected_planes if dp.orientation_class == "vertical_wall" and len(dp.inlier_indices)]
    wall_z_maxs = [float(pts[dp.inlier_indices, 2].max()) for dp in detected_planes if dp.orientation_class == "vertical_wall" and len(dp.inlier_indices)]

    z_min_cands = [*floors, *wall_z_mins]
    z_max_cands = [*ceils, *wall_z_maxs]
    z_min = (min(z_min_cands) if z_min_cands else float(pts[:, 2].min())) - margin_z
    z_max = (max(z_max_cands) if z_max_cands else float(pts[:, 2].max())) + margin_z

    eps_reclaim = float(c_cfg["reclaim_eps_plane_m"])
    cos_reclaim = float(np.cos(np.radians(float(c_cfg["reclaim_max_angle_deg"]))))
    total_reclaimed = 0

    if bool(c_cfg["enable_reclamation"]) and len(clutter_pool) > 0:
        for dp in detected_planes:
            if dp.orientation_class != "vertical_wall" or len(clutter_pool) == 0:
                continue
            cand_pts = pts[clutter_pool]
            cand_nrms = nrms[clutter_pool]
            dist_ok = dp.plane.distance(cand_pts) < eps_reclaim
            norm_ok = np.abs(cand_nrms @ dp.plane.normal) > cos_reclaim
            z_ok = (cand_pts[:, 2] >= z_min) & (cand_pts[:, 2] <= z_max)

            hit_local = np.nonzero(dist_ok & norm_ok & z_ok)[0]
            if len(hit_local) > 0:
                reclaimed = clutter_pool[hit_local]
                dp.reclaimed_clutter_indices = reclaimed
                dp.inlier_indices = np.union1d(dp.inlier_indices, reclaimed)
                dp.plane = HessianPlane.fit_svd(
                    pts[dp.inlier_indices],
                    norm_eps=float(n_cfg["norm_eps"]),
                    canonical_sign=bool(p_cfg["canonical_sign"]),
                )
                total_reclaimed += int(len(reclaimed))
                clutter_pool = np.delete(clutter_pool, hit_local)

    # Morphological closing (dilation followed by erosion) on 2D wall occupancy grids
    m_cfg = c_cfg["morphological_closing"]
    occupancy_grids: dict[int, np.ndarray] = {}
    if bool(m_cfg["enable"]):
        res = float(m_cfg["grid_resolution_m"])
        kw = max(1, int(np.round(float(m_cfg["kernel_width_m"]) / res)))
        kh = max(1, int(np.round(float(m_cfg["kernel_height_m"]) / res)))
        struct_elem = np.ones((kw, kh), dtype=bool)

        for idx_p, dp in enumerate(detected_planes):
            if dp.orientation_class != "vertical_wall" or len(dp.inlier_indices) < 4:
                continue
            w_pts = pts[dp.inlier_indices]
            # Horizontal tangent vector along wall: t_u = [-ny, nx, 0] / ||...||
            t_u = np.array([-dp.plane.b, dp.plane.a, 0.0], dtype=np.float64)
            t_norm = float(np.linalg.norm(t_u))
            if t_norm <= float(n_cfg["norm_eps"]):
                continue
            t_u /= t_norm

            u_coord = w_pts @ t_u
            z_coord = w_pts[:, 2]
            u0, u1 = float(u_coord.min()), float(u_coord.max())
            z0, z1 = float(z_coord.min()), float(z_coord.max())
            nu = max(2, int(np.ceil((u1 - u0) / res)) + 1)
            nz = max(2, int(np.ceil((z1 - z0) / res)) + 1)

            grid = np.zeros((nu, nz), dtype=bool)
            iu = np.clip(((u_coord - u0) / res).astype(np.int64), 0, nu - 1)
            iz = np.clip(((z_coord - z0) / res).astype(np.int64), 0, nz - 1)
            grid[iu, iz] = True

            closed = binary_closing(grid, structure=struct_elem)
            dp.gap_closed_cells = int(np.sum(closed & (~grid)))
            occupancy_grids[idx_p] = closed

    return {
        "total_reclaimed_points": total_reclaimed,
        "walls_closed": len(occupancy_grids),
        "total_gap_cells_filled": int(sum(dp.gap_closed_cells for dp in detected_planes)),
    }


# =============================================================================
# SECTION 3.3: MANHATTAN WORLD ALIGNMENT & 90-DEG RECTILINEAR REGULARIZATION
# =============================================================================

def compute_manhattan_dominant_angle(
    wall_normals: np.ndarray,
    weights: np.ndarray | None,
    manhattan_cfg: dict[str, Any],
) -> float:
    """Compute dominant horizontal wall-normal orientation theta_0 in [0, pi/2) (Section 3.3, Eq. 11):

        theta_k = atan2(ny, nx) mod (pi / 2)

    Uses 4-fold wrapped circular vector averaging when `use_fourfold_circular_mean: true` so
    angles straddling the 0 / (pi/2) periodic boundary never suffer wrap-around bias.
    """
    nrms = np.atleast_2d(np.asarray(wall_normals, dtype=np.float64))
    if len(nrms) == 0:
        return 0.0

    horiz_len = np.hypot(nrms[:, 0], nrms[:, 1])
    valid = horiz_len > 1.0e-6
    if not np.any(valid):
        return 0.0

    nx = nrms[valid, 0] / horiz_len[valid]
    ny = nrms[valid, 1] / horiz_len[valid]
    w = np.asarray(weights, dtype=np.float64)[valid] if weights is not None else np.ones(int(np.sum(valid)), dtype=np.float64)

    # Exact Eq. (11) per-normal modulo-(pi/2) angle
    phi = np.arctan2(ny, nx)
    theta_mod = np.mod(phi, 0.5 * np.pi)

    if bool(manhattan_cfg.get("use_fourfold_circular_mean", True)):
        # 4 * theta_mod wraps [0, pi/2) seamlessly onto full circle [0, 2*pi)
        mean_4 = float(np.arctan2(np.sum(w * np.sin(4.0 * theta_mod)), np.sum(w * np.cos(4.0 * theta_mod))))
        return float(np.mod(0.25 * mean_4, 0.5 * np.pi))

    bins = int(manhattan_cfg.get("histogram_bins", 180))
    hist, edges = np.histogram(theta_mod, bins=bins, range=(0.0, 0.5 * np.pi), weights=w)
    peak = int(np.argmax(hist))
    return float(0.5 * (edges[peak] + edges[peak + 1]))


def align_manhattan_world(
    points: np.ndarray,
    normals: np.ndarray | None,
    detected_planes: list[DetectedPlane],
    floorplan_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None, float, np.ndarray]:
    """Rotate scene & planes by R_z(-theta_0) to align vertical walls with world X/Y axes (Section 3.3)."""
    m_cfg = floorplan_cfg["manhattan_alignment"]
    pts = np.asarray(points, dtype=np.float64)
    nrms = np.asarray(normals, dtype=np.float64) if normals is not None else None

    if not bool(m_cfg["enable"]):
        return pts.copy(), (nrms.copy() if nrms is not None else None), 0.0, np.eye(3, dtype=np.float64)

    walls = [dp for dp in detected_planes if dp.orientation_class == "vertical_wall"]
    if walls:
        w_nrms = np.vstack([dp.plane.normal for dp in walls])
        w_weights = np.array([len(dp.inlier_indices) for dp in walls], dtype=np.float64)
    elif nrms is not None and len(nrms) > 0:
        vert_mask = np.abs(nrms[:, 2]) <= float(floorplan_cfg["hessian_plane"]["vertical_max_abs_nz"])
        w_nrms = nrms[vert_mask]
        w_weights = None
    else:
        return pts.copy(), (nrms.copy() if nrms is not None else None), 0.0, np.eye(3, dtype=np.float64)

    theta_0 = compute_manhattan_dominant_angle(w_nrms, w_weights, m_cfg)
    c_t = float(np.cos(-theta_0))
    s_t = float(np.sin(-theta_0))
    rot_minus_theta0 = np.array([
        [c_t, -s_t, 0.0],
        [s_t,  c_t, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=np.float64)

    pts_aligned = pts @ rot_minus_theta0.T
    nrms_aligned = (nrms @ rot_minus_theta0.T) if nrms is not None else None
    for dp in detected_planes:
        dp.plane = dp.plane.rotate_z(-theta_0)

    return pts_aligned, nrms_aligned, theta_0, rot_minus_theta0


def snap_segment_rectilinear(
    p_start: np.ndarray,
    p_end: np.ndarray,
    source_plane_idx: int = -1,
    max_non_manhattan_deg: float = 45.0,
) -> WallSegment2D:
    """Apply Section 3.3 Rectilinear Regularization rule to a 2D wall segment:
      - If |Delta x| >= |Delta y| -> snap to horizontal segment (y = y_mean)
      - If |Delta x| <  |Delta y| -> snap to vertical segment   (x = x_mean)
    Preserves genuine diagonal segments only if angular deviation exceeds `max_non_manhattan_deg`.
    """
    p1 = np.asarray(p_start, dtype=np.float64).reshape(2)
    p2 = np.asarray(p_end, dtype=np.float64).reshape(2)
    delta = p2 - p1
    dx, dy = float(delta[0]), float(delta[1])
    mid = 0.5 * (p1 + p2)

    seg_angle_deg = float(np.degrees(np.arctan2(abs(dy), abs(dx))))  # in [0, 90]
    dev_from_axis = min(seg_angle_deg, 90.0 - seg_angle_deg)

    if dev_from_axis > float(max_non_manhattan_deg):
        seg_len = max(float(np.hypot(dx, dy)), 1.0e-12)
        n2d = np.array([-dy / seg_len, dx / seg_len], dtype=np.float64)
        c_off = -float(np.dot(n2d, mid))
        return WallSegment2D(p1.copy(), p2.copy(), n2d, c_off, "diagonal", source_plane_idx)

    # Exact Section 3.3 rule: |Delta x| >= |Delta y| -> horizontal; else vertical
    if abs(dx) >= abs(dy):
        y_lock = float(mid[1])
        s1 = np.array([p1[0], y_lock], dtype=np.float64)
        s2 = np.array([p2[0], y_lock], dtype=np.float64)
        n2d = np.array([0.0, 1.0], dtype=np.float64)
        return WallSegment2D(s1, s2, n2d, -y_lock, "horizontal", source_plane_idx)

    x_lock = float(mid[0])
    s1 = np.array([x_lock, p1[1]], dtype=np.float64)
    s2 = np.array([x_lock, p2[1]], dtype=np.float64)
    n2d = np.array([1.0, 0.0], dtype=np.float64)
    return WallSegment2D(s1, s2, n2d, -x_lock, "vertical", source_plane_idx)


def intersect_lines_2d(
    seg_a: WallSegment2D,
    seg_b: WallSegment2D,
    det_eps: float = 1.0e-9,
) -> np.ndarray | None:
    """Compute exact analytical 2D line intersection for 90-deg corners (Section 3.3):
        [n_a^T; n_b^T] * V = [-c_a; -c_b]
    """
    a_mat = np.vstack([seg_a.normal_2d, seg_b.normal_2d])
    det = float(np.linalg.det(a_mat))
    if abs(det) <= float(det_eps):
        return None
    rhs = np.array([-seg_a.offset_c, -seg_b.offset_c], dtype=np.float64)
    return np.linalg.solve(a_mat, rhs)


def regularize_rectilinear_polygon(
    raw_vertices_2d: np.ndarray,
    floorplan_cfg: dict[str, Any],
) -> tuple[np.ndarray, list[WallSegment2D]]:
    """Regularize an ordered 2D room boundary loop into exact 90-deg Manhattan corners (Section 3.3)."""
    verts = np.asarray(raw_vertices_2d, dtype=np.float64)
    if len(verts) < 3:
        return verts.copy(), []

    # Drop duplicate closing vertex if present
    if np.linalg.norm(verts[0] - verts[-1]) <= 1.0e-6 and len(verts) > 3:
        verts = verts[:-1]

    m_cfg = floorplan_cfg["manhattan_alignment"]
    det_eps = float(floorplan_cfg["numerical"]["corner_det_eps"])
    max_dev = float(m_cfg["snap_non_manhattan_max_deg"])

    n_edges = len(verts)
    snapped: list[WallSegment2D] = []
    for i in range(n_edges):
        p_a = verts[i]
        p_b = verts[(i + 1) % n_edges]
        snapped.append(snap_segment_rectilinear(p_a, p_b, source_plane_idx=i, max_non_manhattan_deg=max_dev))

    # Merge consecutive parallel (same axis_lock) segments so adjacent edges are orthogonal
    merged: list[WallSegment2D] = []
    for seg in snapped:
        if merged and merged[-1].axis_lock == seg.axis_lock and seg.axis_lock in ("horizontal", "vertical"):
            prev = merged[-1]
            c_avg = 0.5 * (prev.offset_c + seg.offset_c)
            if seg.axis_lock == "horizontal":
                y_m = -c_avg
                merged[-1] = WallSegment2D(
                    np.array([prev.p_start[0], y_m]),
                    np.array([seg.p_end[0], y_m]),
                    prev.normal_2d,
                    c_avg,
                    "horizontal",
                    prev.source_plane_idx,
                )
            else:
                x_m = -c_avg
                merged[-1] = WallSegment2D(
                    np.array([x_m, prev.p_start[1]]),
                    np.array([x_m, seg.p_end[1]]),
                    prev.normal_2d,
                    c_avg,
                    "vertical",
                    prev.source_plane_idx,
                )
        else:
            merged.append(seg)

    if len(merged) >= 2 and merged[0].axis_lock == merged[-1].axis_lock and merged[0].axis_lock in ("horizontal", "vertical"):
        first, last = merged[0], merged[-1]
        c_avg = 0.5 * (first.offset_c + last.offset_c)
        if first.axis_lock == "horizontal":
            y_m = -c_avg
            merged[0] = WallSegment2D(
                np.array([last.p_start[0], y_m]),
                np.array([first.p_end[0], y_m]),
                first.normal_2d,
                c_avg,
                "horizontal",
                first.source_plane_idx,
            )
        else:
            x_m = -c_avg
            merged[0] = WallSegment2D(
                np.array([x_m, last.p_start[1]]),
                np.array([x_m, first.p_end[1]]),
                first.normal_2d,
                c_avg,
                "vertical",
                first.source_plane_idx,
            )
        merged.pop()

    # Compute analytical corner intersections between consecutive segments
    m_cnt = len(merged)
    if m_cnt < 3:
        return verts.copy(), merged

    corners: list[np.ndarray] = []
    for i in range(m_cnt):
        prev_seg = merged[(i - 1) % m_cnt]
        cur_seg = merged[i]
        pt = intersect_lines_2d(prev_seg, cur_seg, det_eps=det_eps)
        if pt is None:
            pt = 0.5 * (prev_seg.p_end + cur_seg.p_start)
        corners.append(pt)

    final_segs: list[WallSegment2D] = []
    for i in range(m_cnt):
        c0 = corners[i]
        c1 = corners[(i + 1) % m_cnt]
        final_segs.append(
            WallSegment2D(c0, c1, merged[i].normal_2d, merged[i].offset_c, merged[i].axis_lock, merged[i].source_plane_idx)
        )

    return np.vstack(corners), final_segs
