"""Data-derived statistical parameters and GPU memory budgeting (Zero Hardcoded Constants).

Computes scale-invariant and density-aware geometric quantities directly from observed
point clouds and runtime hardware measurements:
  - Characteristic point spacing s_bar = median_i ||p_i - NN_1(p_i)||
  - Robust surface noise sigma_hat = 1.4826 * MAD(sqrt(lambda_min_i))
  - Universal chi-square outlier threshold sqrt(chi2_{d, 1-alpha}) * sigma_hat
  - Adaptive octree depth D = ceil(log2(max_extent / (c_D * s_bar)))
  - Measured free-VRAM MVS resolution budget
  - Largest-relative-gap component retention (replaces manual multi-object profiles)
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import chi2

# Exact mathematical normalization factor converting MAD to Gaussian standard deviation:
# 1 / Phi^{-1}(0.75) = 1.482602218505602
MAD_TO_SIGMA: float = 1.482602218505602


def chi2_radius(dof: int, alpha: float) -> float:
    """Return sqrt(chi2_{dof, 1-alpha}) outlier radius for a given significance level alpha."""
    a = float(np.clip(alpha, 1.0e-9, 0.5))
    return float(math.sqrt(float(chi2.ppf(1.0 - a, df=max(int(dof), 1)))))


def compute_local_spacings(points: np.ndarray, k: int = 8) -> np.ndarray:
    """Compute per-point local spacing (median distance to k nearest neighbors)."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    k_eff = min(max(int(k), 1) + 1, n)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k_eff, workers=-1)
    if k_eff == 2:
        return np.asarray(dists[:, 1], dtype=np.float64)
    return np.median(dists[:, 1:], axis=1).astype(np.float64)


@dataclasses.dataclass
class AdaptivePointStats:
    """Data-derived geometric statistics for a single point cloud."""

    num_points: int
    point_spacing: float
    noise_sigma: float
    extent: np.ndarray
    diag: float
    k_neighbors: int
    alpha: float
    chi2_1d: float
    chi2_3d: float
    inlier_tol: float
    gap_tol: float
    poisson_depth: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_points": int(self.num_points),
            "point_spacing": float(self.point_spacing),
            "noise_sigma": float(self.noise_sigma),
            "extent": [float(v) for v in self.extent],
            "diag": float(self.diag),
            "k_neighbors": int(self.k_neighbors),
            "alpha": float(self.alpha),
            "chi2_1d": float(self.chi2_1d),
            "chi2_3d": float(self.chi2_3d),
            "inlier_tol": float(self.inlier_tol),
            "gap_tol": float(self.gap_tol),
            "poisson_depth": int(self.poisson_depth),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def compute_adaptive_point_stats(
    points: np.ndarray,
    cfg_adapt: dict[str, Any],
    norm_eps: float = 1.0e-12,
    sample_cap: int = 8000,
    seed: int = 42,
) -> AdaptivePointStats:
    """Derive characteristic spacing, noise scale, outlier tolerances, and octree depth from points."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    alpha = float(cfg_adapt["alpha"])
    k_min = int(cfg_adapt["k_min"])
    k_max = int(cfg_adapt["k_max"])
    k_rad_mult = float(cfg_adapt["k_radius_mult"])
    spacing_floor_mult = float(cfg_adapt["spacing_floor_mult"])
    gap_mult = float(cfg_adapt["gap_spacing_mult"])
    cell_mult = float(cfg_adapt["poisson_cell_spacing_mult"])
    d_min, d_max = int(cfg_adapt["poisson_depth_bounds"][0]), int(cfg_adapt["poisson_depth_bounds"][1])

    c1 = chi2_radius(1, alpha)
    c3 = chi2_radius(3, alpha)

    if n < 4:
        return AdaptivePointStats(
            num_points=n,
            point_spacing=norm_eps,
            noise_sigma=norm_eps,
            extent=np.zeros(3, dtype=np.float64),
            diag=norm_eps,
            k_neighbors=max(1, min(k_min, n)),
            alpha=alpha,
            chi2_1d=c1,
            chi2_3d=c3,
            inlier_tol=norm_eps,
            gap_tol=norm_eps,
            poisson_depth=d_min,
        )

    p_lo = np.percentile(pts, 1.0, axis=0)
    p_hi = np.percentile(pts, 99.0, axis=0)
    extent = np.maximum(p_hi - p_lo, norm_eps)
    diag = max(float(np.linalg.norm(extent)), norm_eps)

    tree = cKDTree(pts)
    rng = np.random.default_rng(seed)
    if n > sample_cap:
        sub_idx = rng.choice(n, size=sample_cap, replace=False)
        sub_pts = pts[sub_idx]
    else:
        sub_pts = pts

    # 1. Characteristic point spacing s_bar = median ||p_i - NN_1(p_i)||
    d_nn2, _ = tree.query(sub_pts, k=2, workers=-1)
    nn1 = d_nn2[:, 1]
    pos_nn1 = nn1[nn1 > norm_eps]
    s_bar = float(np.median(pos_nn1)) if len(pos_nn1) > 0 else max(diag / math.sqrt(float(n)), norm_eps)

    # 2. Adaptive neighborhood size k from local ball of radius k_radius_mult * s_bar
    rad_k = k_rad_mult * s_bar
    counts = tree.query_ball_point(sub_pts[: min(len(sub_pts), 2000)], r=rad_k, return_length=True)
    k_med = int(round(float(np.median(counts))))
    k_eff = int(np.clip(k_med, min(k_min, n - 1), min(k_max, n - 1)))
    k_eff = max(k_eff, min(3, n - 1))

    # 3. Robust surface noise sigma_hat = 1.4826 * MAD(sqrt(lambda_min_i))
    _, nbr_idx = tree.query(sub_pts[: min(len(sub_pts), 3000)], k=k_eff + 1, workers=-1)
    nbr_pts = pts[nbr_idx]  # (M, k+1, 3)
    centered = nbr_pts - np.mean(nbr_pts, axis=1, keepdims=True)
    covs = np.einsum("mki,mkj->mij", centered, centered) / float(max(k_eff, 1))
    eigvals = np.linalg.eigvalsh(covs)  # ascending: eigvals[:, 0] is smallest
    lam_min = np.maximum(eigvals[:, 0], 0.0)
    local_std = np.sqrt(lam_min)
    med_std = float(np.median(local_std))
    mad_std = float(np.median(np.abs(local_std - med_std)))
    sigma_hat = max(MAD_TO_SIGMA * mad_std, med_std, spacing_floor_mult * s_bar * 0.25, norm_eps)

    # 4. Universal tolerances derived from (sigma_hat, s_bar, alpha)
    inlier_tol = max(c1 * sigma_hat, spacing_floor_mult * s_bar)
    gap_tol = max(gap_mult * s_bar, inlier_tol)

    # 5. Adaptive Poisson octree depth D = ceil(log2(max_extent / (c_D * s_bar)))
    max_ext = float(np.max(extent))
    ratio = max(max_ext / max(cell_mult * s_bar, norm_eps), 2.0)
    depth = int(np.clip(math.ceil(math.log2(ratio)), d_min, d_max))

    return AdaptivePointStats(
        num_points=n,
        point_spacing=s_bar,
        noise_sigma=sigma_hat,
        extent=extent,
        diag=diag,
        k_neighbors=k_eff,
        alpha=alpha,
        chi2_1d=c1,
        chi2_3d=c3,
        inlier_tol=inlier_tol,
        gap_tol=gap_tol,
        poisson_depth=depth,
    )


def estimate_vram_mvs_max_image_size(
    cfg_adapt: dict[str, Any],
) -> tuple[int, dict[str, float]]:
    """Derive MVS max image resolution from measured free GPU VRAM (no hardcoded 1400 px)."""
    safety_frac = float(cfg_adapt["vram_safety_frac"])
    bytes_per_px = float(cfg_adapt["mvs_bytes_per_pixel_est"])
    lo_sz, hi_sz = int(cfg_adapt["mvs_size_bounds"][0]), int(cfg_adapt["mvs_size_bounds"][1])

    free_bytes = 0.0
    total_bytes = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            free_bytes = float(free_b)
            total_bytes = float(total_b)
    except Exception:
        pass

    if free_bytes <= 0.0:
        # CPU or unavailable CUDA: choose conservative midpoint inside configured bounds
        mid = int(round(0.5 * (lo_sz + hi_sz)))
        return mid, {"free_vram_gb": 0.0, "total_vram_gb": 0.0, "budget_gb": 0.0, "derived_max_image_size": float(mid)}

    budget_bytes = free_bytes * safety_frac
    # Assuming ~16:9 image aspect (area ~ 0.5625 * max_side^2)
    max_pixels = budget_bytes / max(bytes_per_px, 1.0)
    est_side = int(round(math.sqrt(max_pixels / 0.5625) / 32.0) * 32)
    derived = int(np.clip(est_side, lo_sz, hi_sz))
    return derived, {
        "free_vram_gb": round(free_bytes / (1024.0 ** 3), 3),
        "total_vram_gb": round(total_bytes / (1024.0 ** 3), 3),
        "budget_gb": round(budget_bytes / (1024.0 ** 3), 3),
        "derived_max_image_size": float(derived),
    }


def largest_gap_component_mask(
    sizes: np.ndarray,
    min_abs_size: int = 1,
    gap_ratio_threshold: float = 3.0,
) -> np.ndarray:
    """Keep significant connected components via largest log-size gap (replaces manual multi-object profiles).

    Automatically separates multiple real objects of comparable scale from tiny dust fragments.
    """
    arr = np.asarray(sizes, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n == 1:
        return arr >= float(min_abs_size)

    order = np.argsort(arr)[::-1]
    sorted_s = arr[order]
    valid = sorted_s >= float(min_abs_size)
    if not np.any(valid):
        keep = np.zeros(n, dtype=bool)
        keep[order[0]] = True
        return keep

    ratios = sorted_s[:-1] / np.maximum(sorted_s[1:], 1.0e-12)
    best_split = int(np.argmax(ratios))
    if float(ratios[best_split]) >= float(gap_ratio_threshold):
        keep_count = best_split + 1
    else:
        keep_count = int(np.sum(valid))

    keep = np.zeros(n, dtype=bool)
    keep[order[:keep_count]] = True
    keep &= arr >= float(min_abs_size)
    if not np.any(keep):
        keep[order[0]] = True
    return keep
