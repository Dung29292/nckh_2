"""Bu khuyet vat the 3D bang noi suy (Per-Component Symmetry + Visual Hull + Biharmonic Depth Proxy).

Hoat dong doc lap cho tung cum vat the (multi-object & shape-agnostic, 0 VRAM, CPU/SciPy):
  1. Khac Visual Hull 3D tu Mask Buoc 2 + Camera COLMAP (neu co) lam bien chan ngoai.
  2. Tach Point Cloud thanh tung cum vat the doc lap bang DBSCAN.
  3. Voi moi cum:
     - Pha A (Symmetry): Kiem chung doi xung cuc bo (truc X, Y) bang ICP overlap >= min_overlap;
       chi dien diem vao vung khuyet va kep trong Visual Hull (confidence = 0.75).
     - Pha B (Biharmonic 2.5D): Chieu len 8 camera ao -> phat hien lo khuyet (mask_occ) ->
       giai phuong trinh Biharmonic 2D thua -> nang nguoc 3D (confidence = 0.45) + truyen mau k-NN.
  4. Hop nhat diem thuc (w=1.0) + diem noi suy, uoc luong phap tuyen huong ra ngoai theo camera ao,
     tra ve (pc_filled, confidence_array, stats).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from nckh_2.backends.geometry import robust_diag
from nckh_2.pipeline.completion.visibility import (
    CalibratedView,
    VisualHull,
    build_visual_hull,
     estimate_object_frame,
    load_calibrated_views,
    render_virtual_views,
    seed_symmetry_points,
    select_consistent_stationary_views,
    solve_biharmonic_depth_proxy,
)


def _split_components_dbscan(
    pts: np.ndarray,
    eps_frac: float = 0.035,
    min_points: int = 30,
    min_ratio: float = 0.03,
) -> list[np.ndarray]:
    """Tach point cloud thanh cac cum vat the doc lap (tra ve danh sach mang chi so)."""
    n = len(pts)
    if n < max(min_points * 2, 100):
        return [np.arange(n, dtype=np.int64)]

    diag = max(robust_diag(pts), 1e-6)
    pc_tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    labels = np.asarray(pc_tmp.cluster_dbscan(eps=eps_frac * diag, min_points=min_points, print_progress=False))
    valid = labels >= 0
    if not np.any(valid):
        return [np.arange(n, dtype=np.int64)]

    counts = np.bincount(labels[valid])
    thr = max(min_points, int(counts.max() * min_ratio))
    groups = [np.nonzero(labels == cid)[0] for cid, cnt in enumerate(counts) if cnt >= thr]
    return groups if groups else [np.arange(n, dtype=np.int64)]


def _biharmonic_fill_component(
    comp_pts: np.ndarray,
    comp_cols: np.ndarray,
    vh: VisualHull | None,
    view_size: int = 256,
    gap_frac: float = 0.02,
    max_bridge_frac: float = 0.35,
    obb_pad_frac: float = 0.15,
    vh_prior_weight: float = 0.0,
    max_pts: int = 30000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Noi suy Song dieu hoa (Biharmonic 2.5D) CHI lap lo khuyet noi vi, kep chat trong OBB cum thuc."""
    if len(comp_pts) < 50:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    views = render_virtual_views(comp_pts, comp_cols, visual_hull=vh, image_size=int(view_size))
    if not views:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    diag = max(robust_diag(comp_pts), 1e-6)
    gap_rad = float(gap_frac) * diag
    max_bridge_rad = max(float(max_bridge_frac) * diag, 2.5 * gap_rad)
    obs_tree = cKDTree(comp_pts)

    # Hop bao OBB cua cum diem thuc de cam tuyet doi viec phinh bong bong ra khong gian trong
    c_center, r_obj, _ = estimate_object_frame(comp_pts)
    comp_loc = (comp_pts - c_center[None, :]) @ r_obj.T
    loc_min = comp_loc.min(axis=0)
    loc_max = comp_loc.max(axis=0)
    pad_vec = float(obb_pad_frac) * np.maximum(loc_max - loc_min, 1e-4)
    loc_lo = loc_min - pad_vec
    loc_hi = loc_max + pad_vec

    lifted_pts: list[np.ndarray] = []
    lifted_nrms: list[np.ndarray] = []

    for vw in views:
        if not np.any(vw.mask_occ):
            continue
        depth_proxy = solve_biharmonic_depth_proxy(vw, vh_prior_weight=float(vh_prior_weight))
        pts_3d = vw.unproject_pixels(vw.mask_occ, depth_proxy)
        if len(pts_3d) == 0:
            continue

        # 1. Kep chat trong hop OBB cua cum diem quan sat thuc
        pts_loc = (pts_3d - c_center[None, :]) @ r_obj.T
        in_obb = np.all((pts_loc >= loc_lo[None, :]) & (pts_loc <= loc_hi[None, :]), axis=1)
        pts_3d = pts_3d[in_obb]
        if len(pts_3d) == 0:
            continue

        # 2. Kep chat trong Visual Hull (khong dilate) neu co
        if vh is not None:
            inside = vh.contains_points(pts_3d, dilate_voxels=0)
            pts_3d = pts_3d[inside]
            if len(pts_3d) == 0:
                continue

        # 3. Chi giu diem nam trong lo khuyet thuc su va khong bay qua xa bo diem thuc
        dists, _ = obs_tree.query(pts_3d, k=1)
        pts_3d = pts_3d[(dists > gap_rad) & (dists <= max_bridge_rad)]
        if len(pts_3d) == 0:
            continue

        # Phap tuyen huong ve phia camera ao (ra ngoai be mat)
        cam_center = -vw.R.T @ vw.t
        view_dir = cam_center[None, :] - pts_3d
        v_len = np.maximum(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-12)
        lifted_pts.append(pts_3d)
        lifted_nrms.append(view_dir / v_len)

    if not lifted_pts:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    all_pts = np.vstack(lifted_pts)
    all_nrms = np.vstack(lifted_nrms)

    # Giam mat do voxel nhe cho diem noi suy de dong deu va tranh chong lap giua 8 camera ao
    vox_sz = max(0.006 * diag, 1e-4)
    pc_tmp = o3d.geometry.PointCloud()
    pc_tmp.points = o3d.utility.Vector3dVector(all_pts)
    pc_tmp.normals = o3d.utility.Vector3dVector(all_nrms)
    pc_tmp = pc_tmp.voxel_down_sample(vox_sz)
    pts_out = np.asarray(pc_tmp.points, dtype=np.float64)
    nrms_out = np.asarray(pc_tmp.normals, dtype=np.float64)

    if len(pts_out) > max_pts:
        pick = np.linspace(0, len(pts_out) - 1, max_pts, dtype=np.int64)
        pts_out = pts_out[pick]
        nrms_out = nrms_out[pick]

    # Truyen mau k-NN tu vien diem thuc gan nhat
    _, nn_idx = obs_tree.query(pts_out, k=1)
    cols_out = comp_cols[nn_idx]
    return pts_out, cols_out, nrms_out


def fill_point_cloud_holes(
    pc: o3d.geometry.PointCloud,
    sample_dir: Path | None = None,
    interp_cfg: dict[str, Any] | None = None,
    preloaded_views: list[CalibratedView] | None = None,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, dict[str, Any]]:
    """Bu khuyet point cloud toan cuc theo tung cum vat the (Symmetry + Visual Hull + Biharmonic).

    Tra ve ``(pc_filled, confidence, stats)`` voi:
      - ``confidence``: mang (N,) gia tri 1.0 (MVS thuc), 0.75 (doi xung ICP), 0.45 (Biharmonic).
    """
    cfg = interp_cfg or {}
    pts_obs = np.asarray(pc.points, dtype=np.float64)
    n_obs = len(pts_obs)
    if n_obs < 50 or not bool(cfg.get("enable", True)):
        return pc, np.ones(n_obs, dtype=np.float64), {
            "enabled": False, "components": 0, "sym_added": 0, "biharmonic_added": 0, "has_visual_hull": False
        }

    has_cols = pc.has_colors() and len(pc.colors) == n_obs
    cols_obs = np.asarray(pc.colors, dtype=np.float64) if has_cols else np.full((n_obs, 3), 0.7, dtype=np.float64)
    has_nrms = pc.has_normals() and len(pc.normals) == n_obs
    nrms_obs = np.asarray(pc.normals, dtype=np.float64) if has_nrms else None

    # 1. Khac Visual Hull 3D tu Mask Buoc 2 + Camera COLMAP (chi dung doan camera tinh dong nhat)
    vh: VisualHull | None = None
    if bool(cfg.get("use_visual_hull", True)) and (preloaded_views or (sample_dir is not None and sample_dir.exists())):
        try:
            all_views = preloaded_views if preloaded_views else load_calibrated_views(sample_dir)  # type: ignore[arg-type]
            views = preloaded_views if preloaded_views else select_consistent_stationary_views(all_views)
            if len(views) >= 2:
                vh = build_visual_hull(
                    pts_obs,
                    views,
                    grid_res=int(cfg.get("visual_hull_res", 72)),
                    pad_frac=float(cfg.get("visual_hull_pad_frac", 0.12)),
                    min_mask_consistency=float(cfg.get("visual_hull_consistency", 0.90)),
                )
                if len(views) < len(all_views) and bool(cfg.get("carve_shifted_ghosts", True)):
                    in_vh = vh.contains_points(pts_obs, dilate_voxels=1)
                    if int(np.sum(in_vh)) >= max(100, int(0.25 * len(pts_obs))):
                        removed_ghost = int(len(pts_obs) - np.sum(in_vh))
                        if removed_ghost > 0:
                            print(f"  [Visual Hull Carve] Loai {removed_ghost:,} diem bong ma do vat the dich chuyen giua video")
                            pts_obs = pts_obs[in_vh]
                            cols_obs = cols_obs[in_vh]
                            if nrms_obs is not None:
                                nrms_obs = nrms_obs[in_vh]
                            n_obs = len(pts_obs)
                            pc = o3d.geometry.PointCloud()
                            pc.points = o3d.utility.Vector3dVector(pts_obs)
                            if has_cols:
                                pc.colors = o3d.utility.Vector3dVector(cols_obs)
                            if nrms_obs is not None:
                                pc.normals = o3d.utility.Vector3dVector(nrms_obs)
        except Exception as err:
            print(f"  [Interpolation] Bo qua Visual Hull ({err})")

    # 2. Tach cum vat the doc lap (an toan cho canh 2+ vat the khac hinh dang)
    groups = _split_components_dbscan(pts_obs)
    use_sym = bool(cfg.get("use_symmetry", True))
    sym_min_ov = float(cfg.get("symmetry_min_overlap", 0.72))
    use_bih = bool(cfg.get("use_biharmonic", True))
    view_size = int(cfg.get("virtual_view_size", 256))
    max_added_total = int(float(cfg.get("max_added_ratio", 0.60)) * n_obs)

    added_pts_list: list[np.ndarray] = []
    added_cols_list: list[np.ndarray] = []
    added_nrms_list: list[np.ndarray] = []
    added_conf_list: list[np.ndarray] = []
    total_sym = 0
    total_bih = 0

    for g_idx in groups:
        c_pts = pts_obs[g_idx]
        c_cols = cols_obs[g_idx]
        c_cent = c_pts.mean(axis=0)
        cur_pts = c_pts
        cur_cols = c_cols

        # Pha A: Noi suy Doi xung cuc bo co kiem chung ICP
        if use_sym and len(c_pts) >= 100:
            sym_pts, sym_cols, sym_conf = seed_symmetry_points(
                c_pts,
                c_cols,
                bilateral_axes=[0, 1],
                rotational_order=0,
                visual_hull=vh,
                min_overlap=sym_min_ov,
                gap_frac=0.022,
            )
            if len(sym_pts) > 0:
                # Uoc luong phap tuyen huong ra ngoai tam cum (se duoc lam muot tiep o Buoc 4)
                s_nrm = sym_pts - c_cent[None, :]
                s_nrm /= np.maximum(np.linalg.norm(s_nrm, axis=1, keepdims=True), 1e-12)
                added_pts_list.append(sym_pts)
                added_cols_list.append(sym_cols)
                added_nrms_list.append(s_nrm)
                added_conf_list.append(np.full(len(sym_pts), 0.75, dtype=np.float64))
                total_sym += len(sym_pts)
                cur_pts = np.vstack([cur_pts, sym_pts])
                cur_cols = np.vstack([cur_cols, sym_cols])

        # Pha B: Noi suy Song dieu hoa (Biharmonic 2.5D) tren 8 camera ao (chi lap lo noi vi)
        if use_bih and len(cur_pts) >= 80:
            bih_pts, bih_cols, bih_nrms = _biharmonic_fill_component(
                cur_pts,
                cur_cols,
                vh=vh,
                view_size=view_size,
                gap_frac=0.022,
                max_bridge_frac=float(cfg.get("biharmonic_max_bridge_frac", 0.35)),
                obb_pad_frac=float(cfg.get("obb_pad_frac", 0.15)),
                vh_prior_weight=float(cfg.get("biharmonic_vh_prior_weight", 0.0)),
                max_pts=max(2000, max_added_total // max(len(groups), 1)),
            )
            if len(bih_pts) > 0:
                added_pts_list.append(bih_pts)
                added_cols_list.append(bih_cols)
                added_nrms_list.append(bih_nrms)
                added_conf_list.append(np.full(len(bih_pts), 0.45, dtype=np.float64))
                total_bih += len(bih_pts)

    if not added_pts_list:
        return pc, np.ones(n_obs, dtype=np.float64), {
            "enabled": True,
            "components": len(groups),
            "sym_added": 0,
            "biharmonic_added": 0,
            "has_visual_hull": vh is not None,
        }

    add_pts = np.vstack(added_pts_list)
    add_cols = np.vstack(added_cols_list)
    add_nrms = np.vstack(added_nrms_list)
    add_conf = np.concatenate(added_conf_list)

    # Gioi han tong so diem noi suy <= max_added_ratio * n_obs
    if len(add_pts) > max_added_total and max_added_total > 0:
        pick = np.linspace(0, len(add_pts) - 1, max_added_total, dtype=np.int64)
        add_pts = add_pts[pick]
        add_cols = add_cols[pick]
        add_nrms = add_nrms[pick]
        add_conf = add_conf[pick]

    all_pts = np.vstack([pts_obs, add_pts])
    all_conf = np.concatenate([np.ones(n_obs, dtype=np.float64), add_conf])

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(all_pts)
    if has_cols:
        out.colors = o3d.utility.Vector3dVector(np.vstack([cols_obs, add_cols]))
    if nrms_obs is not None:
        out.normals = o3d.utility.Vector3dVector(np.vstack([nrms_obs, add_nrms]))

    stats = {
        "enabled": True,
        "components": len(groups),
        "sym_added": int(total_sym),
        "biharmonic_added": int(total_bih),
        "total_added": int(len(add_pts)),
        "has_visual_hull": vh is not None,
    }
    print(
        f"  Interpolation fill ({len(groups)} comp, VH={vh is not None}): "
        f"+{total_sym:,} symmetry, +{total_bih:,} biharmonic -> {len(all_pts):,} diem"
    )
    return out, all_conf, stats
