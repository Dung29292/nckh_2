"""Explicit Surface/Object Problem Router & End-to-End Mathematical Orchestrator (Section 1, 3, 4, 6).

Strictly implements Principle 2 of Section 6 (nckh.pdf):
  'Classify the geometric problem BEFORE selecting the estimation solver'
  1. `architectural_planar` : Dual-Validated Iterative RANSAC (Eq. 1-2, 10) + Quadrics (Eq. 3-5)
                              + Clutter Reclamation + Manhattan Alignment (Eq. 11) -> .dxf & .ifc
  2. `freeform_object`      : Differential Quadratic Monge MLS (Eq. 6) -> Screened Poisson
                              (Delta chi = div V, Eq. 12) + Marching Cubes -> Volume-Preserving
                              Taubin Smoothing (lambda > 0, mu < -lambda) -> _refined.obj
  3. `room_layout`          : Top-Down Canonical Manhattan Room Prior + Local Residual RANSAC
                              (Eq. 13, 15, 16) + Clutter/Gap Closing -> .dxf & .ifc
  4. `topdown_object`       : Top-Down Superquadric/Poisson Canonical Prior (Eq. 14) + Elastic
                              Residual Field (Eq. 13, 15, 16) -> _refined.obj
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from nckh_2.backends.cad_bim_export import (
    export_bim_ifc,
    export_floorplan_dxf,
    validate_room_polygon_for_export,
)
from nckh_2.backends.geometry import export_obj, robust_diag, smooth_point_cloud_normals
from nckh_2.backends.ransac_manhattan import (
    WallSegment2D,
    align_manhattan_world,
    iterative_ransac_planes,
    reclaim_clutter_and_close_gaps,
    regularize_rectilinear_polygon,
)
from nckh_2.backends.smoothing import drift_monitored_smooth, taubin_volume_preserving_smooth
from nckh_2.backends.surfaces import CircularCylinder, SphereSurface, mls_monge_project_point_cloud
from nckh_2.backends.topdown_residual import (
    reconstruct_object_topdown_residual,
    reconstruct_room_topdown_residual,
)
from nckh_2.core.config import load_floorplan_config


def classify_geometry_problem(
    points: np.ndarray,
    normals: np.ndarray | None,
    cfg: dict[str, Any],
    floorplan_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Automatically classify 3D point cloud into one of the canonical mathematical problem classes."""
    r_cfg = cfg["surface_router"]
    explicit_mode = str(r_cfg.get("mode", "auto")).strip().lower()
    if cfg.get("topdown_residual", {}).get("enable", False) and explicit_mode == "auto":
        explicit_mode = "topdown_object"

    if explicit_mode != "auto":
        return {"route": explicit_mode, "reason": "configured_explicitly"}

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 30:
        return {"route": "freeform_object", "reason": "sparse_point_cloud"}

    extents = pts.max(axis=0) - pts.min(axis=0)
    horiz_span = float(max(extents[0], extents[1]))

    if normals is not None and len(normals) == len(pts):
        nrms = np.asarray(normals, dtype=np.float64)
    else:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        diag = max(robust_diag(pts), 1.0e-6)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03 * diag, max_nn=30))
        nrms = np.asarray(pcd.normals, dtype=np.float64)

    nrms = nrms / np.maximum(np.linalg.norm(nrms, axis=1, keepdims=True), 1.0e-12)

    fp_cfg = floorplan_cfg or load_floorplan_config()
    vert_max_nz = float(fp_cfg["hessian_plane"]["vertical_max_abs_nz"])
    vert_wall_mask = np.abs(nrms[:, 2]) <= vert_max_nz
    vert_ratio = float(np.mean(vert_wall_mask))

    # Spherical normal histogram concentration (planar scenes concentrate normals in <= 6 sharp peaks;
    # ergonomic freeform surfaces like mouse_01 disperse normals smoothly across the hemisphere)
    az_bins = int(r_cfg["histogram_azimuth_bins"])
    el_bins = int(r_cfg["histogram_elevation_bins"])
    az = np.arctan2(nrms[:, 1], nrms[:, 0])
    el = np.arcsin(np.clip(nrms[:, 2], -1.0, 1.0))
    hist2d, _, _ = np.histogram2d(
        az, el,
        bins=[az_bins, el_bins],
        range=[[-np.pi, np.pi], [-0.5 * np.pi, 0.5 * np.pi]],
    )
    top6_ratio = float(np.sum(np.sort(hist2d.ravel())[-6:]) / max(len(pts), 1))

    # Count distinct vertical wall azimuth peaks for full-room enclosure check
    distinct_wall_dirs = 0
    if int(np.sum(vert_wall_mask)) >= 50:
        w_az = np.arctan2(nrms[vert_wall_mask, 1], nrms[vert_wall_mask, 0])
        az_hist, _ = np.histogram(w_az, bins=8, range=(-np.pi, np.pi))
        distinct_wall_dirs = int(np.sum(az_hist >= 0.08 * np.sum(vert_wall_mask)))

    is_room_scale = horiz_span >= float(r_cfg["room_min_extent_m"])
    has_vertical_walls = vert_ratio >= float(r_cfg["vertical_wall_min_ratio"])
    is_planar_peaked = top6_ratio >= float(r_cfg["planar_normal_peak_ratio"])

    if is_room_scale and has_vertical_walls and distinct_wall_dirs >= int(r_cfg["enclosure_min_walls"]):
        return {
            "route": "room_layout",
            "reason": "room_scale_multi_wall_enclosure",
            "horiz_span_m": round(horiz_span, 3),
            "vertical_normal_ratio": round(vert_ratio, 3),
            "distinct_wall_dirs": distinct_wall_dirs,
            "top6_normal_peak_ratio": round(top6_ratio, 3),
        }

    if is_room_scale and (is_planar_peaked or has_vertical_walls):
        return {
            "route": "architectural_planar",
            "reason": "architectural_scale_planar_or_quadric",
            "horiz_span_m": round(horiz_span, 3),
            "vertical_normal_ratio": round(vert_ratio, 3),
            "top6_normal_peak_ratio": round(top6_ratio, 3),
        }

    return {
        "route": "freeform_object",
        "reason": "continuous_freeform_curvature",
        "horiz_span_m": round(horiz_span, 3),
        "top6_normal_peak_ratio": round(top6_ratio, 3),
    }


def run_freeform_differential_pipeline(
    pc: o3d.geometry.PointCloud,
    cfg: dict[str, Any],
    out_dir: Path | None = None,
    stem: str = "object",
) -> tuple[o3d.geometry.TriangleMesh, dict[str, Any]]:
    """Execute exact Section 3.4 Freeform Pipeline (MLS Eq. 6 -> Poisson Eq. 12 -> Taubin Smoothing)."""
    e_cfg = cfg["mesh_export"]
    num_cfg = cfg["numerical"]
    r_cfg = cfg["mesh_refine"]

    pts = np.asarray(pc.points, dtype=np.float64)
    diag = max(robust_diag(pts), float(num_cfg["norm_eps"]))

    pcd = o3d.geometry.PointCloud(pc)
    if not pcd.has_normals() or len(pcd.normals) != len(pcd.points):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=float(e_cfg["normal_radius_frac"]) * diag,
                max_nn=int(e_cfg["normal_nn"]),
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=int(e_cfg["orient_k"]))

    # Stage 1: Differential Quadratic Monge Patch MLS (Eq. 6)
    pcd_mls = mls_monge_project_point_cloud(
        pcd,
        iterations=int(e_cfg["mls_iter"]),
        k_neighbors=int(e_cfg["mls_neighbors"]),
        alpha=float(e_cfg["mls_alpha"]),
        outlier_sigma=float(e_cfg["mls_outlier_sigma"]),
        radius=float(e_cfg["mls_radius_frac"]) * diag,
        bandwidth=float(e_cfg["mls_weight_bandwidth_frac"]) * diag,
        reg_eps=float(num_cfg["svd_reg_eps"]),
        recompute_normals=bool(e_cfg["mls_recompute_normals"]),
        chunk_size=int(e_cfg["mls_chunk_size"]),
    )
    if int(e_cfg["normal_smooth_iter"]) > 0:
        pcd_mls = smooth_point_cloud_normals(
            pcd_mls,
            iterations=int(e_cfg["normal_smooth_iter"]),
            k=int(e_cfg["normal_smooth_k"]),
            sigma_angle_deg=float(e_cfg["normal_smooth_sigma_deg"]),
        )

    # Stage 2: Screened Poisson Surface Reconstruction Delta chi = div V (Eq. 12) + Marching Cubes
    raw_mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_mls,
        depth=int(e_cfg["poisson_depth"]),
        scale=float(e_cfg["poisson_scale"]),
        linear_fit=bool(e_cfg["poisson_linear_fit"]),
    )
    dens_arr = np.asarray(dens, dtype=np.float64)
    trim_pct = float(e_cfg["poisson_density_trim_percentile"])
    if trim_pct > 0.0 and len(raw_mesh.vertices) > 0:
        raw_mesh.remove_vertices_by_mask(dens_arr < np.percentile(dens_arr, trim_pct))

    max_dist_frac = float(e_cfg.get("poisson_max_dist_frac", 0.0))
    if max_dist_frac > 0.0 and len(raw_mesh.vertices) > 0:
        from scipy.spatial import cKDTree
        v_raw = np.asarray(raw_mesh.vertices, dtype=np.float64)
        d_nn, _ = cKDTree(np.asarray(pcd_mls.points, dtype=np.float64)).query(v_raw, k=1)
        far = d_nn > (max_dist_frac * diag)
        if np.any(far) and int(np.sum(~far)) > 50:
            raw_mesh.remove_vertices_by_mask(far)

    raw_mesh.remove_degenerate_triangles()
    raw_mesh.remove_unreferenced_vertices()
    raw_mesh.compute_triangle_normals()
    raw_mesh.compute_vertex_normals()

    if out_dir is not None:
        export_obj(raw_mesh, out_dir / f"{stem}_raw.obj")

    # Stage 3: Alternating Two-Step Volume-Preserving Taubin Smoothing (lambda > 0, mu < -lambda)
    lam = float(e_cfg["smooth_lambda"])
    mu = float(e_cfg["smooth_mu"])
    smoothed = taubin_volume_preserving_smooth(
        raw_mesh,
        iterations=int(e_cfg["smooth_iter"]),
        lambda_pos=lam,
        mu_neg=mu,
    )
    # Apply Stage 5 drift guard if configured
    s_cfg = r_cfg.get("smoothing", {})
    refined, s_stats = drift_monitored_smooth(
        smoothed,
        mode="taubin",
        batch_iters=int(s_cfg.get("batch_iters", 2)),
        max_batches=int(s_cfg.get("max_batches", 4)),
        max_drift_frac=float(s_cfg.get("max_drift_frac", 0.008)),
        taubin_lambda=lam,
        taubin_mu=mu,
    )

    if out_dir is not None:
        export_obj(refined, out_dir / f"{stem}_refined.obj")

    return refined, {
        "pipeline": "freeform_mls_poisson_taubin",
        "mls_points": int(len(pcd_mls.points)),
        "raw_vertices": int(len(raw_mesh.vertices)),
        "refined_vertices": int(len(refined.vertices)),
        "refined_triangles": int(len(refined.triangles)),
        "smoothing": s_stats,
    }


def route_and_reconstruct(
    pc: o3d.geometry.PointCloud,
    cfg: dict[str, Any],
    out_dir: Path | str,
    stem: str = "scene",
    floorplan_cfg: dict[str, Any] | None = None,
    semantic_labels: np.ndarray | None = None,
    prior_polygon_2d: np.ndarray | None = None,
) -> dict[str, Any]:
    """Classify input 3D cloud and execute the mathematically matched reconstruction pipeline."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fp_cfg = floorplan_cfg or cfg.get("floorplan") or load_floorplan_config()
    pts = np.asarray(pc.points, dtype=np.float64)
    nrms = np.asarray(pc.normals, dtype=np.float64) if pc.has_normals() and len(pc.normals) == len(pts) else None

    decision = classify_geometry_problem(pts, nrms, cfg, fp_cfg)
    route = decision["route"]

    result: dict[str, Any] = {"routing": decision, "outputs": {}}

    if route == "room_layout":
        room_res = reconstruct_room_topdown_residual(
            pts, nrms, fp_cfg, semantic_labels=semantic_labels, prior_polygon_2d=prior_polygon_2d
        )
        # Check remaining non-wall points for circular architectural columns (Eq. 4)
        columns: list[CircularCylinder] = []
        if bool(fp_cfg["quadrics"]["enable_column_detection"]):
            wall_idx = (
                np.unique(np.concatenate([w.inlier_indices for w in room_res["detected_walls"]]))
                if room_res["detected_walls"]
                else np.empty(0, dtype=np.int64)
            )
            rem_mask = np.ones(len(pts), dtype=bool)
            rem_mask[wall_idx] = False
            # Exclude horizontal slab points
            rem_mask &= np.abs(room_res["aligned_normals"][:, 2]) <= float(fp_cfg["hessian_plane"]["vertical_max_abs_nz"])
            rem_pts = room_res["aligned_points"][rem_mask]
            rem_nrms = room_res["aligned_normals"][rem_mask]
            if len(rem_pts) >= int(fp_cfg["quadrics"]["min_inliers"]):
                cyl, _ = CircularCylinder.fit_ransac_with_normals(
                    rem_pts, rem_nrms, fp_cfg["quadrics"], fp_cfg["numerical"], constrain_vertical=True
                )
                if cyl is not None:
                    columns.append(cyl)

        val_gate = validate_room_polygon_for_export(room_res["actual_polygon_2d"], fp_cfg["cad_bim_export"])
        result["validation"] = val_gate
        result["manhattan_theta0_deg"] = round(float(np.degrees(room_res["manhattan_theta0_rad"])), 4)
        result["wall_residual_shifts_m"] = room_res["wall_residual_shifts_m"]
        result["clutter_recovery"] = room_res["clutter_recovery"]
        result["columns_detected"] = len(columns)

        if bool(fp_cfg["cad_bim_export"]["export_dxf"]):
            dxf_p = export_floorplan_dxf(
                out_path / f"{stem}_floorplan.dxf",
                room_res["wall_segments_2d"],
                room_res["actual_polygon_2d"],
                columns,
                fp_cfg["cad_bim_export"],
            )
            result["outputs"]["dxf"] = str(dxf_p)
        if bool(fp_cfg["cad_bim_export"]["export_ifc"]):
            ifc_p = export_bim_ifc(
                out_path / f"{stem}_bim.ifc",
                room_res["wall_segments_2d"],
                room_res["actual_polygon_2d"],
                columns,
                room_res["z_floor"],
                room_res["z_ceiling"],
                fp_cfg["topdown_room"],
                fp_cfg["cad_bim_export"],
            )
            result["outputs"]["ifc"] = str(ifc_p)

    elif route == "architectural_planar":
        planes, rem_idx = iterative_ransac_planes(pts, nrms, fp_cfg)
        nrms_ensured = (
            nrms if nrms is not None else np.asarray(pc.normals, dtype=np.float64)
        )
        if nrms_ensured is None or len(nrms_ensured) != len(pts):
            from nckh_2.backends.ransac_manhattan import _ensure_unit_normals
            nrms_ensured = _ensure_unit_normals(pts, None, fp_cfg["adaptive_ransac"], fp_cfg["numerical"])

        clutter_stats = reclaim_clutter_and_close_gaps(pts, nrms_ensured, semantic_labels, planes, fp_cfg)
        pts_al, nrms_al, theta_0, _ = align_manhattan_world(pts, nrms_ensured, planes, fp_cfg)
        assert nrms_al is not None

        # Fit quadrics (Cylinder Eq. 4 & Sphere Eq. 5) on un-assigned remainder points
        columns = []
        spheres: list[SphereSurface] = []
        if len(rem_idx) >= int(fp_cfg["quadrics"]["min_inliers"]):
            if bool(fp_cfg["quadrics"]["enable_column_detection"]):
                cyl, cyl_in = CircularCylinder.fit_ransac_with_normals(
                    pts_al[rem_idx], nrms_al[rem_idx], fp_cfg["quadrics"], fp_cfg["numerical"]
                )
                if cyl is not None:
                    columns.append(cyl)
                    rem_idx = np.delete(rem_idx, cyl_in)
            if bool(fp_cfg["quadrics"]["enable_sphere_detection"]) and len(rem_idx) >= int(fp_cfg["quadrics"]["min_inliers"]):
                try:
                    sph = SphereSurface.fit_least_squares(pts_al[rem_idx])
                    if int(np.sum(sph.inlier_mask(pts_al[rem_idx], float(fp_cfg["quadrics"]["eps_sphere_m"])))) >= int(fp_cfg["quadrics"]["min_inliers"]):
                        spheres.append(sph)
                except Exception:
                    pass

        # Build 2D segments from vertical walls
        wall_segs: list[WallSegment2D] = []
        for i, dp in enumerate(planes):
            if dp.orientation_class != "vertical_wall":
                continue
            w_xy = pts_al[dp.inlier_indices, :2]
            t2d = np.array([-dp.plane.b, dp.plane.a], dtype=np.float64)
            t2d /= max(float(np.linalg.norm(t2d)), 1.0e-12)
            proj = w_xy @ t2d
            c_pt = -dp.plane.d * dp.plane.normal[:2]
            p0 = c_pt + float(proj.min()) * t2d
            p1 = c_pt + float(proj.max()) * t2d
            from nckh_2.backends.ransac_manhattan import snap_segment_rectilinear
            wall_segs.append(
                snap_segment_rectilinear(p0, p1, i, float(fp_cfg["manhattan_alignment"]["snap_non_manhattan_max_deg"]))
            )

        floors = [np.median(pts_al[dp.inlier_indices, 2]) for dp in planes if dp.orientation_class == "horizontal_floor"]
        ceils = [np.median(pts_al[dp.inlier_indices, 2]) for dp in planes if dp.orientation_class == "horizontal_ceiling"]
        zf = float(min(floors)) if floors else float(pts_al[:, 2].min())
        zc = float(max(ceils)) if ceils else zf + float(fp_cfg["topdown_room"]["fallback_room_height_m"])

        if bool(fp_cfg["cad_bim_export"]["export_dxf"]):
            dxf_p = export_floorplan_dxf(out_path / f"{stem}_floorplan.dxf", wall_segs, None, columns, fp_cfg["cad_bim_export"])
            result["outputs"]["dxf"] = str(dxf_p)
        if bool(fp_cfg["cad_bim_export"]["export_ifc"]):
            ifc_p = export_bim_ifc(
                out_path / f"{stem}_bim.ifc", wall_segs, None, columns, zf, zc, fp_cfg["topdown_room"], fp_cfg["cad_bim_export"]
            )
            result["outputs"]["ifc"] = str(ifc_p)

        result["planes"] = [dp.to_dict() for dp in planes]
        result["clutter_recovery"] = clutter_stats
        result["columns_detected"] = len(columns)
        result["spheres_detected"] = len(spheres)
        result["manhattan_theta0_deg"] = round(float(np.degrees(theta_0)), 4)

    elif route == "topdown_object":
        decomp, sq_prior = reconstruct_object_topdown_residual(pc, cfg)
        canon_p = out_path / f"{stem}_canonical.obj"
        ref_p = out_path / f"{stem}_refined.obj"
        export_obj(decomp.canonical_mesh, canon_p)
        export_obj(decomp.actual_mesh, ref_p)
        result["topdown_decomposition"] = decomp.to_dict()
        if sq_prior is not None:
            result["superquadric"] = {
                "semi_axes": [round(float(x), 6) for x in sq_prior.semi_axes],
                "eps1": round(float(sq_prior.eps1), 4),
                "eps2": round(float(sq_prior.eps2), 4),
            }
        result["outputs"]["canonical_obj"] = str(canon_p)
        result["outputs"]["refined_obj"] = str(ref_p)

    else:  # freeform_object (e.g., mouse_01)
        _, ff_stats = run_freeform_differential_pipeline(pc, cfg, out_dir=out_path, stem=stem)
        result["freeform"] = ff_stats
        result["outputs"]["raw_obj"] = str(out_path / f"{stem}_raw.obj")
        result["outputs"]["refined_obj"] = str(out_path / f"{stem}_refined.obj")

    report_path = out_path / f"{stem}_routing_report.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["outputs"]["report_json"] = str(report_path)
    return result
