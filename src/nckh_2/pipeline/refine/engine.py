"""Loi tinh chinh mesh 3D (.obj/.ply) dung chung cho ca Pipeline Video, Single-File va Batch.

Thuc hien 5 pha con co the bat/tat qua cau hinh (ho tro ca dict phang lan YAML phan nhom):
  5a. Audit & Pose/Axis Skew Check (PCA/OBB + 6 mat)
  5b. Ghost-Shell Pruning & Optional Alignment (mac dinh: preserve he toa do goc)
  5c. Scale-Relative Cleanup & Topology Repair (clean_mesh_global, peel_slivers, spike filter)
  5d. Isotropic Remeshing & Drift-Monitored Volume-Preserving Smoothing
  5e. Attribute-Safe Export + JSON Ledger + 6-Side Visual Audit
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
import time
from typing import Any

import numpy as np
import open3d as o3d

from nckh_2.backends.alignment import (
    align_pca_canonical,
    align_to_reference_multistart,
    detect_and_prune_ghost_shells,
)
from nckh_2.backends.geometry import (
    MeshDiagnostics,
    clean_mesh_global,
    compute_mesh_diagnostics,
    export_obj,
    extract_obj_mtl_headers,
    ground_mesh_z0,
    make_watertight,
    peel_boundary_slivers,
    prune_excess_mesh_extrapolation,
)
from nckh_2.backends.remesh import isotropic_remesh
from nckh_2.backends.smoothing import (
    bilateral_normal_smooth,
    drift_monitored_smooth,
    interpolate_point_confidence_to_mesh,
    pymeshlab_two_step_smooth,
    suppress_mesh_spikes,
    transfer_observed_detail_residuals,
)
from nckh_2.evaluation.mesh_audit import (
    MeshAuditReport,
    audit_mesh,
    load_reference_points,
    print_audit_summary,
    save_audit_comparison_figure,
    save_stage_ledger,
)
from nckh_2.evaluation.metrics import evaluate_detail_preservation


@dataclasses.dataclass
class RefineReport:
    """Bao cao kiem dinh chi tiet truoc/sau cua loi tinh chinh mesh."""

    before: MeshDiagnostics
    after: MeshDiagnostics
    components_removed: int
    zero_area_tris_removed: int
    ghost_shells_removed: int
    boundary_slivers_peeled: int
    spikes_suppressed: int
    remesh_applied: bool
    smooth_mode: str
    elapsed_sec: float
    audit_reports: list[MeshAuditReport] = dataclasses.field(default_factory=list)
    stage_notes: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "components_removed": self.components_removed,
            "zero_area_tris_removed": self.zero_area_tris_removed,
            "ghost_shells_removed": self.ghost_shells_removed,
            "boundary_slivers_peeled": self.boundary_slivers_peeled,
            "spikes_suppressed": self.spikes_suppressed,
            "remesh_applied": self.remesh_applied,
            "smooth_mode": self.smooth_mode,
            "elapsed_sec": round(self.elapsed_sec, 4),
            "stage_notes": self.stage_notes,
        }


def _opt(cfg: dict[str, Any], section: str, key: str, default: Any, flat_aliases: tuple[str, ...] = ()) -> Any:
    """Doc tham so linh hoat tu ca YAML phan nhom (nested) lan dict phang (flat)."""
    for k in (key, *flat_aliases):
        if k in cfg:
            return cfg[k]
    sec = cfg.get(section)
    if isinstance(sec, dict):
        for k in (key, *flat_aliases):
            if k in sec:
                return sec[k]
    return default


def _snapshot_attributes(
    mesh: o3d.geometry.TriangleMesh,
) -> tuple[np.ndarray | None, np.ndarray | None, list[int] | None]:
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)
    v_colors = (
        np.asarray(mesh.vertex_colors, dtype=np.float64).copy()
        if mesh.has_vertex_colors() and len(mesh.vertex_colors) == n_v
        else None
    )
    tri_uvs = (
        np.asarray(mesh.triangle_uvs, dtype=np.float64).copy()
        if mesh.has_triangle_uvs() and len(mesh.triangle_uvs) == 3 * n_t and n_t > 0
        else None
    )
    mat_ids = (
        list(mesh.triangle_material_ids)
        if mesh.has_triangle_material_ids() and len(mesh.triangle_material_ids) == n_t and n_t > 0
        else None
    )
    return v_colors, tri_uvs, mat_ids


def _restore_attributes(
    mesh: o3d.geometry.TriangleMesh,
    v_colors: np.ndarray | None,
    tri_uvs: np.ndarray | None,
    mat_ids: list[int] | None,
) -> None:
    if v_colors is not None and len(v_colors) == len(mesh.vertices):
        mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors)
    if tri_uvs is not None and len(tri_uvs) == 3 * len(mesh.triangles):
        mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)
    if mat_ids is not None and len(mat_ids) == len(mesh.triangles):
        mesh.triangle_material_ids = o3d.utility.IntVector(mat_ids)


def refine_mesh_with_report(
    mesh: o3d.geometry.TriangleMesh,
    r_cfg: dict[str, Any] | None = None,
    ref_points: np.ndarray | None = None,
    ref_confidence: np.ndarray | None = None,
    audit_only: bool = False,
    audit_dir: Path | None = None,
    mtl_headers: list[str] | None = None,
) -> tuple[o3d.geometry.TriangleMesh, RefineReport]:
    """Loi tinh chinh 5 pha dung chung, bao toan he toa do goc & thuoc tinh."""
    t0 = time.perf_counter()
    cfg = r_cfg or {}

    out = o3d.geometry.TriangleMesh(mesh)
    diag_before = compute_mesh_diagnostics(out)

    if diag_before.num_vertices == 0 or diag_before.num_triangles == 0:
        return out, RefineReport(
            before=diag_before,
            after=diag_before,
            components_removed=0,
            zero_area_tris_removed=0,
            ghost_shells_removed=0,
            boundary_slivers_peeled=0,
            spikes_suppressed=0,
            remesh_applied=False,
            smooth_mode="none",
            elapsed_sec=time.perf_counter() - t0,
        )

    out.compute_triangle_normals()
    out.compute_vertex_normals()
    orig_snapshot = o3d.geometry.TriangleMesh(out)

    ar_bad_thr = float(_opt(cfg, "diagnostics", "aspect_ratio_bad_thr", 4.0))
    ang_bad_deg = float(_opt(cfg, "diagnostics", "min_angle_bad_deg", 15.0))
    enable_audit = bool(_opt(cfg, "audit", "enable_report", audit_dir is not None))
    save_inter = bool(_opt(cfg, "audit", "save_intermediates", False)) and (audit_dir is not None)

    audit_reports: list[MeshAuditReport] = []
    stage_notes: dict[str, Any] = {}

    # --- PHA 5a: Chan doan ban dau & Kiem tra lech truc (PCA/OBB + 6 mat) ---
    if enable_audit or audit_only:
        rep_in = audit_mesh(
            out,
            stage_label="stage0_input",
            ref_points=ref_points,
            aspect_ratio_bad_thr=ar_bad_thr,
            min_angle_bad_deg=ang_bad_deg,
        )
        audit_reports.append(rep_in)
        print_audit_summary(rep_in)

    if audit_only:
        if audit_dir is not None and audit_reports:
            save_stage_ledger(audit_reports, audit_dir / "stage_metrics.json", extra={"audit_only": True})
        return out, RefineReport(
            before=diag_before,
            after=diag_before,
            components_removed=0,
            zero_area_tris_removed=0,
            ghost_shells_removed=0,
            boundary_slivers_peeled=0,
            spikes_suppressed=0,
            remesh_applied=False,
            smooth_mode="none",
            elapsed_sec=time.perf_counter() - t0,
            audit_reports=audit_reports,
            stage_notes={"audit_only": True},
        )

    # --- PHA 5b: Loc vo trung lap (Ghost Shells) & Can truc (tuy chon) ---
    ghost_removed = 0
    if bool(_opt(cfg, "alignment", "prune_ghost_shells", False)):
        out, g_stats = detect_and_prune_ghost_shells(
            out,
            overlap_dist_frac=float(_opt(cfg, "alignment", "overlap_dist_frac", 0.015)),
            ghost_overlap_thr=float(_opt(cfg, "alignment", "ghost_overlap_thr", 0.65)),
            normal_align_cos=float(_opt(cfg, "alignment", "ghost_normal_align_cos", 0.50)),
            prune=True,
        )
        ghost_removed = int(g_stats["ghost_shells_removed"])
        stage_notes["ghost_shells"] = g_stats
        if ghost_removed > 0:
            print(f"  Ghost-shell filter: da loai {ghost_removed} vo trung lap lech pha")

    align_mode = str(_opt(cfg, "alignment", "align_mode", "preserve")).strip().lower()
    if align_mode == "pca_canonical":
        out, _ = align_pca_canonical(
            out,
            normalize_unit_scale=bool(_opt(cfg, "alignment", "normalize_unit_scale", False)),
            eigen_ratio_min=float(_opt(cfg, "diagnostics", "pca_eigen_ratio_min", 1.15)),
        )
        print("  Alignment: da can truc chinh PCA (det=+1)")
    elif align_mode == "reference_icp" and ref_points is not None:
        out, _, _ = align_to_reference_multistart(
            out,
            ref_points,
            match_scale=bool(_opt(cfg, "alignment", "match_reference_scale", False)),
            icp_max_corr_frac=float(_opt(cfg, "alignment", "icp_max_corr_frac", 0.05)),
        )
        print("  Alignment: da can khop vao tham chieu bang multi-start ICP")
    stage_notes["align_mode"] = align_mode

    # --- PHA 5c: Lam sach topo theo ty le, tia ria bien & khu gai nhon ---
    out, clean_stats = clean_mesh_global(
        out,
        min_component_tri_ratio=float(_opt(cfg, "cleanup", "min_component_tri_ratio", 0.005)),
        min_component_area_ratio=float(_opt(cfg, "cleanup", "min_component_area_ratio", 0.002)),
        min_component_triangles=int(_opt(cfg, "cleanup", "min_component_triangles", 15)),
        remove_non_manifold=bool(_opt(cfg, "cleanup", "repair_non_manifold", False, ("remove_non_manifold",))),
        orient_consistent=bool(_opt(cfg, "cleanup", "orient_consistent", False)),
    )
    peeled = 0
    peel_ar = float(_opt(cfg, "cleanup", "peel_boundary_sliver_ar", 0.0))
    if peel_ar > 0:
        out, peeled = peel_boundary_slivers(
            out,
            aspect_ratio_thr=peel_ar,
            max_passes=int(_opt(cfg, "cleanup", "peel_boundary_passes", 2)),
        )
    if clean_stats["zero_area_tris_removed"] > 0:
        print(f"  Topo clean: loai {clean_stats['zero_area_tris_removed']:,} tam giac suy bien")
    if clean_stats["components_removed"] > 0:
        kept = clean_stats["components_before"] - clean_stats["components_removed"]
        print(f"  Component filter: loai {clean_stats['components_removed']:,} manh vun, giu {kept:,}/{clean_stats['components_before']:,}")
    if peeled > 0:
        print(f"  Boundary sliver peel: loai {peeled:,} tam giac ria bien qua det")

    if bool(cfg.get("subdivide", False)) and len(out.triangles) > 0:
        s_method = str(cfg.get("subdivide_method", "midpoint")).strip().lower()
        out = out.subdivide_loop(1) if s_method == "loop" else out.subdivide_midpoint(1)

    v_colors, tri_uvs, mat_ids = _snapshot_attributes(out)

    n_spikes = 0
    if bool(_opt(cfg, "cleanup", "suppress_spikes", True)):
        out, n_spikes = suppress_mesh_spikes(
            out,
            sigma_threshold=float(_opt(cfg, "cleanup", "spike_sigma_threshold", 3.5)),
            pull_strength=float(_opt(cfg, "cleanup", "spike_pull_strength", 0.85)),
        )
        if n_spikes > 0:
            print(f"  Spike suppression: xu ly {n_spikes:,} dinh gai nhon")

    stage_notes["cleanup"] = {**clean_stats, "boundary_slivers_peeled": peeled, "spikes_suppressed": n_spikes}
    if save_inter and audit_dir is not None:
        export_obj(out, audit_dir / "stage3_cleaned.obj", mtl_headers=mtl_headers)

    # --- PHA 5d: Isotropic Remeshing & Lam min kiem soat do troi ---
    remesh_applied = False
    if bool(_opt(cfg, "remesh", "enable", False)):
        out, r_stats = isotropic_remesh(
            out,
            target_edge_frac=float(_opt(cfg, "remesh", "target_edge_frac", 0.005)),
            iterations=int(_opt(cfg, "remesh", "iterations", 3)),
            feature_angle_deg=float(_opt(cfg, "remesh", "feature_angle_deg", 35.0)),
            adaptive=bool(_opt(cfg, "remesh", "adaptive", True)),
            preserve_uvs=bool(_opt(cfg, "remesh", "preserve_uvs", True)),
        )
        remesh_applied = r_stats.applied
        stage_notes["remesh"] = dataclasses.asdict(r_stats)
        if r_stats.applied:
            print(
                f"  Isotropic remesh: {r_stats.tris_before:,} -> {r_stats.tris_after:,} tris "
                f"(target_edge={r_stats.target_edge_len:.5f}, P95_drift={r_stats.p95_drift:.5f})"
            )
            v_colors, tri_uvs, mat_ids = _snapshot_attributes(out)
            if save_inter and audit_dir is not None:
                export_obj(out, audit_dir / "stage4a_remeshed.obj", mtl_headers=mtl_headers)
        else:
            print(f"  Isotropic remesh: bo qua ({r_stats.reason})")

    smooth_mode = str(_opt(cfg, "smoothing", "mode", "bilateral_normal", ("smooth_mode",))).strip().lower()
    use_drift_guard = isinstance(cfg.get("smoothing"), dict) or ("max_drift_frac" in cfg)

    v_conf = None
    if ref_points is not None and ref_confidence is not None and len(ref_points) == len(ref_confidence):
        v_conf = interpolate_point_confidence_to_mesh(out, ref_points, ref_confidence)

    pre_smooth_mesh = o3d.geometry.TriangleMesh(out) if enable_audit else None

    if smooth_mode == "none":
        print("  Smoothing: bo qua (mode='none')")
    elif use_drift_guard and smooth_mode in ("hybrid", "taubin", "bilateral_normal"):
        out, s_stats = drift_monitored_smooth(
            out,
            mode=smooth_mode,
            batch_iters=int(_opt(cfg, "smoothing", "batch_iters", 2)),
            max_batches=int(_opt(cfg, "smoothing", "max_batches", 5)),
            max_drift_frac=float(_opt(cfg, "smoothing", "max_drift_frac", 0.004)),
            taubin_lambda=float(_opt(cfg, "smoothing", "taubin_lambda", 0.35)),
            taubin_mu=float(_opt(cfg, "smoothing", "taubin_mu", -0.37)),
            bilateral_sigma_s_scale=float(_opt(cfg, "smoothing", "bilateral_sigma_s_scale", 1.5)),
            bilateral_sigma_r=float(_opt(cfg, "smoothing", "bilateral_sigma_r", 0.25)),
            bilateral_anchor_weight=float(_opt(cfg, "smoothing", "bilateral_anchor_weight", 0.08)),
            vertex_confidence=v_conf,
        )
        stage_notes["smoothing"] = s_stats
        print(
            f"  Drift-monitored {smooth_mode}: {s_stats['total_iters']} iters | "
            f"P95_drift={s_stats['p95_drift']} ({s_stats['stopped_reason']})"
        )
    elif smooth_mode == "bilateral_normal":
        out = bilateral_normal_smooth(
            out,
            normal_iter=int(_opt(cfg, "smoothing", "bilateral_normal_iter", 4)),
            vertex_iter=int(_opt(cfg, "smoothing", "bilateral_vertex_iter", 10)),
            sigma_s_scale=float(_opt(cfg, "smoothing", "bilateral_sigma_s_scale", 1.5)),
            sigma_r=float(_opt(cfg, "smoothing", "bilateral_sigma_r", 0.25)),
            vertex_confidence=v_conf,
            anchor_weight=float(_opt(cfg, "smoothing", "bilateral_anchor_weight", 0.08)),
        )
    elif smooth_mode == "pymeshlab_twostep":
        out = pymeshlab_two_step_smooth(
            out,
            steps=int(_opt(cfg, "smoothing", "twostep_steps", 3)),
            normal_thr_deg=float(_opt(cfg, "smoothing", "twostep_normal_thr_deg", 45.0)),
        )
    else:
        taubin_iter = int(_opt(cfg, "smoothing", "taubin_iter", 8))
        if taubin_iter > 0:
            out = out.filter_smooth_taubin(
                number_of_iterations=taubin_iter,
                lambda_filter=float(_opt(cfg, "smoothing", "taubin_lambda", 0.35)),
                mu=float(_opt(cfg, "smoothing", "taubin_mu", -0.37)),
            )

    if cfg.get("detail_residual_transfer", False) and ref_points is not None and ref_confidence is not None:
        obs_pts = ref_points[ref_confidence >= 0.85]
        if len(obs_pts) >= 50:
            out = transfer_observed_detail_residuals(
                out,
                observed_points=obs_pts,
                vertex_confidence=v_conf,
                strength=float(cfg.get("detail_residual_strength", 0.25)),
            )

    if bool(cfg.get("align_upright", False)) and len(out.vertices) > 0:
        out = ground_mesh_z0(out)

    if diag_before.is_watertight and not out.is_watertight() and len(out.triangles) > 0:
        out = make_watertight(out)

    _restore_attributes(out, v_colors, tri_uvs, mat_ids)
    out.compute_triangle_normals()
    out.compute_vertex_normals()

    # --- PHA 5e: Kiem dinh sau xu ly, luu JSON & bieu do 6 mat ---
    if enable_audit:
        rep_out = audit_mesh(
            out,
            stage_label="stage4_refined",
            ref_points=ref_points,
            aspect_ratio_bad_thr=ar_bad_thr,
            min_angle_bad_deg=ang_bad_deg,
        )
        audit_reports.append(rep_out)
        print_audit_summary(rep_out)

        if pre_smooth_mesh is not None and len(pre_smooth_mesh.triangles) == len(out.triangles):
            det = evaluate_detail_preservation(pre_smooth_mesh, out)
            stage_notes["detail_retention"] = det.to_dict()

        if audit_dir is not None:
            audit_dir.mkdir(parents=True, exist_ok=True)
            ledger_path = audit_dir / "stage_metrics.json"
            save_stage_ledger(audit_reports, ledger_path, extra={"stage_notes": stage_notes})
            if bool(_opt(cfg, "audit", "render_visuals", True)):
                fig_path = audit_dir / "before_after_6side_audit.png"
                dpi = int(_opt(cfg, "audit", "render_dpi", 150))
                save_audit_comparison_figure(orig_snapshot, out, audit_reports[0], audit_reports[-1], fig_path, dpi=dpi)
                print(f"  [Audit] Da luu JSON -> {ledger_path}")
                print(f"  [Audit] Da luu PNG  -> {fig_path}")

    diag_after = compute_mesh_diagnostics(out)
    report = RefineReport(
        before=diag_before,
        after=diag_after,
        components_removed=clean_stats["components_removed"],
        zero_area_tris_removed=clean_stats["zero_area_tris_removed"],
        ghost_shells_removed=ghost_removed,
        boundary_slivers_peeled=peeled,
        spikes_suppressed=n_spikes,
        remesh_applied=remesh_applied,
        smooth_mode=smooth_mode,
        elapsed_sec=time.perf_counter() - t0,
        audit_reports=audit_reports,
        stage_notes=stage_notes,
    )
    return out, report


def refine_mesh(
    mesh: o3d.geometry.TriangleMesh,
    r_cfg: dict[str, Any] | None = None,
    ref_points: np.ndarray | None = None,
    ref_confidence: np.ndarray | None = None,
) -> o3d.geometry.TriangleMesh:
    """Wrapper gon tra ve truc tiep TriangleMesh da tinh chinh."""
    refined, _ = refine_mesh_with_report(mesh, r_cfg=r_cfg, ref_points=ref_points, ref_confidence=ref_confidence)
    return refined


def refine_obj_file(
    input_path: Path | str,
    output_path: Path | str,
    r_cfg: dict[str, Any] | None = None,
    reference_path: Path | str | None = None,
    ref_points: np.ndarray | None = None,
    ref_confidence: np.ndarray | None = None,
    audit_only: bool = False,
    audit_dir: Path | str | None = None,
) -> RefineReport:
    """Doc, tinh chinh va xuat 1 file .obj / .ply bat ky kem bao cao kiem dinh 6 mat."""
    in_p = Path(input_path)
    out_p = Path(output_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Khong tim thay file 3D dau vao: {in_p}")

    # Cho phep truyen output_path la thu muc -> tu dat ten <stem>_refined.obj
    if out_p.suffix.lower() not in (".obj", ".ply"):
        out_p.mkdir(parents=True, exist_ok=True)
        resolved_audit_dir = Path(audit_dir) if audit_dir is not None else out_p
        out_p = out_p / f"{in_p.stem}_refined.obj"
    else:
        resolved_audit_dir = Path(audit_dir) if audit_dir is not None else None

    mtl_headers = extract_obj_mtl_headers(in_p) if in_p.suffix.lower() == ".obj" else []
    mesh = o3d.io.read_triangle_mesh(str(in_p), enable_post_processing=False)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"File 3D rong hoac khong chua luoi tam giac hop le: {in_p}")

    if ref_points is None and reference_path is not None and Path(reference_path).exists():
        ref_points = load_reference_points(Path(reference_path))

    mesh_refined, report = refine_mesh_with_report(
        mesh,
        r_cfg=r_cfg,
        ref_points=ref_points,
        ref_confidence=ref_confidence,
        audit_only=audit_only,
        audit_dir=resolved_audit_dir,
        mtl_headers=mtl_headers,
    )
    if not audit_only:
        export_obj(mesh_refined, out_p, mtl_headers=mtl_headers)
    return report
