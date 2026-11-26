"""Buoc 5 cua Pipeline Video -> 3D (gan ket voi data/samples/<mau> va outputs/<mau>)."""
from __future__ import annotations

import json
import shutil
from typing import Any

import numpy as np
import open3d as o3d

from nckh_2.core import paths
from nckh_2.core.types import SceneContext, StageResult
from nckh_2.pipeline.refine.engine import refine_obj_file


def run(sample_name: str, cfg: dict[str, Any]) -> None:
    """Chay Buoc 5 tinh chinh, remesh & kiem dinh 6 mat cho mau `sample_name`."""
    r_cfg = cfg.get("mesh_refine", {})
    out_dir = paths.output_dir(sample_name)
    main_obj = out_dir / f"{sample_name}.obj"
    raw_obj = out_dir / f"{sample_name}_raw.obj"
    refined_obj = out_dir / f"{sample_name}_refined.obj"
    audit_dir = out_dir / "refine_audit"

    if not main_obj.exists() and not raw_obj.exists():
        raise FileNotFoundError(f"Khong tim thay {main_obj}, vui long chay Buoc 4 truoc.")

    # Sao luu ban goc 1 lan duy nhat va khong bao gio ghi de raw_obj
    if not raw_obj.exists():
        shutil.copyfile(str(main_obj), str(raw_obj))
        print(f"  da sao luu ban goc (bat bien) -> {raw_obj.name}")

    # Nap point_cloud_clean.ply (va confidence neu co) lam moc tham chieu cho kiem dinh 6 mat
    ref_points, ref_confidence = None, None
    clean_ply = out_dir / "point_cloud_clean.ply"
    conf_npy = out_dir / "point_confidence.npy"
    if clean_ply.exists():
        try:
            pc_ref = o3d.io.read_point_cloud(str(clean_ply))
            pts = np.asarray(pc_ref.points, dtype=np.float64)
            if len(pts) > 0:
                ref_points = pts
                if conf_npy.exists():
                    c_arr = np.load(str(conf_npy))
                    if len(c_arr) == len(pts):
                        ref_confidence = c_arr.astype(np.float64)
        except Exception:
            pass

    print(f"[Buoc 5] Tinh chinh, remesh & kiem dinh 6 mat: {raw_obj.name}...")
    report = refine_obj_file(
        raw_obj,
        refined_obj,
        r_cfg=r_cfg,
        ref_points=ref_points,
        ref_confidence=ref_confidence,
        audit_dir=audit_dir,
    )
    shutil.copyfile(str(refined_obj), str(main_obj))
    shutil.copyfile(str(refined_obj), str(out_dir / "mesh_refined.obj"))

    # Xuat file JSON tong hop ket qua Buoc 5 ngay tai outputs/<sample_name>/step5_result.json
    step5_json = out_dir / "step5_result.json"
    step5_payload = {
        "sample_name": sample_name,
        "input_obj": str(raw_obj),
        "refined_obj": str(refined_obj),
        "main_obj": str(main_obj),
        "refined_size_kb": round(refined_obj.stat().st_size / 1024.0, 2),
        **report.to_dict(),
        "audit_stages": [r.to_dict() for r in report.audit_reports],
    }
    step5_json.write_text(json.dumps(step5_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Buoc 5] Hoan tat trong {report.elapsed_sec:.2f}s!")
    print(f"  -> {refined_obj.name} ({refined_obj.stat().st_size / 1024:.1f} KB)")
    print(f"  -> {main_obj.name} (da dong bo ban tinh chinh)")
    print(f"  -> {step5_json.name} (ket qua JSON Buoc 5)")


class MeshRefineStage:
    stage_id = 5
    title = "Tinh chinh, remesh & kiem dinh 6 mat"

    def execute(self, ctx: SceneContext) -> StageResult:
        run(ctx.sample_name, ctx.config)
        ctx.artifacts["mesh_refined"] = ctx.output_dir / f"{ctx.sample_name}_refined.obj"
        ctx.artifacts["mesh"] = ctx.output_dir / f"{ctx.sample_name}.obj"
        ctx.artifacts["refine_audit"] = ctx.output_dir / "refine_audit"
        return StageResult(stage_id=self.stage_id, stage_name=self.title, success=True, artifacts=ctx.artifacts)
