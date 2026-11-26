"""Trinh chay hang loat (Batch Runner) goi chung loi `pipeline.refine.engine`."""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import time
from typing import Any

from nckh_2.core.config import load_config
from nckh_2.pipeline.refine.engine import refine_obj_file


def _find_reference(asset_path: Path, ref_dir: Path | None) -> Path | None:
    if ref_dir is not None and ref_dir.exists():
        for ext in (".obj", ".ply", ".stl"):
            cand = ref_dir / f"{asset_path.stem}{ext}"
            if cand.exists():
                return cand
    sibling_pc = asset_path.parent / "point_cloud_clean.ply"
    if sibling_pc.exists() and sibling_pc.resolve() != asset_path.resolve():
        return sibling_pc
    return None


def run_batch_refine(
    input_dir: Path | str,
    out_dir: Path | str,
    pattern: str = "*.obj",
    cfg: dict[str, Any] | None = None,
    profile: str | None = None,
    ref_dir: Path | str | None = None,
    audit_only: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Chay tinh chinh hang loat tren toan bo tai san 3D khop `pattern` trong `input_dir`."""
    t0 = time.perf_counter()
    in_root = Path(input_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ref_root = Path(ref_dir) if ref_dir is not None else None

    if not in_root.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc dau vao: {in_root}")

    full_cfg = cfg if cfg is not None else load_config(profile=profile)
    r_cfg = full_cfg.get("mesh_refine", full_cfg)

    assets = sorted(p for p in in_root.glob(pattern) if p.is_file())
    print(f"[Batch] Tim thay {len(assets)} tai san khop '{pattern}' trong {in_root}")

    rows: list[dict[str, Any]] = []
    n_ok, n_err = 0, 0

    for idx, asset in enumerate(assets, 1):
        rel_name = asset.stem if asset.parent == in_root else f"{asset.parent.name}_{asset.stem}"
        asset_out = out_root / rel_name
        ref_path = _find_reference(asset, ref_root)
        print(f"\n[Batch {idx}/{len(assets)}] Dang xu ly: {asset.name} -> {asset_out}")

        try:
            rep = refine_obj_file(
                asset,
                asset_out,
                r_cfg=r_cfg,
                reference_path=ref_path,
                audit_only=audit_only,
                audit_dir=asset_out,
            )
            a_in = rep.audit_reports[0] if rep.audit_reports else None
            a_out = rep.audit_reports[-1] if rep.audit_reports else None
            rows.append({
                "asset": asset.name,
                "status": "ok",
                "verts_before": rep.before.num_vertices,
                "verts_after": rep.after.num_vertices,
                "tris_before": rep.before.num_triangles,
                "tris_after": rep.after.num_triangles,
                "comps_before": rep.before.num_components,
                "comps_after": rep.after.num_components,
                "skew_before_deg": a_in.max_axis_skew_deg if a_in else 0.0,
                "skew_after_deg": a_out.max_axis_skew_deg if a_out else 0.0,
                "bad_ar_before_pct": a_in.bad_aspect_ratio_pct if a_in else 0.0,
                "bad_ar_after_pct": a_out.bad_aspect_ratio_pct if a_out else 0.0,
                "density_cv_before": a_in.density_cv_6sides if a_in else 0.0,
                "density_cv_after": a_out.density_cv_6sides if a_out else 0.0,
                "elapsed_sec": round(rep.elapsed_sec, 3),
                "error": "",
            })
            n_ok += 1
        except Exception as exc:
            n_err += 1
            print(f"  [Batch ERROR] {asset.name}: {exc}")
            rows.append({
                "asset": asset.name,
                "status": "error",
                "verts_before": 0,
                "verts_after": 0,
                "tris_before": 0,
                "tris_after": 0,
                "comps_before": 0,
                "comps_after": 0,
                "skew_before_deg": 0.0,
                "skew_after_deg": 0.0,
                "bad_ar_before_pct": 0.0,
                "bad_ar_after_pct": 0.0,
                "density_cv_before": 0.0,
                "density_cv_after": 0.0,
                "elapsed_sec": 0.0,
                "error": str(exc),
            })
            if fail_fast:
                raise
        finally:
            gc.collect()

    csv_path = out_root / "batch_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "input_dir": str(in_root),
        "pattern": pattern,
        "total_assets": len(assets),
        "succeeded": n_ok,
        "failed": n_err,
        "total_elapsed_sec": round(time.perf_counter() - t0, 3),
        "csv_path": str(csv_path),
        "results": rows,
    }
    json_path = out_root / "batch_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Batch] Hoan tat: {n_ok}/{len(assets)} thanh cong, {n_err} loi ({summary['total_elapsed_sec']:.2f}s)")
    print(f"  -> {csv_path}")
    print(f"  -> {json_path}")
    return summary
