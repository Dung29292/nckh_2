"""Cong CLI duy nhat cua du an nckh_2 (Video->3D Pipeline | Single .obj/.ply | Batch).

Cac kich ban su dung chinh:
  poetry run nckh                                                          # Menu tuong tac
  poetry run nckh --source sample --name mouse_02 --start 1 --until 5      # Pipeline Video -> 3D du 5 buoc
  poetry run nckh --source sample --name mouse_02 --start 5 --until 5 --profile curved_object
  poetry run nckh --input model.obj --output out_dir --reference ref.obj   # Tinh chinh 1 file 3D roi
  poetry run nckh --input model.obj --output out_dir --audit-only          # Chi kiem tra lech truc & 6 mat
  poetry run nckh --batch-dir models_dir --output batch_out --pattern "**/*.obj"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from nckh_2.capture.recorder import record_video
from nckh_2.capture.sample_store import choose_sample, list_samples
from nckh_2.core import paths
from nckh_2.core.config import list_profiles, load_config
from nckh_2.core.types import SampleMetadata
from nckh_2.pipeline.runner import (
    MODES,
    STEPS,
    STEPS_NEED_MODE,
    STEPS_NEED_SOURCE,
    run_pipeline,
)


def _menu_source(cfg: dict[str, Any]) -> str:
    print("=== nckh_2 - 3D Spatial Reconstruction ===")
    print("  [1] Quay video truc tiep (Webcam may tinh -> mau moi)")
    print("  [2] Quay bang Camera dien thoai (Quet QR / Wi-Fi LAN)")
    print("  [3] Dung mau co san")
    while True:
        c = input("Chon 1, 2 hoac 3: ").strip()
        if c == "1":
            return "record"
        if c == "2":
            return "mobile"
        if c == "3":
            return "sample"


def _load_meta(name: str) -> SampleMetadata:
    return SampleMetadata.load(paths.sample_dir(name))


def _save_meta(name: str, meta: SampleMetadata) -> None:
    meta.save(paths.sample_dir(name))


def _resolve_sample(args: argparse.Namespace, cfg: dict[str, Any], need_source: bool) -> tuple[str, str | None]:
    if not need_source:
        if args.name:
            return "sample", args.name
        print("Chon mau can xu ly:")
        name = choose_sample(cfg["video_file"])
        return "sample", name

    source = args.source or _menu_source(cfg)
    if source == "mobile":
        cfg["record"]["camera"] = "mobile"
        name = record_video(args.name, cfg)
    elif source == "record":
        name = record_video(args.name, cfg)
    else:
        name = (
            args.name
            if args.name in list_samples(cfg["video_file"])
            else choose_sample(cfg["video_file"])
        )
    return source, name


def _resolve_mode(name: str, cfg: dict[str, Any], cli_mode: str | None, need_mode: bool) -> str:
    meta = _load_meta(name)
    saved = meta.capture_mode if meta.capture_mode else None
    default = saved or cfg["reconstruction"].get("capture_mode", "orbit")

    if not need_mode:
        return default

    if cli_mode:
        mode = cli_mode
    elif sys.stdin.isatty():
        keys = list(MODES)
        print("\nChe do quay cua mau nay:")
        for i, k in enumerate(keys, 1):
            tag = "  (mac dinh)" if k == default else ""
            print(f"  [{i}] {MODES[k]}{tag}")
        mode = default
        while True:
            c = input(f"Chon 1-{len(keys)} (Enter = giu '{default}'): ").strip()
            if not c:
                break
            if c.isdigit() and 1 <= int(c) <= len(keys):
                mode = keys[int(c) - 1]
                break
    else:
        mode = default

    if saved != mode:
        meta.capture_mode = mode
        _save_meta(name, meta)
    return mode


def _apply_refine_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Ap dung cac co ghi de nhanh tu CLI vao nhanh mesh_refine trong cfg."""
    mr = cfg.setdefault("mesh_refine", {})
    if args.align_mode is not None:
        mr.setdefault("alignment", {})["align_mode"] = args.align_mode
    if args.smooth_mode is not None:
        mr.setdefault("smoothing", {})["mode"] = args.smooth_mode
    if args.no_remesh:
        mr.setdefault("remesh", {})["enable"] = False
    if args.target_edge_frac is not None:
        mr.setdefault("remesh", {})["target_edge_frac"] = args.target_edge_frac
    if args.max_drift_frac is not None:
        mr.setdefault("smoothing", {})["max_drift_frac"] = args.max_drift_frac
    if args.save_intermediates:
        mr.setdefault("audit", {})["save_intermediates"] = True


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    avail_profiles = list_profiles()
    profile_help = f"ho so cau hinh trong muc profiles cua configs/default.yaml (co san: {', '.join(avail_profiles)})"

    ap = argparse.ArgumentParser(description="nckh_2 - 3D Spatial Reconstruction & Mesh Refinement CLI")
    # --- Nhom 1: Cau hinh chung & ho so rieng ---
    ap.add_argument("--config", type=Path, default=None, help="duong dan YAML cau hinh tuy chon")
    ap.add_argument("--floorplan-config", type=Path, default=None, help="duong dan configs/floorplan.yml tuy chon")
    ap.add_argument("--profile", default=None, help=profile_help)
    ap.add_argument(
        "--route-mode",
        choices=["auto", "freeform_object", "architectural_planar", "room_layout", "topdown_object"],
        default=None,
        help="dinh tuyen giai thuat hinh hoc theo loai be mat / khong gian (Section 6.2 nckh.pdf)",
    )

    # --- Nhom 2: Pipeline Video -> 3D (Buoc 1..5) ---
    ap.add_argument("--source", choices=["record", "mobile", "sample"],
                    help="nguon video: record = webcam, mobile = camera dien thoai (QR), sample = mau co san")
    ap.add_argument("--name", help="ten mau trong data/samples/")
    ap.add_argument("--camera", help="chi so camera, 'mobile', hoac URL (ghi de config)")
    ap.add_argument("--host-ip", help="ghi de dia chi IP LAN phat QR cho camera dien thoai")
    ap.add_argument("--prompt", help="tu khoa nhan dien vat the cho Grounded-SAM 2")
    ap.add_argument("--mode", choices=list(MODES), help="che do quay: orbit | turntable")
    ap.add_argument("--start", type=int, default=1, choices=list(STEPS), help="bat dau tu buoc nay (mac dinh: 1)")
    ap.add_argument("--until", type=int, default=None, choices=list(STEPS), help="chay den buoc nay (mac dinh: bang --start)")
    ap.add_argument("--only-record", action="store_true", help="chi quay / chon mau video roi dung")

    # --- Nhom 3: Tinh chinh 1 file 3D roi hoac chay hang loat (Batch) ---
    ap.add_argument("--input", type=Path, default=None, help="duong dan 1 file .obj/.ply roi can tinh chinh")
    ap.add_argument("--batch-dir", type=Path, default=None, help="thu muc chua nhieu file .obj/.ply can chay hang loat")
    ap.add_argument("--output", type=Path, default=None, help="file hoac thu muc dau ra cho --input / --batch-dir")
    ap.add_argument("--pattern", default="*.obj", help="glob pattern cho --batch-dir (mac dinh: '*.obj')")
    ap.add_argument("--reference", type=Path, default=None, help="file tham chieu .obj/.ply cho kiem dinh 6 mat")
    ap.add_argument("--ref-dir", type=Path, default=None, help="thu muc song song chua file tham chieu cho --batch-dir")

    # --- Nhom 4: Co ghi de nhanh cho Buoc 5 (Mesh Refine & Audit) ---
    ap.add_argument("--align-mode", choices=["preserve", "pca_canonical", "reference_icp"], default=None)
    ap.add_argument("--smooth-mode", choices=["hybrid", "taubin", "bilateral_normal", "none"], default=None)
    ap.add_argument("--no-remesh", action="store_true", help="bo qua PyMeshLab isotropic remeshing")
    ap.add_argument("--target-edge-frac", type=float, default=None, help="do dai canh remesh / bbox_diag")
    ap.add_argument("--max-drift-frac", type=float, default=None, help="gioi han do troi P95 / bbox_diag")
    ap.add_argument("--audit-only", action="store_true", help="chi chan doan lech truc & 6 mat, khong sua mesh")
    ap.add_argument("--save-intermediates", action="store_true", help="luu them file .obj trung gian tung pha")
    ap.add_argument("--fail-fast", action="store_true", help="dung ngay khi gap loi trong --batch-dir")
    args = ap.parse_args()

    paths.ensure_base_dirs()
    cfg = load_config(
        config_path=args.config,
        profile=args.profile,
        floorplan_path=args.floorplan_config,
        include_floorplan=(args.route_mode is not None or args.floorplan_config is not None),
    )
    if args.route_mode is not None:
        cfg.setdefault("surface_router", {})["mode"] = args.route_mode
    _apply_refine_cli_overrides(cfg, args)

    # -------------------------------------------------------------------------
    # LUONG 0: Dinh tuyen bai toan & Tai tao 3D / CAD / BIM tu Point Cloud (.ply)
    # -------------------------------------------------------------------------
    if args.input is not None and args.route_mode is not None and args.input.suffix.lower() == ".ply":
        from nckh_2.backends.geometry import load_gaussian_ply
        from nckh_2.pipeline.surface_router import route_and_reconstruct

        out_target = args.output or (args.input.parent / f"{args.input.stem}_routed")
        pc_in = load_gaussian_ply(args.input)
        res = route_and_reconstruct(pc_in, cfg, out_dir=out_target, stem=args.input.stem)
        print(f"\n[Router] Route='{res['routing']['route']}' hoan tat -> {out_target}")
        return

    # -------------------------------------------------------------------------
    # LUONG A: Chay hang loat (Batch) tren thu muc tai san 3D
    # -------------------------------------------------------------------------
    if args.batch_dir is not None:
        from nckh_2.pipeline.refine.batch import run_batch_refine

        out_dir = args.output or (args.batch_dir / "refined_batch")
        run_batch_refine(
            args.batch_dir,
            out_dir,
            pattern=args.pattern,
            cfg=cfg,
            ref_dir=args.ref_dir,
            audit_only=args.audit_only,
            fail_fast=args.fail_fast,
        )
        return

    # -------------------------------------------------------------------------
    # LUONG B: Tinh chinh / Kiem dinh 1 file .obj / .ply roi bat ky
    # -------------------------------------------------------------------------
    if args.input is not None:
        from nckh_2.pipeline.refine.engine import refine_obj_file

        out_target = args.output or (args.input.parent / f"{args.input.stem}_refine_audit")
        rep = refine_obj_file(
            args.input,
            out_target,
            r_cfg=cfg.get("mesh_refine", {}),
            reference_path=args.reference,
            audit_only=args.audit_only,
        )
        print(f"\n[Refine] Hoan tat trong {rep.elapsed_sec:.2f}s -> {out_target}")
        return

    # -------------------------------------------------------------------------
    # LUONG C: Pipeline Video -> 3D (Buoc 1..5)
    # -------------------------------------------------------------------------
    if args.until is None:
        args.until = args.start
    if args.start > args.until:
        ap.error(f"--start ({args.start}) phai <= --until ({args.until})")

    if args.camera is not None:
        cfg["record"]["camera"] = args.camera
    if args.host_ip is not None:
        cfg["record"]["mobile_host_ip"] = args.host_ip

    steps_to_run = set(range(args.start, args.until + 1))
    need_source = bool(steps_to_run & STEPS_NEED_SOURCE)
    need_mode = bool(steps_to_run & STEPS_NEED_MODE)

    source, name = _resolve_sample(args, cfg, need_source)
    if not name:
        print("Khong co mau de xu ly. Thoat.")
        return

    if args.only_record:
        print(f"\nDa luu mau: {name}. Hoan tat theo yeu cau (--only-record).")
        return

    meta = _load_meta(name)
    if args.prompt:
        cfg["masking"]["prompt"] = args.prompt
        meta.prompt = args.prompt
        _save_meta(name, meta)
    elif meta.prompt:
        cfg["masking"]["prompt"] = meta.prompt
    else:
        prefix = name.split("_")[0].lower()
        if prefix and prefix not in ("sample", "video") and cfg["masking"]["prompt"] == "box":
            cfg["masking"]["prompt"] = prefix

    mode = _resolve_mode(name, cfg, args.mode, need_mode)
    cfg["reconstruction"]["capture_mode"] = mode

    prof_tag = f"  |  Profile: {args.profile}" if args.profile else ""
    print(f"\n>>> Mau: {name}  |  Buoc: {args.start}-{args.until}  |  Mode: {mode}  |  Prompt: '{cfg['masking']['prompt']}'{prof_tag}")
    run_pipeline(name, cfg, start=args.start, until=args.until)


if __name__ == "__main__":
    main()
