"""Diem vao cua pipeline.

  poetry run nckh                                  # menu tuong tac: quay moi / dung mau co san
  poetry run nckh --source record --name box_01    # quay video moi, luu thanh mau box_01
  poetry run nckh --source sample --name box_01    # dung mau co san
  poetry run nckh --mode turntable                 # bo qua cau hoi che do quay
  poetry run nckh --start 4 --until 4              # chi chay buoc 4
"""
import argparse
import json
import sys

from . import paths
from .config import load_config
from .input_source import choose_sample, list_samples, record_video
from . import step1_blur_filter, step2_masking, step3_reconstruct, step4_mesh_export

STEPS = {
    1: ("Loc frame mo",              step1_blur_filter.run),
    2: ("Tach phong (Grounded-SAM 2)", step2_masking.run),
    3: ("COLMAP + 3DGS",             step3_reconstruct.run),
    4: ("Lam sach & xuat .obj",      step4_mesh_export.run),
}

# Buoc nao can biet nguon video / camera
_STEPS_NEED_SOURCE = {1, 2}
# Buoc nao can biet che do quay (orbit/turntable)
_STEPS_NEED_MODE   = {2, 3}

MODES = {
    "orbit":      "Vat dung yen, camera di quanh vat",
    "turntable":  "Camera dung yen, vat xoay (ban xoay / xoay tay)",
}


# ---------------------------------------------------------------------------
# TIEN ICH
# ---------------------------------------------------------------------------

def _menu_source(cfg: dict) -> str:
    print("=== nckh_2 - Pha 1 ===")
    print("  [1] Quay video truc tiep (luu thanh mau moi)")
    print("  [2] Dung mau co san")
    while True:
        c = input("Chon 1 hoac 2: ").strip()
        if c == "1":
            return "record"
        if c == "2":
            return "sample"


def _load_meta(name: str) -> dict:
    p = paths.sample_dir(name) / "meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save_meta(name: str, meta: dict) -> None:
    p = paths.sample_dir(name) / "meta.json"
    p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_sample(args: argparse.Namespace, cfg: dict, need_source: bool) -> tuple[str, str]:
    """Tra ve (source, name). Chi hoi menu neu thuc su can."""
    if not need_source:
        # Buoc chi can ten mau, khong can nguon video
        if args.name:
            return "sample", args.name
        # Khong co --name: hoi chon mau co san (khong hoi record/sample)
        print("Chon mau can xu ly:")
        name = choose_sample(cfg["video_file"])
        return "sample", name

    # Buoc can nguon (buoc 1, 2): hoi record hay sample
    source = args.source or _menu_source(cfg)
    if source == "record":
        name = record_video(args.name, cfg)
    else:
        name = (
            args.name
            if args.name in list_samples(cfg["video_file"])
            else choose_sample(cfg["video_file"])
        )
    return source, name


def _resolve_mode(name: str, cfg: dict, cli_mode: str | None, need_mode: bool) -> str:
    """Tra ve capture_mode. Chi hoi neu buoc can biet mode."""
    meta    = _load_meta(name)
    saved   = meta.get("capture_mode")
    default = saved or cfg["reconstruction"].get("capture_mode", "orbit")

    if not need_mode:
        # Buoc 1, 4: khong can hoi, lay tu meta/config
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
        meta["capture_mode"] = mode
        _save_meta(name, meta)
    return mode


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="nckh_2 pipeline")
    ap.add_argument("--source",  choices=["record", "sample"],
                    help="nguon du lieu: record = quay moi, sample = mau co san")
    ap.add_argument("--name",    help="ten mau")
    ap.add_argument("--camera",  help="chi so camera hoac URL (ghi de config)")
    ap.add_argument("--mode",    choices=list(MODES),
                    help="che do quay: orbit | turntable")
    ap.add_argument("--start",   type=int, default=1, choices=list(STEPS),
                    help="bat dau tu buoc nay (mac dinh: 1)")
    ap.add_argument("--until",   type=int, default=None, choices=list(STEPS),
                    help="chay den buoc nay (mac dinh: bang --start)")
    args = ap.parse_args()

    # --until mac dinh bang --start (khong truyen = chi chay 1 buoc)
    if args.until is None:
        args.until = args.start

    if args.start > args.until:
        ap.error(f"--start ({args.start}) phai <= --until ({args.until})")

    paths.ensure_base_dirs()
    cfg = load_config()

    if args.camera is not None:
        cfg["record"]["camera"] = args.camera

    # Xac dinh xem cac buoc se chay co can source/mode khong
    steps_to_run  = set(range(args.start, args.until + 1))
    need_source   = bool(steps_to_run & _STEPS_NEED_SOURCE)
    need_mode     = bool(steps_to_run & _STEPS_NEED_MODE)

    # Lay ten mau (va nguon neu can)
    source, name = _resolve_sample(args, cfg, need_source)
    if not name:
        print("Khong co mau de xu ly. Thoat.")
        return

    # Lay che do quay (chi hoi khi buoc 2 hoac 3 chay)
    mode = _resolve_mode(name, cfg, args.mode, need_mode)
    cfg["reconstruction"]["capture_mode"] = mode

    print(f"\n>>> Mau: {name}  |  Buoc: {args.start}-{args.until}  |  Mode: {mode}")

    for step in range(args.start, args.until + 1):
        title, fn = STEPS[step]
        print(f"\n--- Buoc {step}: {title} ---")
        try:
            fn(name, cfg)
        except NotImplementedError as e:
            print(e)
            break


if __name__ == "__main__":
    main()