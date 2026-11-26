"""Dieu phoi va thuc thi cac buoc trong pipeline tai tao 3D."""
from __future__ import annotations

from typing import Any, Callable

STEP_TITLES = {
    1: "Loc frame mo",
    2: "Tach phong (Grounded-SAM 2)",
    3: "COLMAP + 3DGS",
    4: "Dung mesh khoi tao (Poisson)",
    5: "Tinh chinh & lam min mesh",
}

MODES = {
    "orbit": "Vat dung yen, camera di quanh vat",
    "turntable": "Camera dung yen, vat xoay (ban xoay / xoay tay)",
}

STEPS_NEED_SOURCE = {1, 2}
STEPS_NEED_MODE = {2, 3}


def run_step(step_idx: int, name: str, cfg: dict[str, Any]) -> None:
    """Lazy import tung buoc khi thuc su chay, tranh crash khi thieu thu vien o buoc khac."""
    try:
        if step_idx == 1:
            from nckh_2.pipeline import blur_filter
            return blur_filter.run(name, cfg)
        elif step_idx == 2:
            from nckh_2.pipeline import masking
            return masking.run(name, cfg)
        elif step_idx == 3:
            from nckh_2.pipeline import reconstruct
            return reconstruct.run(name, cfg)
        elif step_idx == 4:
            from nckh_2.pipeline import mesh_export
            return mesh_export.run(name, cfg)
        elif step_idx == 5:
            from nckh_2.pipeline import mesh_refine
            return mesh_refine.run(name, cfg)
    except ImportError as exc:
        print(f"\n[Loi] Khong the chay Buoc {step_idx} ({STEP_TITLES.get(step_idx)}) do thieu thu vien:")
        print(f"  -> {exc}")
        print(f"  Goi y: Hay kiem tra moi truong hoac cai dat thu vien phu hop cho rieng buoc nay.\n")
        return None
    raise ValueError(f"Khong ton tai buoc {step_idx}")


STEPS: dict[int, tuple[str, Callable[[str, dict[str, Any]], None]]] = {
    1: (STEP_TITLES[1], lambda name, cfg: run_step(1, name, cfg)),
    2: (STEP_TITLES[2], lambda name, cfg: run_step(2, name, cfg)),
    3: (STEP_TITLES[3], lambda name, cfg: run_step(3, name, cfg)),
    4: (STEP_TITLES[4], lambda name, cfg: run_step(4, name, cfg)),
    5: (STEP_TITLES[5], lambda name, cfg: run_step(5, name, cfg)),
}


def run_pipeline(name: str, cfg: dict[str, Any], start: int = 1, until: int = 1) -> None:
    """Chay cac buoc tu start den until cho mau `name`."""
    for step in range(start, until + 1):
        title, fn = STEPS[step]
        print(f"\n--- Buoc {step}: {title} ---")
        try:
            fn(name, cfg)
        except NotImplementedError as e:
            print(e)
            break
