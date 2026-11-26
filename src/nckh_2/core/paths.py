"""Quy uoc duong dan cua du an + ep cac thu vien tai model ve thu muc models/."""
from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    """Tim thu muc goc cua du an dua tren su ton tai cua pyproject.toml."""
    cur = Path(__file__).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback ve thu muc cach 3 cap (src/nckh_2/core/paths.py -> root)
    return cur.parents[3]


ROOT = _find_project_root()
CONFIG_FILE = ROOT / "configs" / "default.yaml"
FLOORPLAN_CONFIG_FILE = ROOT / "configs" / "floorplan.yml"
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"

# ffmpeg dat trong models/ffmpeg/bin, khong cai he thong -> tu them vao PATH cho session nay
FFMPEG_BIN = MODELS_DIR / "ffmpeg" / "bin"
if FFMPEG_BIN.exists():
    os.environ["PATH"] = str(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")

# colmap.exe dat thang trong models/colmap/, khong cai he thong -> tu them vao PATH
COLMAP_BIN = MODELS_DIR / "colmap"
if COLMAP_BIN.exists():
    os.environ["PATH"] = str(COLMAP_BIN) + os.pathsep + os.environ.get("PATH", "")

# Model tai qua Hugging Face / torch hub se nam trong models/ thay vi C:\\Users\\...\\.cache
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch_cache"))


def resolve_project_path(p: str | Path | None) -> Path | None:
    """Chuyen doi duong dan tuong doi ve tuyet doi dua tren thu muc ROOT cua du an."""
    if p is None:
        return None
    path_obj = Path(p)
    if path_obj.is_absolute():
        return path_obj
    return ROOT / path_obj


def sample_dir(name: str) -> Path:
    return SAMPLES_DIR / name


def output_dir(name: str) -> Path:
    return OUTPUTS_DIR / name


def ensure_base_dirs() -> None:
    for d in (SAMPLES_DIR, OUTPUTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
