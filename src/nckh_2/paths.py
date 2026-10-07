"""Quy uoc duong dan cua du an + ep cac thu vien tai model ve thu muc models/."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "configs" / "default.yaml"
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

# Model tai qua Hugging Face / torch hub se nam trong models/ thay vi C:\Users\...\.cache
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch_cache"))


def sample_dir(name: str) -> Path:
    return SAMPLES_DIR / name


def output_dir(name: str) -> Path:
    return OUTPUTS_DIR / name


def ensure_base_dirs() -> None:
    for d in (SAMPLES_DIR, OUTPUTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)