"""Quan ly va lua chon tap mau du lieu (samples)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from nckh_2.core import paths


def list_samples(video_file: str = "video.mp4") -> list[str]:
    """Liet ke danh sach ten mau co chua file video trong data/samples."""
    if not paths.SAMPLES_DIR.exists():
        return []
    return sorted(
        d.name for d in paths.SAMPLES_DIR.iterdir() if d.is_dir() and (d / video_file).exists()
    )


def choose_sample(video_file: str = "video.mp4") -> str | None:
    """Hien danh sach mau va cho nguoi dung chon bang so tu terminal."""
    samples = list_samples(video_file)
    if not samples:
        print(f"Chua co mau nao. Hay dat video vao data/samples/<ten_mau>/{video_file} hoac chon che do quay.")
        return None
    print("\nCac mau co san:")
    for i, name in enumerate(samples, 1):
        print(f"  [{i}] {name}")
    while True:
        s = input("Chon mau (so, Enter de huy): ").strip()
        if not s:
            return None
        if s.isdigit() and 1 <= int(s) <= len(samples):
            return samples[int(s) - 1]
        print("Lua chon khong hop le.")


def unique_name(name: str | None) -> str:
    """Tao ten mau duy nhat khong bi trung lap trong data/samples/."""
    if not name:
        return "box_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    base, n = name, 1
    while paths.sample_dir(name).exists():
        n += 1
        name = f"{base}_{n}"
    return name
