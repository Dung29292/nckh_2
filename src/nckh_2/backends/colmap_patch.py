"""Sua ten tham so COLMAP cu trong nerfstudio de tuong thich COLMAP 4.2.0."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def patch_nerfstudio_colmap_utils() -> int:
    """Sua SiftExtraction/SiftMatching -> FeatureExtraction/FeatureMatching trong nerfstudio."""
    spec = importlib.util.find_spec("nerfstudio.process_data.colmap_utils")
    if spec is None or spec.origin is None:
        print("nerfstudio khong duoc cai dat trong moi truong nay. Bo qua patch.")
        return 0

    path = Path(spec.origin)
    text = path.read_text(encoding="utf-8")

    replacements = {
        "SiftExtraction.use_gpu": "FeatureExtraction.use_gpu",
        "SiftMatching.use_gpu": "FeatureMatching.use_gpu",
    }
    changed = 0
    for old, new in replacements.items():
        count = text.count(old)
        if count:
            text = text.replace(old, new)
            changed += count
            print(f"  {old} -> {new} ({count} cho)")

    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"Da sua {path} ({changed} cho tong cong)")
    else:
        print(f"File {path} da duoc patch hoac khong can sua.")
    return changed


if __name__ == "__main__":
    patch_nerfstudio_colmap_utils()
