"""Chay 1 lan: sua ten tham so COLMAP cu (SiftExtraction/SiftMatching) ma nerfstudio 1.1.5
con hardcode, thanh ten moi (FeatureExtraction/FeatureMatching) ma COLMAP 4.2.0 dung."""
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("nerfstudio.process_data.colmap_utils")
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

path.write_text(text, encoding="utf-8")
print(f"Da sua {path} ({changed} cho tong cong)")