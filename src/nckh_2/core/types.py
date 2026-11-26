"""Dinh nghia cac kieu du lieu, dataclass chia se giua cac module va pipeline stages."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Literal


@dataclasses.dataclass
class SampleMetadata:
    """Thong tin metadata luu tru tai data/samples/<sample_name>/meta.json."""

    source: str = "sample"
    camera: str = "0"
    width: int = 1280
    height: int = 720
    fps: float = 60.0
    created: str = ""
    prompt: str = ""
    capture_mode: str = "orbit"
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def load(cls, sample_dir: Path) -> "SampleMetadata":
        meta_file = sample_dir / "meta.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                known_fields = {f.name for f in dataclasses.fields(cls) if f.name != "extra"}
                known_data = {k: v for k, v in data.items() if k in known_fields}
                extra_data = {k: v for k, v in data.items() if k not in known_fields}
                return cls(**known_data, extra=extra_data)
            except Exception:
                pass
        return cls()

    def save(self, sample_dir: Path) -> None:
        sample_dir.mkdir(parents=True, exist_ok=True)
        meta_file = sample_dir / "meta.json"
        data = dataclasses.asdict(self)
        extra = data.pop("extra", {})
        data.update(extra)
        meta_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclasses.dataclass
class SceneContext:
    """Ngu canh khong gian / vat the xuyen suot toan bo pipeline tai tao 3D."""

    sample_name: str
    sample_dir: Path
    output_dir: Path
    config: dict[str, Any]
    metadata: SampleMetadata = dataclasses.field(default_factory=SampleMetadata)
    mode: Literal["single_object", "multi_object", "full_scene"] = "single_object"
    artifacts: dict[str, Path] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class StageResult:
    """Ket qua thuc thi cua mot pipeline stage."""

    stage_id: int
    stage_name: str
    success: bool
    message: str = ""
    artifacts: dict[str, Path] = dataclasses.field(default_factory=dict)
