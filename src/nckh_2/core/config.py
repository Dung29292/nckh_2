"""Unified YAML configuration loader for configs/default.yaml and configs/floorplan.yml.

Strictly enforces the Zero Hardcoded Constants rule (Section 1.2 & Section 6.1 of nckh.pdf):
all geometric thresholds, RANSAC tolerances, slice/grid resolutions, morphological kernel sizes,
and smoothing / elastic-energy hyperparameters are loaded from centralized YAML files.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
import yaml

from nckh_2.core.paths import CONFIG_FILE, FLOORPLAN_CONFIG_FILE


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two configuration dictionaries without clobbering sibling keys."""
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def list_profiles(config_path: str | Path | None = None) -> list[str]:
    """List available profile names under `profiles:` in configs/default.yaml."""
    target_path = Path(config_path) if config_path is not None else CONFIG_FILE
    if not target_path.exists():
        return []
    with target_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    profiles = cfg.get("profiles")
    if isinstance(profiles, dict):
        return sorted(str(k) for k in profiles.keys())
    return []


def load_floorplan_config(floorplan_path: str | Path | None = None) -> dict[str, Any]:
    """Load architectural Scan-to-CAD / Scan-to-BIM parameters from configs/floorplan.yml."""
    target = Path(floorplan_path) if floorplan_path is not None else FLOORPLAN_CONFIG_FILE
    if not target.exists():
        raise FileNotFoundError(f"Khong tim thay file cau hinh floorplan: {target}")
    with target.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    return cfg


def load_config(
    config_path: str | Path | None = None,
    profile: str | None = None,
    floorplan_path: str | Path | None = None,
    include_floorplan: bool = False,
) -> dict[str, Any]:
    """Load configs/default.yaml, apply optional profile, and optionally attach `floorplan` config."""
    target_path = Path(config_path) if config_path is not None else CONFIG_FILE
    if not target_path.exists():
        raise FileNotFoundError(f"Khong tim thay file cau hinh: {target_path}")

    with target_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    if profile:
        profiles = cfg.get("profiles") if isinstance(cfg.get("profiles"), dict) else {}
        if profile in profiles and isinstance(profiles[profile], dict):
            cfg = deep_merge(cfg, profiles[profile])
        else:
            prof_path = Path(profile)
            if prof_path.exists():
                with prof_path.open("r", encoding="utf-8") as f:
                    prof_cfg: dict[str, Any] = yaml.safe_load(f) or {}
                cfg = deep_merge(cfg, prof_cfg)
            else:
                avail = ", ".join(sorted(profiles.keys())) or "(khong co)"
                raise FileNotFoundError(f"Khong tim thay profile '{profile}' trong default.yaml (co san: {avail})")

    if include_floorplan or floorplan_path is not None:
        cfg["floorplan"] = load_floorplan_config(floorplan_path)

    return cfg
