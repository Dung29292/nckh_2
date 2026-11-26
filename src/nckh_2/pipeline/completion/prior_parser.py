"""Phase 3A: Semantic & structural prior parser from user text prompt + observed cutouts.

Converts natural-language object descriptions (e.g. ``"orange lid black plastic toolbox
with front latch and top recessed handle"``) plus representative real cutouts into a
structured ``StructuralPrior`` JSON specifying:
  - Symmetry priors (bilateral planes, rotational order, planar base)
  - Viewpoint-specific web image search queries (back, side, bottom, 360 product)
  - Viewpoint-specific 2D completion prompts
  - Sharp crease vs smooth surface descriptors
  - Real observed color palette statistics in CIE Lab & human color names
"""
from __future__ import annotations

import base64
import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclasses.dataclass
class StructuralPrior:
    """Machine-readable structural & retrieval contract driving occluded-region completion."""

    prompt: str
    object_category: str
    detected_product_keywords: list[str]
    bilateral_axes: list[int]          # local object axes: 0=mirror X (YZ plane), 1=mirror Y (XZ plane)
    rotational_order: int              # 0=none, >=2=C_n rotational symmetry around local Z
    planar_base: bool
    web_search_queries: dict[str, list[str]]
    view_prompts: dict[str, str]
    sharp_crease_parts: list[str]
    smooth_parts: list[str]
    observed_color_names: list[str]
    observed_lab_stats: dict[str, list[float]]  # {"mean": [L,a,b], "std": [L,a,b]}

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuralPrior":
        return cls(**data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "StructuralPrior":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# OBSERVED COLOR & PALETTE GROUNDING FROM REAL CUTOUTS
# ---------------------------------------------------------------------------

def _classify_hsv_color_name(h: float, s: float, v: float) -> str:
    """Map OpenCV HSV values (H in [0,180], S/V in [0,255]) to simple color keywords."""
    if v < 55:
        return "black"
    if s < 35:
        return "white" if v > 175 else "gray"
    if h < 10 or h >= 170:
        return "red"
    if h < 24:
        return "orange"
    if h < 36:
        return "yellow"
    if h < 82:
        return "green"
    if h < 102:
        return "cyan"
    if h < 132:
        return "blue"
    if h < 155:
        return "purple"
    return "pink"


def extract_observed_palette(cutout_paths: list[Path], max_images: int = 4) -> tuple[list[str], dict[str, list[float]]]:
    """Compute dominant color names and CIE Lab mean/std from foreground pixels in cutouts."""
    if not cutout_paths:
        return [], {"mean": [128.0, 128.0, 128.0], "std": [35.0, 15.0, 15.0]}

    step = max(1, len(cutout_paths) // max_images)
    chosen = cutout_paths[::step][:max_images]

    fg_bgr_samples: list[np.ndarray] = []
    for p in chosen:
        img = cv2.imread(str(p))
        if img is None:
            continue
        # Cutouts have pure black (0,0,0) background; exclude exact zero pixels
        fg = np.any(img > 8, axis=2)
        if np.sum(fg) < 50:
            continue
        pts = img[fg]
        if len(pts) > 6000:
            idx = np.linspace(0, len(pts) - 1, 6000, dtype=int)
            pts = pts[idx]
        fg_bgr_samples.append(pts)

    if not fg_bgr_samples:
        return [], {"mean": [128.0, 128.0, 128.0], "std": [35.0, 15.0, 15.0]}

    all_bgr = np.vstack(fg_bgr_samples).reshape(-1, 1, 3)
    all_lab = cv2.cvtColor(all_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    all_hsv = cv2.cvtColor(all_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)

    lab_stats = {
        "mean": [round(float(x), 2) for x in all_lab.mean(axis=0)],
        "std": [round(float(max(x, 5.0)), 2) for x in all_lab.std(axis=0)],
    }

    # Quantize pixels into color name histogram to extract top 2-3 real object colors
    counts: dict[str, int] = {}
    sample_step = max(1, len(all_hsv) // 2500)
    for h, s, v in all_hsv[::sample_step]:
        cname = _classify_hsv_color_name(h, s, v)
        counts[cname] = counts.get(cname, 0) + 1
    total = max(sum(counts.values()), 1)
    dominant = [
        name for name, cnt in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        if cnt / total >= 0.08
    ][:3]
    return dominant, lab_stats


# ---------------------------------------------------------------------------
# DETERMINISTIC HEURISTIC PARSER + OPTIONAL VLM API ENRICHMENT
# ---------------------------------------------------------------------------

_CYLINDER_KW = ("bottle", "jar", "cup", "mug", "can", "vase", "tumbler", "cylinder", "thermos")
_BOX_KW = ("box", "toolbox", "carton", "crate", "case", "package", "cube", "container", "chest")
_SYMMETRIC_KW = ("mouse", "chair", "shoe", "car", "keyboard", "monitor", "laptop", "helmet", "controller", "headphone")


def _heuristic_prior(
    prompt: str,
    sample_name: str,
    color_names: list[str],
    lab_stats: dict[str, list[float]],
) -> StructuralPrior:
    clean_prompt = prompt.strip() or sample_name.replace("_", " ")
    low = f"{clean_prompt} {sample_name}".lower()

    is_cyl = any(k in low for k in _CYLINDER_KW)
    is_box = (not is_cyl) and any(k in low for k in _BOX_KW)
    is_sym = any(k in low for k in _SYMMETRIC_KW)

    if is_cyl:
        category = "cylinder_vessel"
        bilateral_axes = [0, 1]
        rotational_order = 8
        sharp_parts = ["lid_rim", "neck_collar", "base_edge"]
        smooth_parts = ["cylindrical_body", "shoulder_curve"]
    elif is_box:
        category = "box_container"
        bilateral_axes = [0]   # left-right symmetry (front latch & back hinges differ along Y)
        rotational_order = 0
        sharp_parts = ["lid_overhang", "front_latch", "recessed_handle", "corner_chamfers", "base_rim"]
        smooth_parts = ["lid_bevel", "flat_panels"]
    elif is_sym:
        category = "symmetric_object"
        bilateral_axes = [0]
        rotational_order = 0
        sharp_parts = ["seam_lines", "buttons", "wheel_edges"]
        smooth_parts = ["ergonomic_shell"]
    else:
        category = "general_object"
        bilateral_axes = [0] if ("symmetric" in low or "symmetrical" in low) else []
        rotational_order = 0
        sharp_parts = ["edges", "corners", "panel_seams"]
        smooth_parts = ["curved_surfaces"]

    # Enrich query keywords with real observed colors not already present in prompt
    extra_colors = [c for c in color_names if c not in low and c != "gray"]
    color_prefix = " ".join(extra_colors[:2])
    base_desc = f"{color_prefix} {clean_prompt}".strip() if color_prefix else clean_prompt

    # Extract concise product/search keywords
    tokens = [w for w in re.findall(r"[a-zA-Z0-9\-\+]+", base_desc) if len(w) > 1]
    kw_main = " ".join(tokens[:8]) if tokens else clean_prompt

    web_queries = {
        "general": [f"{kw_main} product photo", f"{kw_main} 360 view"],
        "back":    [f"{kw_main} back view", f"{kw_main} rear view"],
        "side":    [f"{kw_main} side view", f"{kw_main} side profile"],
        "bottom":  [f"{kw_main} bottom view", f"{kw_main} underside"],
    }
    view_prompts = {
        "back":       f"clean studio rear view of {base_desc}, matching original material and proportions, sharp details",
        "back_left":  f"clean studio rear-left three-quarter view of {base_desc}, consistent structure and color",
        "back_right": f"clean studio rear-right three-quarter view of {base_desc}, consistent structure and color",
        "left":       f"clean studio left side profile view of {base_desc}, sharp edges and molded details",
        "right":      f"clean studio right side profile view of {base_desc}, sharp edges and molded details",
        "bottom":     f"clean studio flat underside bottom view of {base_desc}, consistent matte material",
        "top":        f"clean studio top-down view of {base_desc}, crisp geometry",
    }

    return StructuralPrior(
        prompt=clean_prompt,
        object_category=category,
        detected_product_keywords=[kw_main],
        bilateral_axes=bilateral_axes,
        rotational_order=rotational_order,
        planar_base=True,
        web_search_queries=web_queries,
        view_prompts=view_prompts,
        sharp_crease_parts=sharp_parts,
        smooth_parts=smooth_parts,
        observed_color_names=color_names,
        observed_lab_stats=lab_stats,
    )


def _try_gemini_vlm_prior(
    prompt: str,
    cutout_paths: list[Path],
    fallback: StructuralPrior,
    model_name: str = "gemini-2.0-flash",
) -> StructuralPrior:
    """Optionally enrich StructuralPrior with OCR product name & hidden-part inference via Gemini API."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return fallback

    try:
        import httpx
    except ImportError:
        return fallback

    parts: list[dict[str, Any]] = [{
        "text": (
            "You are a 3D Computer Vision researcher. Inspect these real captured views of an object "
            f"and the user prompt: '{prompt}'.\n"
            "Return ONLY valid JSON with keys:\n"
            '  "object_category": short snake_case string,\n'
            '  "detected_product_keywords": list of 1-3 exact brand/model/product search keywords (read any visible label text),\n'
            '  "bilateral_axes": list of integers (0 if left-right symmetric, 1 if front-back symmetric),\n'
            '  "rotational_order": integer (0 if not radially symmetric, 8 if cylindrical),\n'
            '  "planar_base": boolean,\n'
            '  "web_search_queries": {"general": [...], "back": [...], "side": [...], "bottom": [...]},\n'
            '  "view_prompts": {"back": "...", "left": "...", "right": "...", "bottom": "..."},\n'
            '  "sharp_crease_parts": list of fine sharp features to protect from smoothing,\n'
            '  "smooth_parts": list of smooth curved regions.'
        )
    }]

    step = max(1, len(cutout_paths) // 3)
    for p in cutout_paths[::step][:3]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(1.0, 512.0 / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(buf.tobytes()).decode("ascii"),
                }
            })

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    try:
        resp = httpx.post(
            url,
            json={"contents": [{"parts": parts}], "generationConfig": {"responseMimeType": "application/json"}},
            timeout=20.0,
        )
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
        return StructuralPrior(
            prompt=fallback.prompt,
            object_category=str(parsed.get("object_category", fallback.object_category)),
            detected_product_keywords=list(parsed.get("detected_product_keywords", fallback.detected_product_keywords)),
            bilateral_axes=[int(x) for x in parsed.get("bilateral_axes", fallback.bilateral_axes) if int(x) in (0, 1, 2)],
            rotational_order=int(parsed.get("rotational_order", fallback.rotational_order)),
            planar_base=bool(parsed.get("planar_base", fallback.planar_base)),
            web_search_queries={**fallback.web_search_queries, **parsed.get("web_search_queries", {})},
            view_prompts={**fallback.view_prompts, **parsed.get("view_prompts", {})},
            sharp_crease_parts=list(parsed.get("sharp_crease_parts", fallback.sharp_crease_parts)),
            smooth_parts=list(parsed.get("smooth_parts", fallback.smooth_parts)),
            observed_color_names=fallback.observed_color_names,
            observed_lab_stats=fallback.observed_lab_stats,
        )
    except Exception as exc:
        print(f"  [PriorParser] Gemini API fallback sang heuristic ({exc})")
        return fallback


def parse_structural_prior(
    prompt: str,
    sample_name: str,
    sample_dir: Path | None = None,
    backend: str = "auto",
    cache_path: Path | None = None,
) -> StructuralPrior:
    """Build or load a grounded ``StructuralPrior`` for ``sample_name``."""
    cutouts: list[Path] = []
    if sample_dir is not None:
        cdir = sample_dir / "cutouts"
        if cdir.exists():
            cutouts = sorted(cdir.glob("frame_*.jpg"))
        if not cutouts and (sample_dir / "frames_sharp").exists():
            cutouts = sorted((sample_dir / "frames_sharp").glob("frame_*.jpg"))

    color_names, lab_stats = extract_observed_palette(cutouts)
    prior = _heuristic_prior(prompt, sample_name, color_names, lab_stats)

    if backend in ("auto", "gemini"):
        prior = _try_gemini_vlm_prior(prompt, cutouts, prior)

    if cache_path is not None:
        prior.save(cache_path)
    return prior
