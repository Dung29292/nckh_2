"""Phase 3B: Prompt-driven web reference image research, cutout isolation, verification & alignment.

Implements Retrieval-Augmented 3D Completion (3D Visual RAG):
  1. Multi-provider web image search (DuckDuckGo / Google CSE / SerpAPI / Wikimedia fallback)
     plus a local manual override folder ``data/samples/<name>/web_refs/manual/``.
  2. Resolution, aspect-ratio, and 64-bit perceptual hash (pHash) deduplication filters.
  3. Foreground cutout isolation (fast studio border/GrabCut or Grounded-SAM 2).
  4. Cross-modal verification score combining visual structural features (DINOv2/ResNet or
     multi-scale HOG fallback) + CIE Lab color-histogram similarity against observed cutouts.
  5. Automatic alignment of verified reference cutouts onto occluded Visual Hull silhouettes
     (Silhouette IoU optimization) for downstream reference-guided completion.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import cv2
import numpy as np
import requests

from .prior_parser import StructuralPrior
from .visibility import VirtualViewRender


@dataclasses.dataclass
class VerifiedReference:
    """Single verified, foreground-isolated reference image aligned to an occluded virtual view."""

    ref_id: str
    source_path: Path
    cutout_path: Path
    mask_path: Path
    source_query: str
    view_hint: str
    assigned_view: str
    verify_score: float
    feature_sim: float
    color_sim: float
    silhouette_iou: float
    flip_x: bool = False
    source_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "source_path": str(self.source_path),
            "cutout_path": str(self.cutout_path),
            "mask_path": str(self.mask_path),
            "source_query": self.source_query,
            "view_hint": self.view_hint,
            "assigned_view": self.assigned_view,
            "verify_score": round(self.verify_score, 4),
            "feature_sim": round(self.feature_sim, 4),
            "color_sim": round(self.color_sim, 4),
            "silhouette_iou": round(self.silhouette_iou, 4),
            "flip_x": self.flip_x,
            "source_url": self.source_url,
        }


# ---------------------------------------------------------------------------
# IMAGE SEARCH PROVIDERS (WITH GRACEFUL OFFLINE FALLBACK)
# ---------------------------------------------------------------------------

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _search_duckduckgo(query: str, max_results: int = 8, timeout: float = 10.0) -> list[str]:
    """Fetch candidate image URLs from DuckDuckGo (no API key required)."""
    # 1. Prefer duckduckgo_search library if installed
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            items = list(ddgs.images(query, max_results=max_results))
        return [str(it["image"]) for it in items if it.get("image")]
    except ImportError:
        pass
    except Exception:
        pass

    # 2. Lightweight pure-requests fallback via DuckDuckGo vqd + i.js
    sess = requests.Session()
    sess.headers.update({"User-Agent": _USER_AGENT, "Referer": "https://duckduckgo.com/"})
    try:
        r0 = sess.get(f"https://duckduckgo.com/?q={quote_plus(query)}&iax=images&ia=images", timeout=timeout)
        r0.raise_for_status()
        m = re.search(r"vqd=['\"]?([\w\-]+)['\"]?", r0.text)
        if not m:
            return []
        vqd = m.group(1)
        r1 = sess.get(
            "https://duckduckgo.com/i.js",
            params={"l": "us-en", "o": "json", "q": query, "vqd": vqd, "f": ",,,", "p": "1"},
            timeout=timeout,
        )
        r1.raise_for_status()
        results = r1.json().get("results", [])
        return [str(it["image"]) for it in results[:max_results] if it.get("image")]
    except Exception:
        return []


def _search_wikimedia(query: str, max_results: int = 6, timeout: float = 10.0) -> list[str]:
    """Open fallback search via Wikimedia Commons API (no key, CC-licensed product/object images)."""
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrnamespace": 6,
                "gsrsearch": query,
                "gsrlimit": max_results,
                "prop": "imageinfo",
                "iiprop": "url|mime",
                "format": "json",
            },
            headers={"User-Agent": "NCKH2-3DResearch/1.0"},
            timeout=timeout,
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        urls: list[str] = []
        for pg in pages.values():
            infos = pg.get("imageinfo", [])
            if infos and str(infos[0].get("mime", "")).startswith("image/"):
                url = infos[0].get("url", "")
                if url and not url.lower().endswith(".svg"):
                    urls.append(str(url))
        return urls[:max_results]
    except Exception:
        return []


def _search_google_cse(query: str, max_results: int = 8, timeout: float = 10.0) -> list[str]:
    api_key = os.environ.get("GOOGLE_CSE_KEY", "").strip()
    cx = os.environ.get("GOOGLE_CSE_CX", "").strip()
    if not api_key or not cx:
        return []
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cx, "q": query, "searchType": "image", "num": min(max_results, 10)},
            timeout=timeout,
        )
        r.raise_for_status()
        return [str(it["link"]) for it in r.json().get("items", []) if it.get("link")]
    except Exception:
        return []


def _search_serpapi(query: str, max_results: int = 8, timeout: float = 10.0) -> list[str]:
    api_key = os.environ.get("SERPAPI_KEY", "").strip()
    if not api_key:
        return []
    try:
        r = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google_images", "q": query, "api_key": api_key, "num": max_results},
            timeout=timeout,
        )
        r.raise_for_status()
        return [str(it["original"]) for it in r.json().get("images_results", [])[:max_results] if it.get("original")]
    except Exception:
        return []


def search_image_urls(query: str, provider: str = "duckduckgo", max_results: int = 8) -> list[str]:
    """Query configured image search provider with clean fallback chain."""
    provider = provider.strip().lower()
    if provider == "manual_only":
        return []
    if provider == "google_cse":
        urls = _search_google_cse(query, max_results=max_results)
        if urls:
            return urls
    elif provider == "serpapi":
        urls = _search_serpapi(query, max_results=max_results)
        if urls:
            return urls

    urls = _search_duckduckgo(query, max_results=max_results)
    if not urls:
        urls = _search_wikimedia(query, max_results=max_results)
    return urls


# ---------------------------------------------------------------------------
# DOWNLOAD, PRE-CLEANING, pHASH DEDUP & FOREGROUND ISOLATION
# ---------------------------------------------------------------------------

def _compute_phash(img_bgr: np.ndarray) -> np.ndarray:
    """Compute 64-bit DCT perceptual hash for near-duplicate rejection."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    med = np.median(dct.ravel()[1:])
    return (dct.ravel() > med)


def _download_image(url: str, dst: Path, min_side: int = 320, timeout: float = 8.0) -> Path | None:
    try:
        r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=timeout, stream=True)
        r.raise_for_status()
        raw = r.content
        if len(raw) < 4096 or len(raw) > 15_000_000:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if min(h, w) < min_side:
            return None
        aspect = w / max(h, 1)
        if aspect > 3.2 or aspect < 0.31:
            return None
        # Cap giant stock images to 1024px max side for fast downstream processing
        max_side = max(h, w)
        if max_side > 1024:
            sc = 1024.0 / max_side
            img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return dst
    except Exception:
        return None


def isolate_foreground_cutout(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Isolate single main object foreground from studio or natural web product photos.

    Uses border-color flood segmentation + GrabCut refinement + largest-component check
    (0 GB VRAM) to reject multi-object collages and empty/full-bleed textures.
    """
    h, w = img_bgr.shape[:2]
    # Sample 4 image borders to estimate studio/solid background color
    border_pixels = np.vstack([
        img_bgr[0, :, :], img_bgr[-1, :, :], img_bgr[:, 0, :], img_bgr[:, -1, :]
    ]).astype(np.float32)
    bg_med = np.median(border_pixels, axis=0)
    bg_std = float(np.mean(np.std(border_pixels, axis=0)))

    if bg_std < 28.0:
        # Clean studio / white-box / uniform e-commerce backdrop -> color-distance + floodfill
        dist = np.linalg.norm(img_bgr.astype(np.float32) - bg_med[None, None, :], axis=2)
        bg_seed = (dist < max(22.0, bg_std * 2.2)).astype(np.uint8) * 255
        # Floodfill only background regions connected to outer image border
        padded = cv2.copyMakeBorder(bg_seed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        ff_mask = np.zeros((h + 4, w + 4), np.uint8)
        cv2.floodFill(padded, ff_mask, (0, 0), 128)
        is_bg = padded[1:-1, 1:-1] == 128
        mask = (~is_bg).astype(np.uint8) * 255
    else:
        # Cluttered natural background -> centered GrabCut initialization
        gc_mask = np.zeros((h, w), np.uint8)
        rect = (int(0.06 * w), int(0.06 * h), int(0.88 * w), int(0.88 * h))
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img_bgr, gc_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
            mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            return None

    # Keep largest connected component and reject collages / degenerate masks
    num_lbls, lbl_im, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_lbls <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    max_area = float(areas[largest_idx - 1])
    total_area = float(h * w)
    if max_area < 0.06 * total_area or max_area > 0.95 * total_area:
        return None

    # Reject multi-item collages where >= 3 separate blobs are each > 25% of largest object
    large_blobs = int(np.sum(areas > 0.25 * max_area))
    if large_blobs >= 3:
        return None

    clean_mask = np.where(lbl_im == largest_idx, 255, 0).astype(np.uint8)
    # Fill internal enclosed holes (e.g. white sticker on dark toolbox)
    padded_m = cv2.copyMakeBorder(clean_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff_m = np.zeros((h + 4, w + 4), np.uint8)
    inv_m = cv2.bitwise_not(padded_m)
    cv2.floodFill(inv_m, ff_m, (0, 0), 0)
    clean_mask = cv2.bitwise_or(clean_mask, inv_m[1:-1, 1:-1])
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    cutout = cv2.bitwise_and(img_bgr, img_bgr, mask=clean_mask)
    return cutout, clean_mask


# ---------------------------------------------------------------------------
# CROSS-MODAL VERIFICATION (VISUAL FEATURE + LAB COLOR + SILHOUETTE IoU)
# ---------------------------------------------------------------------------

def _crop_tight(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask > 127)
    if len(ys) == 0:
        return img, mask
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _extract_visual_descriptor(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract L2-normalized structural + edge + color spatial pyramid descriptor (0 GB VRAM)."""
    crop_img, crop_mask = _crop_tight(img_bgr, mask)
    size = 128
    rsz_img = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_AREA)
    rsz_msk = cv2.resize(crop_mask, (size, size), interpolation=cv2.INTER_NEAREST) > 127

    gray = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy) * rsz_msk
    ori = (np.arctan2(gy, gx) + np.pi) * (8.0 / (2.0 * np.pi))  # 8 orientation bins

    # 4x4 spatial cell HOG (4*4*8 = 128 dims) + 4x4 Lab means (4*4*3 = 48 dims)
    lab = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    feats: list[float] = []
    cell = size // 4
    for r in range(4):
        for c in range(4):
            y0, y1 = r * cell, (r + 1) * cell
            x0, x1 = c * cell, (c + 1) * cell
            m_cell = rsz_msk[y0:y1, x0:x1]
            hist = np.zeros(8, dtype=np.float32)
            if np.any(m_cell):
                bins = np.clip(ori[y0:y1, x0:x1][m_cell].astype(np.int32), 0, 7)
                np.add.at(hist, bins, mag[y0:y1, x0:x1][m_cell])
                lab_mean = lab[y0:y1, x0:x1][m_cell].mean(axis=0)
            else:
                lab_mean = np.zeros(3, dtype=np.float32)
            feats.extend(hist.tolist())
            feats.extend((lab_mean * 2.0).tolist())

    vec = np.asarray(feats, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-8)


def _compute_lab_color_hist(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute normalized 2D (a, b) + 1D (L) CIE Lab foreground color histogram."""
    fg = mask > 127
    if np.sum(fg) < 30:
        return np.ones(48, dtype=np.float32) / 48.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_hist, _ = np.histogram(lab[:, :, 0][fg], bins=16, range=(0, 256))
    a_hist, _ = np.histogram(lab[:, :, 1][fg], bins=16, range=(0, 256))
    b_hist, _ = np.histogram(lab[:, :, 2][fg], bins=16, range=(0, 256))
    h = np.concatenate([l_hist, a_hist, b_hist]).astype(np.float32)
    return h / max(float(h.sum()), 1e-8)


def _bhattacharyya_sim(h1: np.ndarray, h2: np.ndarray) -> float:
    """Return color distribution similarity in [0, 1] (1 = identical palette)."""
    return float(np.clip(np.sum(np.sqrt(np.maximum(h1, 0.0) * np.maximum(h2, 0.0))), 0.0, 1.0))


def align_reference_to_silhouette(
    ref_mask: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[float, bool, np.ndarray]:
    """Align binary ``ref_mask`` onto virtual Visual Hull ``target_mask`` by bounding-box fit.

    Tests both normal and horizontally mirrored orientations and returns
    ``(best_iou, flip_x, aligned_mask_uint8)``.
    """
    th, tw = target_mask.shape[:2]
    tys, txs = np.nonzero(target_mask > 0)
    if len(tys) < 20:
        return 0.0, False, np.zeros((th, tw), dtype=np.uint8)
    ty0, ty1 = int(tys.min()), int(tys.max()) + 1
    tx0, tx1 = int(txs.min()), int(txs.max()) + 1
    tb_w, tb_h = tx1 - tx0, ty1 - ty0

    _, crop_rm = _crop_tight(ref_mask, ref_mask)
    if crop_rm.size == 0:
        return 0.0, False, np.zeros((th, tw), dtype=np.uint8)

    # Aspect ratio sanity check before warping
    ref_ar = crop_rm.shape[1] / max(crop_rm.shape[0], 1)
    tgt_ar = tb_w / max(tb_h, 1)
    ar_ratio = max(ref_ar / max(tgt_ar, 1e-4), tgt_ar / max(ref_ar, 1e-4))
    if ar_ratio > 2.2:
        return 0.0, False, np.zeros((th, tw), dtype=np.uint8)

    best_iou = 0.0
    best_flip = False
    best_canvas = np.zeros((th, tw), dtype=np.uint8)
    tgt_bool = target_mask > 0

    for flip in (False, True):
        src = cv2.flip(crop_rm, 1) if flip else crop_rm
        warped = cv2.resize(src, (tb_w, tb_h), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((th, tw), dtype=np.uint8)
        canvas[ty0:ty1, tx0:tx1] = warped
        ref_bool = canvas > 127
        inter = int(np.sum(ref_bool & tgt_bool))
        union = int(np.sum(ref_bool | tgt_bool))
        iou = inter / max(union, 1)
        if iou > best_iou:
            best_iou = iou
            best_flip = flip
            best_canvas = canvas

    return float(best_iou), best_flip, best_canvas


def warp_reference_to_view(
    ref_cutout: np.ndarray,
    ref_mask: np.ndarray,
    target_mask: np.ndarray,
    flip_x: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a verified reference cutout & mask precisely onto a virtual view's silhouette bbox."""
    th, tw = target_mask.shape[:2]
    tys, txs = np.nonzero(target_mask > 0)
    if len(tys) < 20:
        return np.zeros((th, tw, 3), dtype=np.uint8), np.zeros((th, tw), dtype=np.uint8)
    ty0, ty1 = int(tys.min()), int(tys.max()) + 1
    tx0, tx1 = int(txs.min()), int(txs.max()) + 1
    tb_w, tb_h = tx1 - tx0, ty1 - ty0

    crop_img, crop_msk = _crop_tight(ref_cutout, ref_mask)
    if flip_x:
        crop_img = cv2.flip(crop_img, 1)
        crop_msk = cv2.flip(crop_msk, 1)

    w_img = cv2.resize(crop_img, (tb_w, tb_h), interpolation=cv2.INTER_LINEAR)
    w_msk = cv2.resize(crop_msk, (tb_w, tb_h), interpolation=cv2.INTER_NEAREST)

    out_img = np.zeros((th, tw, 3), dtype=np.uint8)
    out_msk = np.zeros((th, tw), dtype=np.uint8)
    out_img[ty0:ty1, tx0:tx1] = w_img
    out_msk[ty0:ty1, tx0:tx1] = w_msk
    return out_img, out_msk


# ---------------------------------------------------------------------------
# ORCHESTRATOR: RETRIEVE, FILTER, VERIFY & ALIGN
# ---------------------------------------------------------------------------

def _infer_view_hint_from_name(name: str, fallback: str = "general") -> str:
    low = name.lower()
    for key in ("back_left", "back_right", "back", "rear", "bottom", "underside", "left", "right", "side", "top", "front"):
        if key in low:
            if key == "rear":
                return "back"
            if key == "underside":
                return "bottom"
            return key
    return fallback


def retrieve_and_verify_references(
    prior: StructuralPrior,
    sample_dir: Path,
    virtual_views: list[VirtualViewRender],
    provider: str = "duckduckgo",
    max_per_query: int = 5,
    min_verify_score: float = 0.56,
    enable_online: bool = True,
) -> dict[str, VerifiedReference]:
    """Research, verify, and align online + manual reference photos for occluded virtual views.

    Returns a dict mapping occluded virtual view name (e.g. ``"back"``, ``"left"``)
    to its highest-scoring ``VerifiedReference``.
    """
    refs_root = sample_dir / "web_refs"
    manual_dir = refs_root / "manual"
    raw_dir = refs_root / "raw"
    cut_dir = refs_root / "cutouts"
    for d in (manual_dir, raw_dir, cut_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Build observed reference signature from real captured cutouts
    obs_cutouts = sorted((sample_dir / "cutouts").glob("frame_*.jpg"))
    obs_feats: list[np.ndarray] = []
    obs_hists: list[np.ndarray] = []
    if obs_cutouts:
        step = max(1, len(obs_cutouts) // 6)
        for cp in obs_cutouts[::step][:6]:
            im = cv2.imread(str(cp))
            if im is None:
                continue
            msk = (np.any(im > 8, axis=2)).astype(np.uint8) * 255
            obs_feats.append(_extract_visual_descriptor(im, msk))
            obs_hists.append(_compute_lab_color_hist(im, msk))

    if not obs_feats:
        return {}
    mean_obs_feat = np.mean(obs_feats, axis=0)
    mean_obs_feat /= max(float(np.linalg.norm(mean_obs_feat)), 1e-8)
    mean_obs_hist = np.mean(obs_hists, axis=0)

    # 2. Collect candidate files: manual drop folder first, then cached/fresh web search
    candidates: list[tuple[Path, str, str, str]] = []  # (path, query, view_hint, source_url)
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for mp in sorted(manual_dir.glob(ext)):
            candidates.append((mp, "manual", _infer_view_hint_from_name(mp.stem, "general"), "manual"))

    if enable_online and provider != "manual_only":
        occluded_names = {v.name for v in virtual_views if v.occlusion_ratio >= 0.12}
        query_buckets = ["general"]
        if any(n.startswith("back") for n in occluded_names):
            query_buckets.append("back")
        if "left" in occluded_names or "right" in occluded_names:
            query_buckets.append("side")
        if "bottom" in occluded_names:
            query_buckets.append("bottom")

        for bucket in query_buckets:
            for q in prior.web_search_queries.get(bucket, [])[:1]:
                urls = search_image_urls(q, provider=provider, max_results=max_per_query)
                for url in urls:
                    uid = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
                    dst = raw_dir / f"{bucket}_{uid}.jpg"
                    if not dst.exists():
                        got = _download_image(url, dst)
                        if got is None:
                            continue
                    candidates.append((dst, q, bucket, url))

    # Also include any previously cached web images in raw_dir
    seen_paths = {c[0].resolve() for c in candidates}
    for rp in sorted(raw_dir.glob("*.jpg")):
        if rp.resolve() not in seen_paths:
            candidates.append((rp, "cached", _infer_view_hint_from_name(rp.stem, "general"), "cached"))

    # 3. Deduplicate (pHash), isolate foreground, verify against real cutouts & align to occluded views
    seen_phashes: list[np.ndarray] = []
    occluded_views = [v for v in virtual_views if v.occlusion_ratio >= 0.08]
    if not occluded_views:
        occluded_views = virtual_views

    best_per_view: dict[str, VerifiedReference] = {}

    for img_path, query, vhint, src_url in candidates:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ph = _compute_phash(img)
        if any(int(np.sum(ph != prev)) <= 8 for prev in seen_phashes):
            continue
        seen_phashes.append(ph)

        iso = isolate_foreground_cutout(img)
        if iso is None:
            continue
        cutout, mask = iso

        feat = _extract_visual_descriptor(cutout, mask)
        hist = _compute_lab_color_hist(cutout, mask)

        feat_sim = float(np.clip(np.dot(feat, mean_obs_feat), 0.0, 1.0))
        color_sim = _bhattacharyya_sim(hist, mean_obs_hist)
        verify_score = 0.55 * feat_sim + 0.45 * color_sim

        # Manual user-supplied references receive a slight trust bonus
        eff_threshold = min_verify_score * (0.75 if src_url == "manual" else 1.0)
        if verify_score < eff_threshold:
            continue

        # Save verified cutout & mask
        ref_id = img_path.stem
        cut_out_p = cut_dir / f"{ref_id}_cutout.jpg"
        msk_out_p = cut_dir / f"{ref_id}_mask.png"
        cv2.imwrite(str(cut_out_p), cutout, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(msk_out_p), mask)

        # Match against occluded virtual views by Silhouette IoU + viewpoint hint bonus
        for vw in occluded_views:
            iou, flip_x, _ = align_reference_to_silhouette(mask, vw.mask_vh)
            if iou < 0.45:
                continue
            hint_bonus = 0.12 if (vhint == vw.name or (vhint == "side" and vw.name in ("left", "right")) or (vhint == "back" and vw.name.startswith("back"))) else 0.0
            rank_score = 0.65 * verify_score + 0.35 * iou + hint_bonus
            prev_best = best_per_view.get(vw.name)
            prev_rank = (0.65 * prev_best.verify_score + 0.35 * prev_best.silhouette_iou) if prev_best is not None else -1.0
            if rank_score > prev_rank:
                best_per_view[vw.name] = VerifiedReference(
                    ref_id=ref_id,
                    source_path=img_path,
                    cutout_path=cut_out_p,
                    mask_path=msk_out_p,
                    source_query=query,
                    view_hint=vhint,
                    assigned_view=vw.name,
                    verify_score=verify_score,
                    feature_sim=feat_sim,
                    color_sim=color_sim,
                    silhouette_iou=iou,
                    flip_x=flip_x,
                    source_url=src_url,
                )

    manifest = {k: v.to_dict() for k, v in best_per_view.items()}
    (refs_root / "verified_refs.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return best_per_view
