"""Buoc 1: Loc mo, khu rung giat (anti-shake) va khu nhieu hat ISO (edge-preserving denoising).

Vao : data/samples/<mau>/video.mp4
Ra  : data/samples/<mau>/frames_sharp/frame_00001.jpg ...  + blur_scores.csv
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from nckh_2.core import paths
from nckh_2.core.types import SceneContext, StageResult


def compute_frame_sharpness(gray: np.ndarray) -> tuple[float, float, float, float]:
    """Tinh to hop do net da chi tieu (Tenengrad, Directional Isotropy, Laplacian variance)."""
    gh, gw = gray.shape
    y1, y2 = int(0.15 * gh), int(0.85 * gh)
    x1, x2 = int(0.20 * gw), int(0.80 * gw)
    roi = gray[y1:y2, x1:x2] if gh > 20 and gw > 20 else gray

    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx**2 + gy**2))

    vx, vy = float(np.var(gx)), float(np.var(gy))
    isotropy = min(vx, vy) / (max(vx, vy) + 1e-5)

    lap_roi = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    lap_global = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lap_var = 0.75 * lap_roi + 0.25 * lap_global

    combined = float(np.sqrt(tenengrad) * (isotropy ** 0.5) * (lap_var ** 0.25))
    return combined, tenengrad, isotropy, lap_var


def select_optimized_keyframes(
    scores: list[float],
    flow_mags: list[float],
    jerks: list[float],
    cfg_b: dict[str, Any],
    laplacians: list[float] | None = None,
    cfg_p: dict[str, Any] | None = None,
) -> list[int]:
    scores_arr = np.asarray(scores, dtype=np.float64)
    flow_arr = np.asarray(flow_mags, dtype=np.float64)
    jerk_arr = np.asarray(jerks, dtype=np.float64)
    lap_arr = np.asarray(laplacians, dtype=np.float64) if laplacians is not None else scores_arr
    n = len(scores_arr)

    prep = cfg_p or {}
    target_frames = int(prep.get("target_frames", cfg_b.get("max_frames", 90)))
    max_frames_cap = int(prep.get("max_frames_cap", max(target_frames, 95)))
    max_frames = max(1, min(target_frames, max_frames_cap))
    min_sharpness = float(prep.get("min_sharpness_score", 0.0))
    min_parallax = float(prep.get("min_parallax_flow", 0.0))
    oversample_factor = float(prep.get("oversample_factor", 2.5))

    if n <= max_frames:
        return list(range(n))

    if cfg_b.get("enable_shake_filter", True) and len(jerk_arr) > 1:
        jerk_p85 = float(np.percentile(jerk_arr, 85))
        penalty_strength = float(cfg_b.get("jerk_penalty_strength", 1.0))
        shake_weight = 1.0 / (1.0 + penalty_strength * np.maximum(0.0, (jerk_arr - jerk_p85) / (jerk_p85 + 1e-5)))
    else:
        shake_weight = np.ones(n, dtype=np.float64)

    quality = scores_arr * shake_weight
    median_q = float(np.median(quality))
    min_ratio = float(cfg_b.get("min_ratio", 0.6))
    adaptive_q_thr = min_ratio * median_q

    # 1. Oversampling along timeline (~2.0x-2.5x target budget)
    n_oversample = min(n, max(max_frames, int(round(max_frames * oversample_factor))))
    cand_indices = np.unique(np.linspace(0, n - 1, n_oversample).round().astype(int))
    cand_mask = np.zeros(n, dtype=bool)
    cand_mask[cand_indices] = True

    # 2. Motion / Parallax verification: skip near-static segments (flow < min_parallax)
    motion_mask = np.ones(n, dtype=bool)
    if min_parallax > 0.0 and len(flow_arr) == n:
        moving = flow_arr >= min_parallax
        moving[0] = True  # always allow first frame
        if int(np.sum(moving & cand_mask)) >= max_frames:
            motion_mask = moving

    # 3. Sharpness filtering via Laplacian Variance (dynamic + adaptive fallback)
    valid_mask = cand_mask & motion_mask & (quality >= adaptive_q_thr)
    if min_sharpness > 0.0:
        lap_valid = lap_arr >= min_sharpness
        if int(np.sum(valid_mask & lap_valid)) >= max_frames:
            valid_mask = valid_mask & lap_valid
        elif int(np.sum(cand_mask & motion_mask & lap_valid)) >= max_frames:
            valid_mask = cand_mask & motion_mask & lap_valid
        else:
            pool_size = max(max_frames, int(prep.get("candidate_pool_size", n_oversample)))
            k_pool = min(n, pool_size)
            top_lap_thr = float(np.partition(lap_arr, n - k_pool)[n - k_pool])
            valid_mask = motion_mask & (lap_arr >= top_lap_thr)

    if int(np.sum(valid_mask)) < max_frames:
        top_q_thr = float(np.partition(quality, n - max_frames)[n - max_frames])
        valid_mask = quality >= top_q_thr

    # 4. Uniform Budget Binning into `max_frames` equidistant intervals
    if cfg_b.get("enable_motion_sampling", True) and len(flow_arr) == n:
        cum_coord = np.cumsum(np.clip(flow_arr, 0.2, None))
    else:
        cum_coord = np.arange(1, n + 1, dtype=np.float64)

    edges = np.linspace(float(cum_coord[0]), float(cum_coord[-1]), max_frames + 1)
    chosen_set: set[int] = set()
    chosen: list[int] = []

    for k in range(max_frames):
        lo = edges[k]
        hi = edges[k + 1]
        if k == max_frames - 1:
            in_bin = (cum_coord >= lo) & (cum_coord <= hi)
        else:
            in_bin = (cum_coord >= lo) & (cum_coord < hi)

        cands = np.nonzero(in_bin & valid_mask)[0]
        if len(cands) == 0:
            cands = np.nonzero(in_bin)[0]

        cands = [int(i) for i in cands if int(i) not in chosen_set]
        if cands:
            best_i = max(cands, key=lambda idx: float(quality[idx]))
            chosen_set.add(best_i)
            chosen.append(best_i)

    # Bo sung tu cac ung vien sac net nhat neu co bin rong do quang duong khong deu
    if len(chosen) < max_frames:
        remaining = [int(i) for i in np.argsort(-quality) if int(i) not in chosen_set]
        for idx in remaining[: max_frames - len(chosen)]:
            chosen_set.add(idx)
            chosen.append(idx)

    chosen.sort()
    return chosen[:max_frames]


def run(sample_name: str, cfg: dict[str, Any]) -> None:
    b = cfg["blur_filter"]
    sdir = paths.sample_dir(sample_name)
    video = sdir / cfg["video_file"]
    if not video.exists():
        raise FileNotFoundError(f"Khong thay video: {video}")

    score_width = int(b.get("score_width", 480))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    scores = []
    tenengrads = []
    isotropies = []
    laplacians = []
    flow_mags = []
    jerks = []

    prev_gray = None
    prev_mag = 0.0

    pbar = tqdm(total=total, desc="Buoc 1a: Phan tich do net & do rung giat (Anti-Shake)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (score_width, int(h * score_width / w)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gh, gw = gray.shape

        comb_score, ten, iso, lap = compute_frame_sharpness(gray)
        scores.append(comb_score)
        tenengrads.append(ten)
        isotropies.append(iso)
        laplacians.append(lap)

        if prev_gray is not None:
            flow = dis.calc(prev_gray, gray, None)
            y1, y2 = int(0.15 * gh), int(0.85 * gh)
            x1, x2 = int(0.20 * gw), int(0.80 * gw)
            roi_flow = flow[y1:y2, x1:x2]
            mag = float(np.median(np.linalg.norm(roi_flow, axis=2)))
            jerk = abs(mag - prev_mag)
            prev_mag = mag
        else:
            mag = 0.0
            jerk = 0.0

        flow_mags.append(mag)
        jerks.append(jerk)
        prev_gray = gray
        pbar.update(1)

    cap.release()
    pbar.close()

    if not scores:
        raise RuntimeError("Video rong hoac khong doc duoc.")

    keep = select_optimized_keyframes(
        scores,
        flow_mags,
        jerks,
        b,
        laplacians=laplacians,
        cfg_p=cfg.get("preprocess", {}),
    )
    keep_set = set(keep)

    out = sdir / "frames_sharp"
    out.mkdir(exist_ok=True)
    for old in out.glob("frame_*.jpg"):
        old.unlink()

    enable_denoise = bool(b.get("enable_denoising", True))
    d = int(b.get("denoise_d", 5))
    sc = float(b.get("denoise_sigma_color", 25))
    ss = float(b.get("denoise_sigma_space", 25))

    cap = cv2.VideoCapture(str(video))
    n = 0
    pbar = tqdm(total=len(scores), desc="Buoc 1b: Xuat frame sac net & khu nhieu ISO")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in keep_set:
            h, w = frame.shape[:2]
            scale = float(b.get("save_max_side", 1920)) / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            if enable_denoise:
                frame = cv2.bilateralFilter(frame, d=d, sigmaColor=sc, sigmaSpace=ss)

            n += 1
            out_file = out / f"frame_{n:05d}.jpg"
            cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, int(b.get("jpg_quality", 95))])

        idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    csv_path = sdir / "blur_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame_index", "sharpness", "tenengrad", "isotropy", "laplacian_var", "flow_mag", "jerk", "kept"])
        for i in range(len(scores)):
            wr.writerow([
                i,
                f"{scores[i]:.2f}",
                f"{tenengrads[i]:.1f}",
                f"{isotropies[i]:.3f}",
                f"{laplacians[i]:.2f}",
                f"{flow_mags[i]:.2f}",
                f"{jerks[i]:.2f}",
                int(i in keep_set),
            ])

    report_dir = paths.output_dir(sample_name) / "motion_analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_fig = report_dir / "step1_shake_blur_optimization.png"
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 7))
        plt.subplot(2, 1, 1)
        plt.plot(scores, color="#2b5c8f", label="Chi so Do net To hop (Tenengrad x Isotropy x Laplacian)", alpha=0.7)
        kept_indices = [i for i in keep]
        kept_scores = [scores[i] for i in keep]
        plt.scatter(kept_indices, kept_scores, color="#e63946", s=35, zorder=5, label=f"{len(keep)} Frame duoc chon")
        plt.title(f"Danh gia Buoc 1: Loc Net & Chon Keyframe Toi uu ({sample_name})")
        plt.ylabel("Do net")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(flow_mags, color="#2a9d8f", label="Van toc di chuyen (Optical Flow speed)", alpha=0.7)
        plt.plot(jerks, color="#e76f51", label="Do rung giat / Gia toc (Jerk)", alpha=0.5)
        kept_jerks = [jerks[i] for i in keep]
        plt.scatter(kept_indices, kept_jerks, color="#264653", s=30, zorder=5, label="Do rung cua frame duoc chon")
        plt.xlabel("Thu tu frame trong video goc")
        plt.ylabel("Diem van toc / Rung")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(report_fig), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[Buoc 1] Khong the luu bieu do: {e}")

    mean_jerk_kept = np.mean([jerks[i] for i in keep]) if keep else 0.0
    mean_sharp_kept = np.mean([scores[i] for i in keep]) if keep else 0.0
    print(f"[Buoc 1] Hoan tat! {len(scores)} frame -> giu {n} frame net nhat (Do net TB: {mean_sharp_kept:.1f}, Do rung giat TB: {mean_jerk_kept:.2f}) -> {out}")


class BlurFilterStage:
    stage_id = 1
    title = "Loc frame mo"

    def execute(self, ctx: SceneContext) -> StageResult:
        run(ctx.sample_name, ctx.config)
        ctx.artifacts["frames_sharp"] = ctx.sample_dir / "frames_sharp"
        return StageResult(stage_id=self.stage_id, stage_name=self.title, success=True, artifacts=ctx.artifacts)
