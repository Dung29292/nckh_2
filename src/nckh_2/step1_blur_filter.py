"""Buoc 1: loc frame mo bang Laplacian variance.

Vao : data/samples/<mau>/video.mp4
Ra  : data/samples/<mau>/frames_sharp/frame_00001.jpg ...  + blur_scores.csv
"""
import csv

import cv2
import numpy as np
from tqdm import tqdm

from . import paths


def laplacian_variance(frame: np.ndarray, score_width: int) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w > score_width:
        gray = cv2.resize(gray, (score_width, int(h * score_width / w)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_indices(scores, window: int, min_ratio: float, max_frames: int) -> list[int]:
    """Moi cua so `window` frame lay frame net nhat (giu do phu deu quanh vat the),
    bo frame qua mo so voi trung vi, roi gioi han max_frames bang lay mau deu."""
    scores = np.asarray(scores)
    threshold = min_ratio * float(np.median(scores))
    chosen = []
    for start in range(0, len(scores), window):
        seg = scores[start:start + window]
        i = start + int(np.argmax(seg))
        if scores[i] >= threshold:
            chosen.append(i)
    if len(chosen) > max_frames:
        pick = np.linspace(0, len(chosen) - 1, max_frames).round().astype(int)
        chosen = [chosen[j] for j in pick]
    return chosen


def run(sample_name: str, cfg: dict) -> None:
    b = cfg["blur_filter"]
    sdir = paths.sample_dir(sample_name)
    video = sdir / cfg["video_file"]
    if not video.exists():
        raise FileNotFoundError(f"Khong thay video: {video}")

    # Luot 1: chi tinh diem net cho moi frame (khong luu anh -> tiet kiem o dia)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    scores = []
    for _ in tqdm(range(total or 10**9), total=total, desc="Buoc 1a: tinh do net"):
        ok, frame = cap.read()
        if not ok:
            break
        scores.append(laplacian_variance(frame, b["score_width"]))
    cap.release()
    if not scores:
        raise RuntimeError("Video rong hoac khong doc duoc.")

    keep = select_indices(scores, b["window"], b["min_ratio"], b["max_frames"])
    keep_set = set(keep)

    # Luot 2: ghi cac frame duoc chon
    out = sdir / "frames_sharp"
    out.mkdir(exist_ok=True)
    for old in out.glob("frame_*.jpg"):
        old.unlink()

    cap = cv2.VideoCapture(str(video))
    n = 0
    for idx in tqdm(range(len(scores)), desc="Buoc 1b: ghi frame net"):
        ok, frame = cap.read()
        if not ok:
            break
        if idx not in keep_set:
            continue
        h, w = frame.shape[:2]
        scale = b["save_max_side"] / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        n += 1
        cv2.imwrite(str(out / f"frame_{n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, b["jpg_quality"]])
    cap.release()

    with open(sdir / "blur_scores.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame_index", "laplacian_var", "kept"])
        for i, s in enumerate(scores):
            wr.writerow([i, f"{s:.2f}", int(i in keep_set)])

    print(f"[Buoc 1] {len(scores)} frame -> giu {n} frame net -> {out}")
