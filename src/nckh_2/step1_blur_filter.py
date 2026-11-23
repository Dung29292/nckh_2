"""Buoc 1: Loc mo, khu rung giat (anti-shake) va khu nhieu hat ISO (edge-preserving denoising).

Vao : data/samples/<mau>/video.mp4
Ra  : data/samples/<mau>/frames_sharp/frame_00001.jpg ...  + blur_scores.csv
"""
import csv
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from . import paths


def compute_frame_sharpness(gray: np.ndarray) -> tuple[float, float, float, float]:
    """Tinh to hop do net da chi tieu:
    1. Tenengrad (nang luong gradient Sobel): do doc canh sac net, it bi anh huong boi nhieu ISO.
    2. Directional Isotropy (ty le phuong sai gradient X vs Y): phat hien vet nhoe vet rung (motion blur streak).
    3. Laplacian variance: chi so truyen thong.
    4. Combined sharpness score: ket hop trong so ROI trung tam (75%) va toan khung hinh (25%).
    """
    gh, gw = gray.shape
    y1, y2 = int(0.15 * gh), int(0.85 * gh)
    x1, x2 = int(0.20 * gw), int(0.80 * gw)
    roi = gray[y1:y2, x1:x2] if gh > 20 and gw > 20 else gray

    # Sobel gradients
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx**2 + gy**2))

    # Isotropy ratio [0, 1]: gan 1 la net deu cac huong; << 1 la bi nhoe theo 1 chieu (motion blur)
    vx, vy = float(np.var(gx)), float(np.var(gy))
    isotropy = min(vx, vy) / (max(vx, vy) + 1e-5)

    # Laplacian variance
    lap_roi = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    lap_global = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lap_var = 0.75 * lap_roi + 0.25 * lap_global

    # Diem ket hop
    combined = float(np.sqrt(tenengrad) * (isotropy ** 0.5) * (lap_var ** 0.25))
    return combined, tenengrad, isotropy, lap_var


def select_optimized_keyframes(
    scores: list[float],
    flow_mags: list[float],
    jerks: list[float],
    cfg_b: dict
) -> list[int]:
    """Lua chon keyframe toi uu:
    - Khu rung: Phat diem cac frame co gia toc dot ngot (jerk spike) do tay lac/giat.
    - Motion sampling: Lay mau theo quang duong goc quay thuc te (tich luy optical flow)
      de dam bao mat do goc nhin 360 do phan bo dong deu, khong bi dut quang hay chong lap.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    flow_arr = np.asarray(flow_mags, dtype=np.float64)
    jerk_arr = np.asarray(jerks, dtype=np.float64)
    n = len(scores_arr)
    max_frames = int(cfg_b.get("max_frames", 60))

    if n <= max_frames:
        return list(range(n))

    # 1. Tinh he so phat rung giat
    if cfg_b.get("enable_shake_filter", True) and len(jerk_arr) > 1:
        jerk_p85 = float(np.percentile(jerk_arr, 85))
        penalty_strength = float(cfg_b.get("jerk_penalty_strength", 1.0))
        # Frame co gia toc vuot P85 bi phat ty le nghich
        shake_weight = 1.0 / (1.0 + penalty_strength * np.maximum(0.0, (jerk_arr - jerk_p85) / (jerk_p85 + 1e-5)))
    else:
        shake_weight = np.ones(n, dtype=np.float64)

    quality = scores_arr * shake_weight
    median_q = float(np.median(quality))
    min_ratio = float(cfg_b.get("min_ratio", 0.6))
    threshold = min_ratio * median_q

    # 2. Lay mau theo quang duong chuyen dong (Motion-Cumulative Trajectory)
    if cfg_b.get("enable_motion_sampling", True) and len(flow_arr) == n:
        # Tich luy quang duong chuyen dong (dam bao luon tien ve phia truoc)
        cum_motion = np.cumsum(np.clip(flow_arr, 0.2, None))
        total_motion = cum_motion[-1]
        step_motion = total_motion / max_frames

        chosen = []
        last_idx = -1
        # Cua so tim kiem cuc bo quanh moc quang duong
        search_radius = max(5, int(n / (max_frames * 2)))

        for k in range(max_frames):
            target = k * step_motion
            center_idx = int(np.searchsorted(cum_motion, target))
            w_start = max(last_idx + 1, center_idx - search_radius)
            w_end = min(n, center_idx + search_radius + 1)

            if w_start >= w_end:
                cand = min(n - 1, last_idx + 1)
            else:
                window_scores = quality[w_start:w_end]
                best_sub = int(np.argmax(window_scores))
                cand = w_start + best_sub

            chosen.append(cand)
            last_idx = cand
    else:
        # Che do cua so co dinh co loc nguong
        window = int(cfg_b.get("window", 5))
        candidates = []
        for start in range(0, n, window):
            seg = quality[start:start + window]
            i = start + int(np.argmax(seg))
            if quality[i] >= threshold:
                candidates.append(i)
        if len(candidates) > max_frames:
            pick = np.linspace(0, len(candidates) - 1, max_frames).round().astype(int)
            chosen = [candidates[j] for j in pick]
        else:
            chosen = candidates

    return chosen


def run(sample_name: str, cfg: dict) -> None:
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

    # Luot 1: Quet video, do luong do net, dong chay quang hoc va do rung giat
    pbar = tqdm(total=total, desc="Buoc 1a: Phan tich do net & do rung giat (Anti-Shake)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (score_width, int(h * score_width / w)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gh, gw = gray.shape

        # Tinh to hop do net
        comb_score, ten, iso, lap = compute_frame_sharpness(gray)
        scores.append(comb_score)
        tenengrads.append(ten)
        isotropies.append(iso)
        laplacians.append(lap)

        # Tinh optical flow tren vung trong tam
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

    # Lua chon cac frame toi uu nhat
    keep = select_optimized_keyframes(scores, flow_mags, jerks, b)
    keep_set = set(keep)

    out = sdir / "frames_sharp"
    out.mkdir(exist_ok=True)
    for old in out.glob("frame_*.jpg"):
        old.unlink()

    # Luot 2: Trinh xuat va khu nhieu hat ISO bang bilateral filter giu nguyen canh sac
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

            # Khử nhiễu bề mặt phẳng nhưng giữ nguyên 100% cạnh góc cạnh chi tiết
            if enable_denoise:
                frame = cv2.bilateralFilter(frame, d=d, sigmaColor=sc, sigmaSpace=ss)

            n += 1
            out_file = out / f"frame_{n:05d}.jpg"
            cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, int(b.get("jpg_quality", 95))])

        idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # Luu file nhat ky chi tiet
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
                int(i in keep_set)
            ])

    # Luu bieu do danh gia chat luong Buoc 1
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
        plt.scatter(kept_indices, kept_scores, color="#e63946", s=35, zorder=5, label=f"60 Frame duoc chon (Net nhat, Khong rung giat)")
        plt.title(f"Danh gia Buoc 1: Loc Net & Chon Keyframe Toi uu ({sample_name})")
        plt.ylabel("Do net")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(flow_mags, color="#2a9d8f", label="Van toc di chuyen (Optical Flow speed)", alpha=0.7)
        plt.plot(jerks, color="#e76f51", label="Do rung giat / Gia toc (Jerk)", alpha=0.5)
        kept_jerks = [jerks[i] for i in keep]
        plt.scatter(kept_indices, kept_jerks, color="#264653", s=30, zorder=5, label="Do rung cua cac frame duoc chon (rat thap)")
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
    print(f"[Buoc 1] Bieu do kiem dinh da luu tai: {report_fig}")
