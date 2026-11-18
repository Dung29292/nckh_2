"""Buoc 2: tach phong bang Grounded-SAM 2 (Grounding DINO + SAM 2 video predictor).

Vao : data/samples/<mau>/frames_sharp/frame_00001.jpg ...
Ra  : data/samples/<mau>/masks/frame_00001.png ...  (mask nhi phan 0/255, giu on dinh qua thoi gian)
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from . import paths

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _detect_box(frames_dir: Path, frame_files: list[str], cfg_m: dict):
    """Chay Grounding DINO tim bounding box vat the tren nhieu keyframe rai deu khap video.
    Ho tro Multi-Keyframe Focus Box de giu tron ven cac bo phan cung mau (nhu quai xach den
    trung mau man hinh den phia sau) va khong bi mat dau khi lat 6 mat vat the."""
    cache = paths.MODELS_DIR / "grounding_dino"
    try:
        processor = AutoProcessor.from_pretrained(cfg_m["dino_model_id"], cache_dir=cache, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            cfg_m["dino_model_id"], cache_dir=cache, local_files_only=True
        ).to(DEVICE)
    except Exception:
        processor = AutoProcessor.from_pretrained(cfg_m["dino_model_id"], cache_dir=cache)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            cfg_m["dino_model_id"], cache_dir=cache
        ).to(DEVICE)
    model.eval()

    n = len(frame_files)
    k = min(max(int(cfg_m.get("detect_frames", 10)), 10), n)
    idxs = np.linspace(0, n - 1, k).round().astype(int)

    raw_prompt = cfg_m.get("prompt", "object").strip().lower()
    if "box" in raw_prompt:
        prompts_to_try = ["toolbox. box. case.", raw_prompt, "object"]
    else:
        prompts_to_try = [raw_prompt]
        for alt in ["toolbox. box. case.", "object", "item", "product"]:
            if alt not in raw_prompt and alt not in prompts_to_try:
                prompts_to_try.append(alt)

    box_thresh = float(cfg_m.get("box_threshold", 0.28))
    text_thresh = float(cfg_m.get("text_threshold", 0.22))
    pad_frac = float(cfg_m.get("box_pad_frac", 0.02))
    pad_bottom = float(cfg_m.get("base_pad_bottom_frac", 0.0))

    best = None  # (score, frame_idx, padded_box, raw_box, prompt_used)
    keyframe_boxes: dict[int, list[float]] = {}

    for p_try in prompts_to_try:
        p_query = p_try if p_try.endswith(".") else f"{p_try}."
        print(f"[Buoc 2] Thu nhan dien Focus Box voi prompt: '{p_try}' tren {len(idxs)} keyframes...")

        for i in tqdm(idxs, desc=f"Buoc 2a: tim '{p_try}'"):
            img = Image.open(frames_dir / frame_files[i]).convert("RGB")
            w_img, h_img = img.size
            inputs = processor(images=img, text=p_query, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
            try:
                result = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    box_threshold=box_thresh,
                    text_threshold=text_thresh,
                    target_sizes=[img.size[::-1]],
                )[0]
            except TypeError:
                result = processor.post_process_grounded_object_detection(
                    outputs,
                    threshold=box_thresh,
                    text_threshold=text_thresh,
                    target_sizes=[img.size[::-1]],
                )[0]
            if len(result["scores"]) == 0:
                continue

            frame_best = None
            for score_t, box_t in zip(result["scores"], result["boxes"]):
                s = float(score_t)
                bx = box_t.tolist()
                bw = bx[2] - bx[0]
                bh = bx[3] - bx[1]
                # Bo qua box qua nho (< 2% dien tich) hoac chiem toan bo khung hinh (> 82%)
                if bw * bh < 0.02 * (w_img * h_img) or bw * bh > 0.82 * (w_img * h_img):
                    continue
                cx = (bx[0] + bx[2]) / 2.0
                cy = (bx[1] + bx[3]) / 2.0
                dist_center = np.hypot(cx - w_img / 2.0, cy - h_img / 2.0) / (np.hypot(w_img, h_img) / 2.0)
                adj_score = s * (1.0 - 0.2 * dist_center)

                if frame_best is None or adj_score > frame_best[0]:
                    frame_best = (adj_score, bx, (w_img, h_img))

            if frame_best is not None:
                adj_s, rbx, (w_img, h_img) = frame_best
                bw = rbx[2] - rbx[0]
                bh = rbx[3] - rbx[1]
                # Mo rong nhe dinh (top) va 2 ben de bao toan quai xach, mo rong day 3.5% de om tron tem dan trang
                bot_pad = max(pad_frac, 0.035) + pad_bottom
                pbox = [
                    max(0.0, rbx[0] - (pad_frac + pad_bottom * 0.4) * bw),
                    max(0.0, rbx[1] - max(pad_frac, 0.025) * bh),
                    min(float(w_img), rbx[2] + (pad_frac + pad_bottom * 0.4) * bw),
                    min(float(h_img), rbx[3] + bot_pad * bh),
                ]
                if adj_s >= 0.30:
                    keyframe_boxes[int(i)] = pbox
                if best is None or adj_s > best[0]:
                    best = (adj_s, int(i), pbox, rbx, p_try)

        if best is not None and len(keyframe_boxes) >= 2:
            break

    del model, processor
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError(
            f'Khong tim thay vat the voi cac prompt {prompts_to_try} trong {k} frame kiem tra. '
            'Thu truyen --prompt "<ten_vat_the>" hoac giam box_threshold trong configs/default.yaml.'
        )

    score, frame_idx, padded_box, raw_box, prompt_used = best
    if frame_idx not in keyframe_boxes:
        keyframe_boxes[frame_idx] = padded_box

    extra_msg = f", base_pad={pad_bottom:.0%}" if pad_bottom > 0 else ""
    print(
        f"[Buoc 2] Nhan dien thanh cong '{prompt_used}' tren {len(keyframe_boxes)} keyframes "
        f"(tot nhat: {frame_files[frame_idx]}, score={score:.2f}, box={[round(v, 1) for v in padded_box]}{extra_msg})"
    )
    return frame_idx, padded_box, raw_box, keyframe_boxes


def _build_sam2_prompts(
    img_path: Path,
    raw_box: list[float],
    pad_bottom: float,
    is_box: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Tao diem prompt duong (tam vat the) va diem am (nen/sach o duoi) cho SAM 2.
    Chi ap dung diem am nghiem ngat khi target la hinh hop (box) dat tren sach/de.
    Voi chuot (mouse) hoac cac vat the khac, chi dung bounding box chuan tu Grounding DINO
    de tranh diem am de len vien vat the lam mat mask (collpase) khi xoay."""
    if pad_bottom > 0 or not is_box:
        return None, None
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = raw_box
    bw, bh = x2 - x1, y2 - y1
    if bw < 20 or bh < 20:
        return None, None
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_h, r_w = max(1, int(0.15 * bh)), max(1, int(0.15 * bw))
    y_lo, y_hi = max(0, int(cy) - r_h), min(h_img, int(cy) + r_h)
    x_lo, x_hi = max(0, int(cx) - r_w), min(w_img, int(cx) + r_w)
    center_patch = lab[y_lo:y_hi, x_lo:x_hi]
    center_mean = center_patch.mean(axis=(0, 1))
    center_std = center_patch.std(axis=(0, 1)) + 10.0

    pos_pts = [
        [cx, cy],
        [cx, max(y1 + 5, cy - 0.18 * bh)],
        [max(x1 + 5, cx - 0.18 * bw), cy],
        [min(x2 - 5, cx + 0.18 * bw), cy],
        [cx, min(y2 - 10, cy + 0.28 * bh)], # Diem mo neo o nua duoi (bao ve tem dan/nhan trang khoi bi khoet lo)
    ]
    # Chi dat diem am STRICTLY ben duoi bounding box (khong dat ben trong [y1, y2] vat the)
    # de tranh diem am chem thung mat vat the (nhu hop pingpong co tem trang va nap cam)
    cand_neg = [
        [cx, min(h_img - 5, y2 + 0.05 * bh)],
        [x1 + 0.20 * bw, min(h_img - 5, y2 + 0.05 * bh)],
        [x2 - 0.20 * bw, min(h_img - 5, y2 + 0.05 * bh)],
        [cx, min(h_img - 5, y2 + 0.08 * bh)],
    ]
    neg_pts = []
    for px, py in cand_neg:
        ix = int(np.clip(px, 0, w_img - 1))
        iy = int(np.clip(py, 0, h_img - 1))
        dist = float(np.linalg.norm((lab[iy, ix] - center_mean) / center_std))
        if dist > 1.8 or py > y2 + 2:
            neg_pts.append([px, py])

    all_pts = np.array(pos_pts + neg_pts, dtype=np.float32)
    all_labels = np.array([1] * len(pos_pts) + [0] * len(neg_pts), dtype=np.int32)
    return all_pts, all_labels


def run(sample_name: str, cfg: dict) -> None:
    m = cfg["masking"]
    sdir = paths.sample_dir(sample_name)
    frames_dir = sdir / "frames_sharp"
    frame_files = sorted(p.name for p in frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"Khong co frame trong {frames_dir}, chay Buoc 1 truoc.")

    anchor_idx, box, raw_box, keyframe_boxes = _detect_box(frames_dir, frame_files, m)
    pad_bottom = float(m.get("base_pad_bottom_frac", 0.0))
    use_point_refinement = bool(m.get("use_point_refinement", False))
    pts_prompt, lbl_prompt = (None, None)
    if use_point_refinement:
        prompt_str = str(m.get("prompt", "object")).strip().lower()
        prefix = sample_name.split("_")[0].lower()
        is_box = prefix in ("box", "carton", "cube") or "box" in prompt_str
        pts_prompt, lbl_prompt = _build_sam2_prompts(
            frames_dir / frame_files[anchor_idx], raw_box, pad_bottom, is_box=is_box
        )

    from sam2.build_sam import build_sam2_video_predictor

    # SAM2 video predictor doi hoi ten file la so nguyen (0.jpg, 1.jpg, ...) -> tao thu muc tam
    vframes_dir = sdir / "_sam2_frames"
    if vframes_dir.exists():
        shutil.rmtree(vframes_dir)
    vframes_dir.mkdir()
    for new_i, name in enumerate(frame_files):
        shutil.copy(frames_dir / name, vframes_dir / f"{new_i}.jpg")

    predictor = build_sam2_video_predictor(m["sam2_config"], m["sam2_checkpoint"], device=DEVICE)
    state = predictor.init_state(video_path=str(vframes_dir))

    # Dang ky tat ca Keyframe Focus Boxes (Whole-Object Box Mode) de giu tron ven cac chi tiet cung mau
    # (nhu quai xach den trung mau man hinh phia sau) va khong bi troi mask khi xoay 6 mat
    for k_idx, k_box in sorted(keyframe_boxes.items()):
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=k_idx,
            obj_id=1,
            box=np.array(k_box, dtype=np.float32),
            points=pts_prompt if k_idx == anchor_idx else None,
            labels=lbl_prompt if k_idx == anchor_idx else None,
        )

    masks_dir = sdir / "masks"
    masks_dir.mkdir(exist_ok=True)
    for old in masks_dir.glob("frame_*.png"):
        old.unlink()

    cutouts_dir = sdir / "cutouts"
    cutouts_dir.mkdir(exist_ok=True)
    for old in cutouts_dir.glob("frame_*.jpg"):
        old.unlink()

    dilate_px = m.get("dilate_px", 0)
    kernel_dilate = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8) if dilate_px > 0 else None
    # Kernel 3x3 dong lo kim do loa sang ma van giu nguyen khe ho kien truc (nhu lo luon tay quai xach)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _save(frame_idx: int, mask_logits: torch.Tensor) -> None:
        raw = (mask_logits[0] > 0.0).cpu().numpy()  # obj duy nhat -> [0]; shape (1, H, W)
        mask = raw[0].astype(np.uint8) * 255
        # Giu thanh phan chinh va cac chi tiet cung vat the (quai xach, chot nap >= 1.5% dien tich cum chinh)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_lbl = 1 + int(np.argmax(areas))
            max_area = float(areas[largest_lbl - 1])
            lx = stats[largest_lbl, cv2.CC_STAT_LEFT]
            ly = stats[largest_lbl, cv2.CC_STAT_TOP]
            lw = stats[largest_lbl, cv2.CC_STAT_WIDTH]
            lh = stats[largest_lbl, cv2.CC_STAT_HEIGHT]
            keep_lbls = [largest_lbl]
            for lbl in range(1, num_labels):
                if lbl == largest_lbl:
                    continue
                area_c = float(stats[lbl, cv2.CC_STAT_AREA])
                cx_c, cy_c = centroids[lbl]
                # Giu cac chi tiet phu (nhu quai xach/chot) nam gan/trong pham vi vat the chinh
                if area_c >= 0.015 * max_area and (lx - 0.1 * lw <= cx_c <= lx + 1.1 * lw) and (ly - 0.35 * lh <= cy_c <= ly + 1.1 * lh):
                    keep_lbls.append(lbl)
            mask = np.where(np.isin(labels_im, keep_lbls), 255, 0).astype(np.uint8)

        # Lap day lo hong ben trong vat the (Hole filling: bao ve tem trang, nhan mac, vung loa sang)
        hm, wm = mask.shape
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        fill_mask = np.zeros((hm + 4, wm + 4), np.uint8)
        inv = cv2.bitwise_not(padded)
        cv2.floodFill(inv, fill_mask, (0, 0), 0)
        holes = inv[1:-1, 1:-1]
        mask = cv2.bitwise_or(mask, holes)

        # Lap lom day ho / cua vom o mep duoi (Bottom arch filling: bao ve tem dan trang sat day khoi bi khoet lo)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw >= 30 and bh >= 30:
                c_pts = c.reshape(-1, 2)
                left_pts = c_pts[(c_pts[:, 0] >= bx) & (c_pts[:, 0] <= bx + 0.25 * bw)]
                right_pts = c_pts[(c_pts[:, 0] >= bx + 0.75 * bw) & (c_pts[:, 0] <= bx + bw)]
                if len(left_pts) > 0 and len(right_pts) > 0:
                    p_left = left_pts[np.argmax(left_pts[:, 1])]
                    p_right = right_pts[np.argmax(right_pts[:, 1])]
                    bridge_mask = mask.copy()
                    cv2.line(bridge_mask, tuple(p_left), tuple(p_right), 255, thickness=max(3, int(0.015 * bh)))
                    padded_b = cv2.copyMakeBorder(bridge_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                    fill_mask_b = np.zeros((hm + 4, wm + 4), np.uint8)
                    inv_b = cv2.bitwise_not(padded_b)
                    cv2.floodFill(inv_b, fill_mask_b, (0, 0), 0)
                    trapped_holes = inv_b[1:-1, 1:-1]
                    if int(np.sum(trapped_holes > 0)) >= 0.01 * (bw * bh):
                        mask = cv2.bitwise_or(bridge_mask, trapped_holes)

        # Bit lo kim nho ben trong vat the do bong sang
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        if kernel_dilate is not None:
            mask = cv2.dilate(mask, kernel_dilate)
        cv2.imwrite(str(masks_dir / frame_files[frame_idx].replace(".jpg", ".png")), mask)

        # Luu anh Focus Cutout (cat tach ro hinh dang vat the, loai bo hoan toan nen)
        orig_img = cv2.imread(str(frames_dir / frame_files[frame_idx]))
        if orig_img is not None:
            cutout = cv2.bitwise_and(orig_img, orig_img, mask=mask)
            cv2.imwrite(str(cutouts_dir / frame_files[frame_idx]), cutout, [cv2.IMWRITE_JPEG_QUALITY, 95])

    first_anchor = min(keyframe_boxes.keys())
    # Lan truyen xuoi tu keyframe dau tien den cuoi video
    for frame_idx, _obj_ids, mask_logits in tqdm(
        predictor.propagate_in_video(state, start_frame_idx=first_anchor, reverse=False),
        desc="Buoc 2b: lan truyen xuoi",
    ):
        _save(frame_idx, mask_logits)
    # Neu keyframe dau tien > 0, lan truyen nguoc ve dau video
    if first_anchor > 0:
        for frame_idx, _obj_ids, mask_logits in tqdm(
            predictor.propagate_in_video(state, start_frame_idx=first_anchor, reverse=True),
            desc="Buoc 2c: lan truyen nguoc",
        ):
            _save(frame_idx, mask_logits)

    shutil.rmtree(vframes_dir, ignore_errors=True)
    print(f"[Buoc 2] Da tao {len(frame_files)} Focus Mask -> {masks_dir} & Cutouts -> {cutouts_dir}")