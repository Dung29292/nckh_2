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
    """Chay Grounding DINO tim bounding box vat the tren tap frame mau.
    Ho tro thu nhieu prompt (prompt chinh + fallback) va mo rong le bounding box
    de tranh bi cat mat cac vien cong cua vat the phi phang."""
    cache = paths.MODELS_DIR / "grounding_dino"
    processor = AutoProcessor.from_pretrained(cfg_m["dino_model_id"], cache_dir=cache)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        cfg_m["dino_model_id"], cache_dir=cache
    ).to(DEVICE)
    model.eval()

    n = len(frame_files)
    k = min(max(cfg_m.get("detect_frames", 5), 8), n)
    idxs = np.linspace(0, n - 1, k).round().astype(int)

    # Danh sach prompt de thu: prompt nguoi dung chi dinh -> prompt tong quat
    raw_prompt = cfg_m.get("prompt", "object").strip().lower()
    prompts_to_try = [raw_prompt]
    for alt in ["object", "item", "product", "toy"]:
        if alt not in raw_prompt and alt not in prompts_to_try:
            prompts_to_try.append(alt)

    box_thresh = float(cfg_m.get("box_threshold", 0.3))
    text_thresh = float(cfg_m.get("text_threshold", 0.25))
    pad_frac = float(cfg_m.get("box_pad_frac", 0.05))

    best = None  # (score, frame_idx, box_xyxy, prompt_used, img_size)

    for p_try in prompts_to_try:
        p_query = p_try if p_try.endswith(".") else f"{p_try}."
        print(f"[Buoc 2] Thu nhan dien voi prompt: '{p_try}'...")

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
            
            # Chon box tot nhat o frame nay
            for score_t, box_t in zip(result["scores"], result["boxes"]):
                s = float(score_t)
                bx = box_t.tolist()
                bw = bx[2] - bx[0]
                bh = bx[3] - bx[1]
                # Bo qua box qua nho (< 1% dien tich) hoac qua lon sat vien toan bo khung hinh
                if bw * bh < 0.01 * (w_img * h_img):
                    continue
                # Tinh khoang cach toi tam anh (uu tien vat the o trung tam)
                cx = (bx[0] + bx[2]) / 2.0
                cy = (bx[1] + bx[3]) / 2.0
                dist_center = np.hypot(cx - w_img / 2.0, cy - h_img / 2.0) / (np.hypot(w_img, h_img) / 2.0)
                # Diem danh gia ket hop confidence va do trung tam
                adj_score = s * (1.0 - 0.2 * dist_center)

                if best is None or adj_score > best[0]:
                    best = (adj_score, int(i), bx, p_try, (w_img, h_img))

        if best is not None:
            break

    del model, processor
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError(
            f'Khong tim thay vat the voi cac prompt {prompts_to_try} trong {k} frame kiem tra. '
            'Thu truyen --prompt "<ten_vat_the>" hoac giam box_threshold trong configs/default.yaml.'
        )

    score, frame_idx, raw_box, prompt_used, (w_img, h_img) = best

    # Mo rong nhe bounding box (padding) de bao toan cac vien cong, tai, goc vat the
    bw = raw_box[2] - raw_box[0]
    bh = raw_box[3] - raw_box[1]
    padded_box = [
        max(0.0, raw_box[0] - pad_frac * bw),
        max(0.0, raw_box[1] - pad_frac * bh),
        min(float(w_img), raw_box[2] + pad_frac * bw),
        min(float(h_img), raw_box[3] + pad_frac * bh),
    ]

    print(
        f"[Buoc 2] Nhan dien thanh cong '{prompt_used}' tren {frame_files[frame_idx]}: "
        f"score={score:.2f}, box={[round(v, 1) for v in padded_box]} (pad={pad_frac:.0%})"
    )
    return frame_idx, padded_box


def run(sample_name: str, cfg: dict) -> None:
    m = cfg["masking"]
    sdir = paths.sample_dir(sample_name)
    frames_dir = sdir / "frames_sharp"
    frame_files = sorted(p.name for p in frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"Khong co frame trong {frames_dir}, chay Buoc 1 truoc.")

    anchor_idx, box = _detect_box(frames_dir, frame_files, m)

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
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=anchor_idx,
        obj_id=1,
        box=np.array(box, dtype=np.float32),
    )

    masks_dir = sdir / "masks"
    masks_dir.mkdir(exist_ok=True)
    for old in masks_dir.glob("frame_*.png"):
        old.unlink()

    dilate_px = m.get("dilate_px", 0)
    kernel_dilate = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8) if dilate_px > 0 else None
    # Kernel dong lo nho do loa sang/phan chieu tren be mat cong
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _save(frame_idx: int, mask_logits: torch.Tensor) -> None:
        raw = (mask_logits[0] > 0.0).cpu().numpy()  # obj duy nhat -> [0]; shape (1, H, W)
        mask = raw[0].astype(np.uint8) * 255
        # Bit lo nho ben trong vat the do bong sang
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        if kernel_dilate is not None:
            mask = cv2.dilate(mask, kernel_dilate)
        cv2.imwrite(str(masks_dir / frame_files[frame_idx].replace(".jpg", ".png")), mask)

    # Lan truyen xuoi tu anchor den cuoi video, roi lan truyen nguoc tu anchor ve dau
    for frame_idx, _obj_ids, mask_logits in tqdm(
        predictor.propagate_in_video(state, start_frame_idx=anchor_idx, reverse=False),
        desc="Buoc 2b: lan truyen xuoi",
    ):
        _save(frame_idx, mask_logits)
    for frame_idx, _obj_ids, mask_logits in tqdm(
        predictor.propagate_in_video(state, start_frame_idx=anchor_idx, reverse=True),
        desc="Buoc 2c: lan truyen nguoc",
    ):
        _save(frame_idx, mask_logits)

    shutil.rmtree(vframes_dir, ignore_errors=True)
    print(f"[Buoc 2] Da tao {len(frame_files)} mask -> {masks_dir}")