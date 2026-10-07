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
    """Chay Grounding DINO tren vai frame rai deu, lay bbox co diem cao nhat."""
    cache = paths.MODELS_DIR / "grounding_dino"
    processor = AutoProcessor.from_pretrained(cfg_m["dino_model_id"], cache_dir=cache)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        cfg_m["dino_model_id"], cache_dir=cache
    ).to(DEVICE)
    model.eval()

    n = len(frame_files)
    k = min(cfg_m["detect_frames"], n)
    idxs = np.linspace(0, n - 1, k).round().astype(int)

    prompt = cfg_m["prompt"].strip()
    if not prompt.endswith("."):
        prompt += "."  # Grounding DINO can dau cham cuoi cau prompt

    best = None  # (score, frame_idx, box_xyxy)
    for i in tqdm(idxs, desc="Buoc 2a: tim bbox"):
        img = Image.open(frames_dir / frame_files[i]).convert("RGB")
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        try:
            result = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=cfg_m["box_threshold"],
                text_threshold=cfg_m["text_threshold"],
                target_sizes=[img.size[::-1]],
            )[0]
        except TypeError:
            # transformers ban moi: doi ten box_threshold -> threshold, input_ids khong can truyen rieng
            result = processor.post_process_grounded_object_detection(
                outputs,
                threshold=cfg_m["box_threshold"],
                text_threshold=cfg_m["text_threshold"],
                target_sizes=[img.size[::-1]],
            )[0]
        if len(result["scores"]) == 0:
            continue
        j = int(result["scores"].argmax())
        score = float(result["scores"][j])
        if best is None or score > best[0]:
            best = (score, int(i), result["boxes"][j].tolist())

    del model, processor
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError(
            f'Khong tim thay "{cfg_m["prompt"]}" trong {k} frame kiem tra. Thu giam box_threshold trong config.'
        )
    score, frame_idx, box = best
    print(f"[Buoc 2] Bbox tot nhat: {frame_files[frame_idx]}, score={score:.2f}, box={[round(v, 1) for v in box]}")
    return frame_idx, box


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
    kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8) if dilate_px > 0 else None

    def _save(frame_idx: int, mask_logits: torch.Tensor) -> None:
        raw = (mask_logits[0] > 0.0).cpu().numpy()  # obj duy nhat -> [0]; shape (1, H, W)
        mask = raw[0].astype(np.uint8) * 255
        if kernel is not None:
            mask = cv2.dilate(mask, kernel)
        cv2.imwrite(str(masks_dir / frame_files[frame_idx].replace(".jpg", ".png")), mask)

    # lan truyen xuoi tu anchor den cuoi video, roi lan truyen nguoc tu anchor ve dau
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