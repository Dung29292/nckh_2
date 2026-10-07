"""Nguon dau vao: (1) quay video truc tiep -> luu thanh mau moi, hoac (2) chon mau co san."""
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from . import paths


# ---------------------------------------------------------------- mau co san
def list_samples(video_file: str) -> list[str]:
    if not paths.SAMPLES_DIR.exists():
        return []
    return sorted(
        d.name for d in paths.SAMPLES_DIR.iterdir() if d.is_dir() and (d / video_file).exists()
    )


def choose_sample(video_file: str) -> str | None:
    """Hien danh sach mau va cho nguoi dung chon bang so."""
    samples = list_samples(video_file)
    if not samples:
        print(f"Chua co mau nao. Hay dat video vao data/samples/<ten_mau>/{video_file} hoac chon che do quay.")
        return None
    print("\nCac mau co san:")
    for i, name in enumerate(samples, 1):
        print(f"  [{i}] {name}")
    while True:
        s = input("Chon mau (so, Enter de huy): ").strip()
        if not s:
            return None
        if s.isdigit() and 1 <= int(s) <= len(samples):
            return samples[int(s) - 1]
        print("Lua chon khong hop le.")


# ---------------------------------------------------------------- quay video
def _open_camera(camera, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if isinstance(camera, str) and camera.isdigit():
        camera = int(camera)
    # CAP_DSHOW on dinh hon tren Windows khi la webcam; URL thi dung mac dinh
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW) if isinstance(camera, int) else cv2.VideoCapture(camera)
    if isinstance(camera, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def _unique_name(name: str | None) -> str:
    if not name:
        return "box_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    base, n = name, 1
    while paths.sample_dir(name).exists():
        n += 1
        name = f"{base}_{n}"
    return name


def record_video(name: str | None, cfg: dict) -> str | None:
    """Quay tu camera. Phim: R = bat dau/dung quay, Q hoac ESC = thoat. Tra ve ten mau da luu."""
    rec = cfg["record"]
    cap = _open_camera(rec["camera"], rec["width"], rec["height"], rec["fps"])
    if not cap.isOpened():
        print("Khong mo duoc camera. Kiem tra record.camera trong configs/default.yaml.")
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or rec["fps"]
    if fps < 1 or fps > 120:
        fps = rec["fps"]

    name = _unique_name(name)
    sdir = paths.sample_dir(name)
    video_path = sdir / cfg["video_file"]

    writer, t0, saved = None, 0.0, False
    print(f"Camera {w}x{h} @ {fps:.0f}fps. Nhan R de bat dau/dung quay, Q de thoat.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Mat tin hieu camera.")
            break
        if writer is not None:
            writer.write(frame)

        view = frame.copy()
        if writer is not None:
            cv2.circle(view, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(view, f"REC {time.time() - t0:5.1f}s", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(view, "R: start/stop   Q: quit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("nckh_2 - record", cv2.resize(view, (960, int(960 * h / w))))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("r"), ord("R")):
            if writer is None:
                sdir.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                t0 = time.time()
                print("Bat dau quay...")
            else:
                writer.release()
                writer, saved = None, True
                print("Da dung quay.")
                break
        elif key in (ord("q"), ord("Q"), 27):
            break

    if writer is not None:  # thoat khi dang quay -> van giu file
        writer.release()
        saved = True
    cap.release()
    cv2.destroyAllWindows()

    if not saved:
        return None
    meta = {"source": "recorded", "camera": str(rec["camera"]), "width": w, "height": h,
            "fps": fps, "created": datetime.now().isoformat(timespec="seconds")}
    (sdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Da luu mau: data/samples/{name}/{cfg['video_file']}")
    return name

