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
def _is_mobile_source(camera) -> bool:
    return isinstance(camera, str) and camera.strip().lower() in ("mobile", "phone", "qr")


def _is_mobile_cap(cap) -> bool:
    return getattr(cap, "is_mobile", False) or cap.__class__.__name__ == "MobileVideoCapture"


def _open_camera(camera, width: int, height: int, fps: int, rec_cfg: dict | None = None):
    if _is_mobile_source(camera):
        try:
            from .mobile_camera import MobileVideoCapture
        except ImportError as exc:
            print(f"Khong the mo Camera dien thoai do thieu thu vien ({exc}).")
            print("Goi y: chay 'poetry install' hoac 'pip install flask simple-websocket cryptography psutil' de dung tinh nang nay.")
            return cv2.VideoCapture()
        rec_cfg = rec_cfg or {}
        return MobileVideoCapture(
            width=width,
            height=height,
            fps=fps,
            port=int(rec_cfg.get("mobile_port", 8443)),
            host_ip=rec_cfg.get("mobile_host_ip") or None,
            use_ssl=bool(rec_cfg.get("mobile_ssl", True)),
        )

    if isinstance(camera, str) and camera.isdigit():
        camera = int(camera)
    # CAP_DSHOW on dinh nhat tren Windows
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW) if isinstance(camera, int) else cv2.VideoCapture(camera)
    if not cap.isOpened() and isinstance(camera, int):
        cap = cv2.VideoCapture(camera)

    # TRIET TIEU DO TRE (ZERO LATENCY):
    # 1. Ep FourCC MJPG truoc de camera USB truyen nen phan cung (dat 30-60 FPS)
    if isinstance(camera, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # 2. Buffer size = 1: Chi giu frame moi nhat, tranh ton dong hang doi gay lag
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _read_cam_props(cap, default_fps: float) -> tuple[int, int, float]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or default_fps
    if fps < 1 or fps > 120:
        fps = default_fps
    return w, h, fps


def _finalize_recorded_video(
    temp_video_path: Path,
    video_path: Path,
    recorded_frames: int,
    elapsed: float,
    nominal_fps: float,
    size: tuple[int, int],
) -> float:
    """Dong bo lai FPS thuc te neu do tre lam lech thoi luong video."""
    actual_fps = recorded_frames / elapsed if elapsed > 0 and recorded_frames > 0 else nominal_fps
    print(f"Da dung quay: {recorded_frames} khung hinh trong {elapsed:.1f}s -> FPS thuc te: {actual_fps:.2f}")
    if abs(actual_fps - nominal_fps) > 1.5 and actual_fps >= 1.0 and temp_video_path.exists():
        print(f"Dong bo lai video theo FPS thuc te ({actual_fps:.2f} FPS) de khong bi tua nhanh...")
        rcap = cv2.VideoCapture(str(temp_video_path))
        rwriter = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, size)
        while True:
            rok, rframe = rcap.read()
            if not rok:
                break
            rwriter.write(rframe)
        rcap.release()
        rwriter.release()
        try:
            temp_video_path.unlink()
        except Exception:
            pass
    elif temp_video_path.exists():
        if video_path.exists():
            video_path.unlink()
        temp_video_path.rename(video_path)
    return actual_fps


def _unique_name(name: str | None) -> str:
    if not name:
        return "box_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    base, n = name, 1
    while paths.sample_dir(name).exists():
        n += 1
        name = f"{base}_{n}"
    return name


def record_video(name: str | None, cfg: dict) -> str | None:
    """Quay tu camera (Webcam USB hoac Camera dien thoai qua QR/Wi-Fi). Phim:
      - R: bat dau / dung quay
      - M: chuyen sang Camera dien thoai (quet QR)
      - C hoac 0, 1, 2, 3: doi camera truc tiep tren man hinh
      - P: doi 1080p / 720p
      - Q hoac ESC: thoat
    Tra ve ten mau da luu.
    """
    rec = cfg["record"]
    current_cam = rec["camera"]
    cap = _open_camera(current_cam, rec["width"], rec["height"], rec["fps"], rec_cfg=rec)
    if not cap.isOpened() and not _is_mobile_source(current_cam):
        print(f"Khong mo duoc camera {current_cam}. Thu doi sang camera khac...")
        for fb_cam in [0, 1, 2]:
            if fb_cam != current_cam:
                cap = _open_camera(fb_cam, rec["width"], rec["height"], rec["fps"], rec_cfg=rec)
                if cap.isOpened():
                    current_cam = fb_cam
                    print(f"Da tu dong chuyen sang Camera {current_cam}.")
                    break
    if not cap.isOpened():
        print("Khong tim thay camera nao kha dung. Kiem tra record.camera hoac thu --camera mobile.")
        return None

    w, h, fps = _read_cam_props(cap, rec["fps"])

    name = _unique_name(name)
    sdir = paths.sample_dir(name)
    video_path = sdir / cfg["video_file"]

    writer, t0, saved = None, 0.0, False
    recorded_frames = 0
    temp_video_path = sdir / "temp_record.mp4"
    actual_fps = fps
    last_mobile_frame = None

    def _print_banner() -> None:
        if _is_mobile_cap(cap):
            cap.print_connection_info()
        else:
            print(f"Camera [{current_cam}] {w}x{h} @ {fps:.0f}fps.")
        print("  Phim: [R] Bat dau/dung quay  |  [M] Camera dien thoai (QR)  |  [C/0-3] Doi camera  |  [P] 1080p/720p  |  [Q/ESC] Thoat")

    def _switch_source(target_cam) -> None:
        nonlocal cap, current_cam, w, h, fps, last_mobile_frame
        print(f"Dang chuyen sang Camera [{target_cam}]...")
        cap.release()
        time.sleep(0.2)
        new_cap = _open_camera(target_cam, rec["width"], rec["height"], rec["fps"], rec_cfg=rec)
        if new_cap.isOpened():
            cap = new_cap
            current_cam = target_cam
            last_mobile_frame = None
            w, h, fps = _read_cam_props(cap, rec["fps"])
            _print_banner()
        else:
            print(f"-> Camera [{target_cam}] khong kha dung. Khoi phuc lai Camera [{current_cam}]...")
            cap = _open_camera(current_cam, w, h, rec["fps"], rec_cfg=rec)

    _print_banner()

    while True:
        is_mobile = _is_mobile_cap(cap)
        new_frame_arrived = False
        if is_mobile:
            if cap.pop_command() == "toggle_record":
                key_trigger_record = True
            else:
                key_trigger_record = False
            ok, frame = cap.read(timeout=0.04)
            if not ok:
                if not cap.has_frame or last_mobile_frame is None:
                    cv2.imshow("nckh_2 - record", cap.get_qr_card())
                    frame_ready = False
                else:
                    frame = last_mobile_frame
                    frame_ready = True
            else:
                last_mobile_frame = frame
                frame_ready = True
                new_frame_arrived = True
                # Dong bo kich thuoc thuc te khi dien thoai xoay ngang/doc luc chua quay
                fh, fw = frame.shape[:2]
                if writer is None and (fw != w or fh != h):
                    w, h = fw, fh
        else:
            key_trigger_record = False
            ok, frame = cap.read()
            if not ok:
                print("Mat tin hieu camera.")
                break
            frame_ready = True
            new_frame_arrived = True

        if frame_ready:
            if writer is not None and new_frame_arrived:
                write_frame = frame
                if frame.shape[1] != w or frame.shape[0] != h:
                    write_frame = cv2.resize(frame, (w, h))
                writer.write(write_frame)
                recorded_frames += 1

            view = frame.copy()
            if writer is not None:
                elapsed = time.time() - t0
                cur_fps = recorded_frames / elapsed if elapsed > 0.5 else fps
                cv2.circle(view, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(view, f"REC [{current_cam}] {elapsed:5.1f}s | {recorded_frames} frames | {cur_fps:4.1f} FPS",
                            (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if cur_fps < 20 and elapsed > 2.0:
                    cv2.putText(view, "CANH BAO: FPS thap (<20). Hay bat den sang hon hoac giam ve 720p!",
                                (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
            else:
                hint = f"Cam [{current_cam}] {w}x{h} | R: Record | M: Mobile QR | C: Switch Cam | Q: Quit"
                cv2.putText(view, hint, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            scale = min(960 / max(w, 1), 540 / max(h, 1))
            disp_w = max(180, int(w * scale))
            disp_h = max(180, int(h * scale))
            cv2.imshow("nckh_2 - record", cv2.resize(view, (disp_w, disp_h)))

        key = cv2.waitKey(1) & 0xFF
        if key_trigger_record or key in (ord("r"), ord("R")):
            if is_mobile and not cap.has_frame:
                print("Hay quet ma QR tren dien thoai va cho hinh anh hien thi truoc khi quay.")
            elif writer is None:
                sdir.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(temp_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                t0 = time.time()
                recorded_frames = 0
                if is_mobile:
                    cap.set_recording_state(True)
                print("Bat dau quay...")
            else:
                writer.release()
                writer, saved = None, True
                if is_mobile:
                    cap.set_recording_state(False)
                actual_fps = _finalize_recorded_video(
                    temp_video_path, video_path, recorded_frames, time.time() - t0, fps, (w, h)
                )
                break

        elif writer is None and not is_mobile and key in (ord("p"), ord("P")):
            new_w, new_h = (1280, 720) if w >= 1920 else (1920, 1080)
            print(f"Dang doi do phan giai sang {new_w}x{new_h}...")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_h)
            cur_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cur_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if cur_w == new_w and cur_h == new_h:
                w, h = cur_w, cur_h
                print(f"-> Da doi sang {w}x{h} @ {fps:.0f}fps")
            else:
                cap.release()
                time.sleep(0.2)
                new_cap = _open_camera(current_cam, new_w, new_h, rec["fps"], rec_cfg=rec)
                if new_cap.isOpened():
                    cap = new_cap
                    w, h, fps = _read_cam_props(cap, rec["fps"])
                    print(f"-> Da doi sang {w}x{h} @ {fps:.0f}fps")
                else:
                    print("Khong doi duoc do phan giai, khoi phuc lai...")
                    cap = _open_camera(current_cam, w, h, rec["fps"], rec_cfg=rec)

        elif writer is None and key in (ord("m"), ord("M")):
            if not is_mobile:
                _switch_source("mobile")

        elif writer is None and (key in (ord("c"), ord("C")) or (ord("0") <= key <= ord("9"))):
            if ord("0") <= key <= ord("9"):
                target_cam = key - ord("0")
            else:
                try:
                    c_idx = int(current_cam) if str(current_cam).isdigit() else -1
                    target_cam = (c_idx + 1) % 4
                except Exception:
                    target_cam = 0
            if target_cam != current_cam:
                _switch_source(target_cam)

        elif key in (ord("q"), ord("Q"), 27):
            break

    if writer is not None:  # thoat khi dang quay -> van luu va dong bo FPS
        writer.release()
        if _is_mobile_cap(cap):
            cap.set_recording_state(False)
        actual_fps = _finalize_recorded_video(
            temp_video_path, video_path, recorded_frames, time.time() - t0, fps, (w, h)
        )
        saved = True
    cap.release()
    cv2.destroyAllWindows()

    if not saved:
        return None
    meta = {
        "source": "mobile" if _is_mobile_source(current_cam) else "recorded",
        "camera": str(current_cam),
        "width": w,
        "height": h,
        "fps": round(actual_fps, 2),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (sdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Da luu mau: data/samples/{name}/{cfg['video_file']}")
    return name

