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


def _unique_name(name: str | None) -> str:
    if not name:
        return "box_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    base, n = name, 1
    while paths.sample_dir(name).exists():
        n += 1
        name = f"{base}_{n}"
    return name


def record_video(name: str | None, cfg: dict) -> str | None:
    """Quay tu camera. Phim:
      - R: bat dau / dung quay
      - C hoac 0, 1, 2, 3: doi camera truc tiep tren man hinh
      - Q hoac ESC: thoat
    Tra ve ten mau da luu.
    """
    rec = cfg["record"]
    current_cam = rec["camera"]
    cap = _open_camera(current_cam, rec["width"], rec["height"], rec["fps"])
    if not cap.isOpened():
        print(f"Khong mo duoc camera {current_cam}. Thu doi sang camera khac...")
        # Fallback thu camera 0 neu cam 1 loi
        for fb_cam in [0, 1, 2]:
            if fb_cam != current_cam:
                cap = _open_camera(fb_cam, rec["width"], rec["height"], rec["fps"])
                if cap.isOpened():
                    current_cam = fb_cam
                    print(f"Da tu dong chuyen sang Camera {current_cam}.")
                    break
        if not cap.isOpened():
            print("Khong tim thay camera nao kha dung. Kiem tra record.camera trong configs/default.yaml.")
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
    recorded_frames = 0
    temp_video_path = sdir / "temp_record.mp4"
    actual_fps = fps

    print(f"Camera [{current_cam}] {w}x{h} @ {fps:.0f}fps.")
    print("  Phim: [R] Bat dau/dung quay  |  [C] hoac [0-3] Doi camera  |  [P] Doi 1080p/720p  |  [Q/ESC] Thoat")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Mat tin hieu camera.")
            break
        if writer is not None:
            writer.write(frame)
            recorded_frames += 1

        view = frame.copy()
        if writer is not None:
            elapsed = time.time() - t0
            cur_fps = recorded_frames / elapsed if elapsed > 0.5 else fps
            cv2.circle(view, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(view, f"REC [{current_cam}] {elapsed:5.1f}s | {recorded_frames} frames | {cur_fps:4.1f} FPS",
                        (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if cur_fps < 20 and elapsed > 2.0:
                cv2.putText(view, "CANH BAO: FPS thap (<20). Hay bat den sang hon hoac chuyen sang 720p (phim P)!",
                            (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        else:
            cv2.putText(view, f"Cam [{current_cam}] {w}x{h} | R: Record | P: 1080p/720p | C: Switch Cam | Q: Quit",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("nckh_2 - record", cv2.resize(view, (960, int(960 * h / w))))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("r"), ord("R")):
            if writer is None:
                sdir.mkdir(parents=True, exist_ok=True)
                # Ghi vao file tam thoi truoc de do FPS thuc te
                writer = cv2.VideoWriter(str(temp_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                t0 = time.time()
                recorded_frames = 0
                print("Bat dau quay...")
            else:
                writer.release()
                writer, saved = None, True
                elapsed = time.time() - t0
                actual_fps = recorded_frames / elapsed if elapsed > 0 else fps
                print(f"Da dung quay: {recorded_frames} khung hinh trong {elapsed:.1f}s -> FPS thuc te: {actual_fps:.2f}")

                # Neu FPS thuc te lech voi FPS danh nghia, dong bo lai file video de thoi luong khong bi tua nhanh
                if abs(actual_fps - fps) > 1.5 and actual_fps >= 1.0 and temp_video_path.exists():
                    print(f"Dong bo lai video theo FPS thuc te ({actual_fps:.2f} FPS) de khong bi tua nhanh...")
                    rcap = cv2.VideoCapture(str(temp_video_path))
                    rwriter = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, (w, h))
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
                else:
                    if temp_video_path.exists():
                        if video_path.exists():
                            video_path.unlink()
                        temp_video_path.rename(video_path)
                break

        elif writer is None and key in (ord("p"), ord("P")):
            # Chuyen doi qua lai giua 1080p va 720p de giam tai/chong giat lag
            new_w, new_h = (1280, 720) if w >= 1920 else (1920, 1080)
            print(f"Dang doi do phan giai sang {new_w}x{new_h}...")
            # Thu doi truc tiep tren luong cap hien tai
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_h)
            cur_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cur_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if cur_w == new_w and cur_h == new_h:
                w, h = cur_w, cur_h
                print(f"-> Da doi sang {w}x{h} @ {fps:.0f}fps")
            else:
                # Neu driver DirectShow yeu cau mo lai: phai release() luong cu truoc de giai phong phan cung
                cap.release()
                time.sleep(0.2)
                new_cap = _open_camera(current_cam, new_w, new_h, rec["fps"])
                if new_cap.isOpened():
                    cap = new_cap
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or rec["fps"]
                    print(f"-> Da doi sang {w}x{h} @ {fps:.0f}fps")
                else:
                    print("Khong doi duoc do phan giai, khoi phuc lai...")
                    cap = _open_camera(current_cam, w, h, rec["fps"])

        elif writer is None and (key in (ord("c"), ord("C")) or (ord("0") <= key <= ord("9"))):
            # Doi camera khi chua bam quay
            if ord("0") <= key <= ord("9"):
                target_cam = key - ord("0")
            else:
                try:
                    c_idx = int(current_cam) if str(current_cam).isdigit() else 0
                    target_cam = (c_idx + 1) % 4
                except Exception:
                    target_cam = 0

            if target_cam != current_cam:
                print(f"Dang thu chuyen sang Camera [{target_cam}]...")
                cap.release()
                time.sleep(0.2)
                new_cap = _open_camera(target_cam, w, h, rec["fps"])
                if new_cap.isOpened():
                    cap = new_cap
                    current_cam = target_cam
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or rec["fps"]
                    if fps < 1 or fps > 120:
                        fps = rec["fps"]
                    print(f"-> Da chuyen thanh cong sang Camera [{current_cam}] ({w}x{h} @ {fps:.0f}fps)")
                else:
                    print(f"-> Camera [{target_cam}] khong kha dung. Khoi phuc lai Camera [{current_cam}]...")
                    cap = _open_camera(current_cam, w, h, rec["fps"])

        elif key in (ord("q"), ord("Q"), 27):
            break

    if writer is not None:  # thoat khi dang quay -> van giu file
        writer.release()
        saved = True
    cap.release()
    cv2.destroyAllWindows()

    if not saved:
        return None
    meta = {"source": "recorded", "camera": str(current_cam), "width": w, "height": h,
            "fps": round(actual_fps, 2) if "actual_fps" in locals() else fps, "created": datetime.now().isoformat(timespec="seconds")}
    (sdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Da luu mau: data/samples/{name}/{cfg['video_file']}")
    return name

