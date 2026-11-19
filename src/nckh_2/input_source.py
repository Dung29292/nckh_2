"""Nguon dau vao: (1) quay video truc tiep -> luu thanh mau moi, hoac (2) chon mau co san."""
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

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

    # Bo nho dem giu phien Mobile Camera song lien tuc (khong mat ket noi khi doi qua lai USB Cam)
    mobile_cap = cap if _is_mobile_cap(cap) else None

    win_name = "nckh_2 - record"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1040, 660)

    show_grid = True
    clicked_action: str | None = None
    button_hitboxes: dict[str, tuple[int, int, int, int]] = {}

    def _on_mouse(event, x, y, _flags, _param):
        nonlocal clicked_action
        if event == cv2.EVENT_LBUTTONUP:
            for act_name, (bx, by, bw, bh) in button_hitboxes.items():
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    clicked_action = act_name
                    break

    cv2.setMouseCallback(win_name, _on_mouse)

    def _print_banner() -> None:
        if _is_mobile_cap(cap):
            cap.print_connection_info()
        else:
            print(f"Camera [{current_cam}] {w}x{h} @ {fps:.0f}fps.")
        has_mob = " (San sang)" if (mobile_cap is not None and mobile_cap.isOpened()) else ""
        print(f"  Phim: [R] Quay/Dung  |  [M] Mobile QR{has_mob}  |  [X] Ngat Mobile  |  [C/0-3] Doi Cam  |  [G] Luoi 3D  |  [Q/ESC] Thoat")

    def _disconnect_mobile() -> None:
        nonlocal mobile_cap, cap, current_cam, w, h, fps
        if mobile_cap is not None or _is_mobile_cap(cap):
            print("\nDa chu dong ngat ket noi Camera dien thoai.")
            if mobile_cap is not None:
                try:
                    mobile_cap.release()
                except Exception:
                    pass
                mobile_cap = None
            if _is_mobile_cap(cap):
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                current_cam = 0
                _switch_source(0)
            else:
                _print_banner()

    def _switch_source(target_cam) -> None:
        nonlocal cap, current_cam, w, h, fps, last_mobile_frame, mobile_cap
        target_is_mobile = _is_mobile_source(target_cam)
        cur_is_mobile = _is_mobile_cap(cap)

        if target_is_mobile:
            if cur_is_mobile:
                return
            if cap is not None and not cur_is_mobile:
                cap.release()
            if mobile_cap is not None and mobile_cap.isOpened():
                print("Chuyen sang Camera dien thoai (tai su dung phien ket noi da co, khong can quet lai QR)...")
                cap = mobile_cap
                current_cam = "mobile"
                w, h, fps = _read_cam_props(cap, rec["fps"])
                _print_banner()
                return

            print("Khoi dong ket noi Camera dien thoai qua QR...")
            new_cap = _open_camera("mobile", rec["width"], rec["height"], rec["fps"], rec_cfg=rec)
            if new_cap.isOpened():
                mobile_cap = new_cap
                cap = new_cap
                current_cam = "mobile"
                last_mobile_frame = None
                w, h, fps = _read_cam_props(cap, rec["fps"])
                _print_banner()
            return

        # Chuyen sang USB Cam (0, 1, 2...)
        if cur_is_mobile:
            # GIU NGUYEN phien mobile_cap chay ngam, khong release de dien thoai giu WebSocket!
            mobile_cap = cap
        elif cap is not None:
            cap.release()

        print(f"Dang chuyen sang Camera [{target_cam}]...")
        time.sleep(0.15)
        new_cap = _open_camera(target_cam, rec["width"], rec["height"], rec["fps"], rec_cfg=rec)
        if new_cap.isOpened():
            cap = new_cap
            current_cam = target_cam
            w, h, fps = _read_cam_props(cap, rec["fps"])
            _print_banner()
        else:
            print(f"-> Camera [{target_cam}] khong kha dung. Khoi phuc lai Camera [{current_cam}]...")
            if cur_is_mobile and mobile_cap is not None and mobile_cap.isOpened():
                cap = mobile_cap
                current_cam = "mobile"
            else:
                cap = _open_camera(current_cam, w, h, rec["fps"], rec_cfg=rec)

    _print_banner()

    while True:
        # Kiem tra dong cua so qua nut 'X'
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

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
                    raw_display = cap.get_qr_card()
                    frame_ready = False
                else:
                    frame = last_mobile_frame
                    raw_display = frame
                    frame_ready = True
            else:
                last_mobile_frame = frame
                raw_display = frame
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
            raw_display = frame
            frame_ready = True
            new_frame_arrived = True

        if frame_ready and writer is not None and new_frame_arrived:
            write_frame = frame
            if frame.shape[1] != w or frame.shape[0] != h:
                write_frame = cv2.resize(frame, (w, h))
            writer.write(write_frame)
            recorded_frames += 1

        # --- LETTERBOXING & PRO HUD RENDERING ---
        win_w, win_h = 1040, 660
        try:
            r = cv2.getWindowImageRect(win_name)
            if r[2] >= 480 and r[3] >= 320:
                win_w, win_h = r[2], r[3]
        except Exception:
            pass

        canvas = np.full((win_h, win_w, 3), (16, 20, 28), dtype=np.uint8)

        # Tinh toan Letterboxing giu nguyen ti le goc
        top_bar_h = 42
        bot_bar_h = 56
        avail_w = win_w
        avail_h = max(100, win_h - top_bar_h - bot_bar_h)

        disp_source = raw_display if (is_mobile and not cap.has_frame) else frame.copy()
        dh, dw = disp_source.shape[:2]
        scale = min(avail_w / max(dw, 1), avail_h / max(dh, 1))
        sc_w = max(1, int(dw * scale))
        sc_h = max(1, int(dh * scale))
        x_off = (avail_w - sc_w) // 2
        y_off = top_bar_h + (avail_h - sc_h) // 2

        # Dat video vao trung tam canvas
        canvas[y_off : y_off + sc_h, x_off : x_off + sc_w] = cv2.resize(disp_source, (sc_w, sc_h))

        # Ve khung vien video nhe
        cv2.rectangle(canvas, (x_off, y_off), (x_off + sc_w, y_off + sc_h), (40, 48, 64), 1)

        # Ve luoi 3D / Rule of Thirds neu bat
        if show_grid and frame_ready and not (is_mobile and not cap.has_frame):
            gx1 = x_off + sc_w // 3
            gx2 = x_off + (2 * sc_w) // 3
            gy1 = y_off + sc_h // 3
            gy2 = y_off + (2 * sc_h) // 3
            grid_col = (70, 85, 110)
            cv2.line(canvas, (gx1, y_off), (gx1, y_off + sc_h), grid_col, 1)
            cv2.line(canvas, (gx2, y_off), (gx2, y_off + sc_h), grid_col, 1)
            cv2.line(canvas, (x_off, gy1), (x_off + sc_w, gy1), grid_col, 1)
            cv2.line(canvas, (x_off, gy2), (x_off + sc_w, gy2), grid_col, 1)
            # Center crosshair
            cx, cy = x_off + sc_w // 2, y_off + sc_h // 2
            cv2.line(canvas, (cx - 12, cy), (cx + 12, cy), (96, 165, 250), 1)
            cv2.line(canvas, (cx, cy - 12), (cx, cy + 12), (96, 165, 250), 1)

        # TOP BAR HUD
        cv2.rectangle(canvas, (0, 0), (win_w, top_bar_h), (11, 15, 23), -1)
        cv2.line(canvas, (0, top_bar_h), (win_w, top_bar_h), (34, 42, 56), 1)

        src_label = "MOBILE WI-FI" if is_mobile else f"USB CAM [{current_cam}]"
        src_color = (250, 180, 50) if is_mobile else (52, 211, 153)
        cv2.circle(canvas, (20, 21), 6, src_color, -1)
        cv2.putText(canvas, src_label, (34, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (241, 245, 249), 1)

        elapsed = (time.time() - t0) if writer is not None else 0.0
        cur_fps = (recorded_frames / elapsed) if elapsed > 0.5 else fps
        info_txt = f"{w}x{h} | {cur_fps:.1f} FPS"
        cv2.putText(canvas, info_txt, (200, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (148, 163, 184), 1)

        if writer is not None:
            # Pulsing REC Badge
            rec_badge_w = 210
            rx = win_w - rec_badge_w - 15
            cv2.rectangle(canvas, (rx, 7), (rx + rec_badge_w, 35), (30, 20, 20), -1)
            cv2.rectangle(canvas, (rx, 7), (rx + rec_badge_w, 35), (239, 68, 68), 1)
            cv2.circle(canvas, (rx + 16, 21), 5, (239, 68, 68), -1)
            cv2.putText(canvas, f"REC {elapsed:4.1f}s | {recorded_frames} f", (rx + 28, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (254, 202, 202), 1)

        # BOTTOM BAR TOOLBAR (Buttons with Mouse Click Hitboxes)
        cv2.rectangle(canvas, (0, win_h - bot_bar_h), (win_w, win_h), (11, 15, 23), -1)
        cv2.line(canvas, (0, win_h - bot_bar_h), (win_w, win_h - bot_bar_h), (34, 42, 56), 1)

        btn_y = win_h - bot_bar_h + 10
        btn_h = 36
        button_hitboxes.clear()

        # Build interactive buttons
        if is_mobile:
            mob_btn = ("DISCONNECT_MOBILE", "NGAT MOBILE (X)", (153, 27, 27))
        elif mobile_cap is not None and mobile_cap.isOpened():
            mob_btn = ("MOBILE", "MOBILE: SAN SANG (M)", (21, 128, 61))
        else:
            mob_btn = ("MOBILE", "MOBILE QR (M)", (51, 65, 85))

        buttons_def = [
            ("REC", "DUNG QUAY (R)" if writer is not None else "QUAY VIDEO (R)", (239, 68, 68) if writer is not None else (37, 99, 235)),
            ("CAM", "DOI CAM (C)", (51, 65, 85)),
            mob_btn,
            ("GRID", "LUOI: ON (G)" if show_grid else "LUOI: OFF (G)", (51, 65, 85)),
            ("QUIT", "THOAT (Q)", (71, 85, 105)),
        ]

        total_btn_w = min(win_w - 30, 780)
        btn_w = total_btn_w // len(buttons_def)
        start_bx = (win_w - (btn_w * len(buttons_def))) // 2

        for idx, (b_act, b_title, b_bg) in enumerate(buttons_def):
            bx = start_bx + idx * btn_w + 3
            bw = btn_w - 6
            button_hitboxes[b_act] = (bx, btn_y, bw, btn_h)
            cv2.rectangle(canvas, (bx, btn_y), (bx + bw, btn_y + btn_h), b_bg, -1)
            cv2.rectangle(canvas, (bx, btn_y), (bx + bw, btn_y + btn_h), (71, 85, 105), 1)
            text_size = cv2.getTextSize(b_title, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
            tx = bx + (bw - text_size[0]) // 2
            ty = btn_y + (btn_h + text_size[1]) // 2
            cv2.putText(canvas, b_title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (241, 245, 249), 1)

        cv2.imshow(win_name, canvas)

        # XU LY PHIM & SU KIEN CHUOT
        key = cv2.waitKey(1) & 0xFF
        action = clicked_action
        clicked_action = None

        trigger_record = key_trigger_record or (action == "REC") or key in (ord("r"), ord("R"))
        trigger_grid = (action == "GRID") or key in (ord("g"), ord("G"))
        trigger_mobile = (action == "MOBILE") or (writer is None and key in (ord("m"), ord("M")))
        trigger_disconnect = (action == "DISCONNECT_MOBILE") or (writer is None and key in (ord("x"), ord("X")))
        trigger_cam = (action == "CAM") or (writer is None and key in (ord("c"), ord("C")))
        trigger_quit = (action == "QUIT") or key in (ord("q"), ord("Q"), 27)

        if trigger_grid:
            show_grid = not show_grid

        elif trigger_disconnect:
            _disconnect_mobile()

        elif trigger_record:
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

        elif trigger_mobile:
            if not is_mobile:
                _switch_source("mobile")

        elif trigger_cam or (writer is None and (ord("0") <= key <= ord("9"))):
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

        elif trigger_quit:
            break

    if writer is not None:  # thoat khi dang quay -> van luu va dong bo FPS
        writer.release()
        if _is_mobile_cap(cap):
            cap.set_recording_state(False)
        actual_fps = _finalize_recorded_video(
            temp_video_path, video_path, recorded_frames, time.time() - t0, fps, (w, h)
        )
        saved = True
    if mobile_cap is not None:
        try:
            mobile_cap.release()
        except Exception:
            pass
        mobile_cap = None

    if cap is not None and not _is_mobile_cap(cap):
        try:
            cap.release()
        except Exception:
            pass
        cap = None

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

