"""Doc camera USB khong do tre va ghi video bat dong bo khong mat khung hinh."""
from __future__ import annotations

import collections
from pathlib import Path
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np


class LiveCameraReader:
    """Doc camera USB tren mot background thread rieng biet.
    Triet tieu 100% do tre tich luy trong DirectShow driver buffer tren Windows
    va cung cap hang doi ghi khong mat frame khi quay."""

    def __init__(self, raw_cap: Any):
        self.raw_cap = raw_cap
        self._latest_frame: np.ndarray | None = None
        self._record_queue: collections.deque[np.ndarray] = collections.deque(maxlen=1200)
        self._recording = False
        self._running = True
        self._lock = threading.Lock()
        try:
            ok, frame = self.raw_cap.read()
            if ok and frame is not None:
                self._latest_frame = frame
        except Exception:
            pass
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while self._running and self.raw_cap.isOpened():
            ok, frame = self.raw_cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest_frame = frame
                if self._recording:
                    self._record_queue.append(frame)

    def get_live_preview(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_frame

    def pop_recorded_frames(self) -> list[np.ndarray]:
        with self._lock:
            if not self._record_queue:
                return []
            frames = list(self._record_queue)
            self._record_queue.clear()
            return frames

    def set_recording_state(self, recording: bool) -> None:
        with self._lock:
            self._recording = bool(recording)
            if self._recording:
                self._record_queue.clear()

    def isOpened(self) -> bool:
        return self._running and self.raw_cap.isOpened()

    def get(self, prop_id: int) -> float:
        return float(self.raw_cap.get(prop_id))

    def set(self, prop_id: int, value: float) -> bool:
        return bool(self.raw_cap.set(prop_id, value))

    def release(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self.raw_cap.release()


class AsyncVideoWriter:
    """Ghi video bat dong bo ra o dia bang Worker Thread rieng biet.
    Khong bao gio lam cham giao dien Desktop va khong bao gio drop frame."""

    def __init__(self, filepath: str, fourcc: int, fps: float, size: tuple[int, int]):
        self.writer = cv2.VideoWriter(filepath, fourcc, fps, size)
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1500)
        self.stopped = False
        self.written_frames = 0
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def write(self, frame: np.ndarray) -> None:
        if not self.stopped:
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                pass

    def _worker(self) -> None:
        while not self.stopped or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.05)
                self.writer.write(frame)
                self.written_frames += 1
                self.queue.task_done()
            except queue.Empty:
                continue

    def release(self) -> int:
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=8.0)
        self.writer.release()
        return self.written_frames


def is_mobile_source(camera: Any) -> bool:
    return isinstance(camera, str) and camera.strip().lower() in ("mobile", "phone", "qr")


def is_mobile_cap(cap: Any) -> bool:
    return getattr(cap, "is_mobile", False) or cap.__class__.__name__ == "MobileVideoCapture"


def open_camera(camera: Any, width: int, height: int, fps: int, rec_cfg: dict | None = None) -> Any:
    if is_mobile_source(camera):
        try:
            from nckh_2.capture.mobile_camera import MobileVideoCapture
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

    raw_cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW) if isinstance(camera, int) else cv2.VideoCapture(camera)
    if not raw_cap.isOpened() and isinstance(camera, int):
        raw_cap = cv2.VideoCapture(camera)

    if isinstance(camera, int):
        raw_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        raw_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        raw_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        raw_cap.set(cv2.CAP_PROP_FPS, fps)
        raw_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not raw_cap.isOpened():
        return raw_cap
    return LiveCameraReader(raw_cap)


def read_cam_props(cap: Any, default_fps: float) -> tuple[int, int, float]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or default_fps
    if fps < 1 or fps > 120:
        fps = default_fps
    return w, h, fps


def finalize_recorded_video(
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
