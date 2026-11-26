"""Capture package: Quay video USB/Mobile camera va quan ly tap mau."""
from nckh_2.capture.camera_io import (
    AsyncVideoWriter,
    LiveCameraReader,
    finalize_recorded_video,
    is_mobile_cap,
    is_mobile_source,
    open_camera,
    read_cam_props,
)
from nckh_2.capture.mobile_camera import MobileVideoCapture, load_mobile_html
from nckh_2.capture.recorder import record_video
from nckh_2.capture.sample_store import choose_sample, list_samples, unique_name

__all__ = [
    "AsyncVideoWriter",
    "LiveCameraReader",
    "MobileVideoCapture",
    "choose_sample",
    "finalize_recorded_video",
    "is_mobile_cap",
    "is_mobile_source",
    "list_samples",
    "load_mobile_html",
    "open_camera",
    "read_cam_props",
    "record_video",
    "unique_name",
]
