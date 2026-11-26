"""Tien ich quan ly thiet bi tinh toan (CPU / CUDA GPU)."""
from __future__ import annotations

import gc


def get_device() -> str:
    """Tra ve 'cuda' neu he thong co GPU NVIDIA ho tro CUDA, nguoc lai tra ve 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def clear_cuda_cache() -> None:
    """Giai phong bo nho dem VRAM khi chuyen giao giua cac buoc tinh toan."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
