"""Doc & phan tich du lieu camera tu COLMAP (binary / text) va Nerfstudio transforms.json."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import numpy as np

from nckh_2.core import paths

_COLMAP_MODEL_PARAMS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float64)


def read_cameras_bin(path: Path) -> dict[int, tuple[int, int, np.ndarray]]:
    cams: dict[int, tuple[int, int, np.ndarray]] = {}
    with path.open("rb") as f:
        num_cams = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cams):
            cam_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            model_name, n_params = _COLMAP_MODEL_PARAMS.get(model_id, ("PINHOLE", 4))
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            if model_name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            cams[cam_id] = (int(width), int(height), K)
    return cams


def read_images_bin(path: Path) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    records: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    with path.open("rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            data = struct.unpack("<idddddddi", f.read(64))
            qvec = np.array(data[1:5], dtype=np.float64)
            tvec = np.array(data[5:8], dtype=np.float64)
            cam_id = int(data[8])
            name_bytes = bytearray()
            while True:
                ch = f.read(1)
                if ch in (b"", b"\x00"):
                    break
                name_bytes.extend(ch)
            name = name_bytes.decode("utf-8", errors="replace")
            num_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(num_pts2d * 24, 1)
            records.append((name, cam_id, qvec_to_rotmat(qvec), tvec))
    return records


def read_colmap_sparse_positions(sparse_0: Path) -> list[np.ndarray]:
    """Doc toa do tam camera 3D truc tiep tu colmap/sparse/0/images.bin."""
    bin_file = sparse_0 / "images.bin"
    positions: list[np.ndarray] = []
    if bin_file.exists():
        try:
            for _name, _cid, R, t in read_images_bin(bin_file):
                positions.append(-R.T @ t)
        except Exception:
            pass
    return positions


def count_colmap_images(sparse_0: Path) -> int:
    bin_file = sparse_0 / "images.bin"
    if bin_file.exists():
        try:
            with open(bin_file, "rb") as f:
                return int(struct.unpack("<Q", f.read(8))[0])
        except Exception:
            pass
    txt_file = sparse_0 / "images.txt"
    if txt_file.exists():
        try:
            lines = [l for l in txt_file.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
            return len(lines) // 2
        except Exception:
            pass
    return 0


def orbit_normal(cam_centers: np.ndarray | None) -> np.ndarray | None:
    if cam_centers is None or len(cam_centers) < 3:
        return None
    cov = np.cov(cam_centers - cam_centers.mean(axis=0), rowvar=False)
    _, ev = np.linalg.eigh(cov)
    return ev[:, 0].copy()


def load_camera_centers(sample_name: str, pc_center: np.ndarray) -> np.ndarray:
    """Lay tam camera, tu chon he toa do (goc / da normalize cua nerfstudio) khop point cloud."""
    sample_dir = paths.sample_dir(sample_name)
    out_dir = paths.output_dir(sample_name)
    cands: list[np.ndarray] = []

    # 1. transforms.json
    for tj in (sample_dir / "nerfstudio" / "transforms.json", sample_dir / "transforms.json"):
        if not tj.exists():
            continue
        try:
            data = json.loads(tj.read_text(encoding="utf-8"))
            cs = [np.array(fr["transform_matrix"])[:3, 3]
                  for fr in data.get("frames", [])
                  if np.array(fr.get("transform_matrix", [])).shape == (4, 4)]
            if cs:
                raw = np.array(cs, dtype=np.float64)
                cands.append(raw)
                dps = list(sample_dir.rglob("dataparser_transforms.json")) + \
                      list(out_dir.rglob("dataparser_transforms.json"))
                for dp in dps[:1]:
                    d = json.loads(dp.read_text(encoding="utf-8"))
                    T = np.array(d["transform"], dtype=np.float64)
                    s = float(d["scale"])
                    cands.append((raw @ T[:, :3].T + T[:, 3]) * s)
                break
        except Exception:
            pass

    # 2. COLMAP images.bin
    if not cands:
        for bf in (sample_dir / "nerfstudio" / "colmap" / "sparse" / "0" / "images.bin",
                   sample_dir / "colmap" / "sparse" / "0" / "images.bin"):
            if bf.exists():
                try:
                    c = [pos for pos in read_colmap_sparse_positions(bf.parent)]
                    if len(c):
                        cands.append(np.array(c, dtype=np.float64))
                        break
                except Exception:
                    pass

    if not cands:
        return np.empty((0, 3), dtype=np.float64)

    best_idx = min(range(len(cands)), key=lambda i: float(np.linalg.norm(cands[i].mean(axis=0) - pc_center)))
    best = cands[best_idx]
    if len(cands) > 1:
        print(f"  chon he toa do camera #{best_idx} (khop point cloud hon)")
    return best
