"""Buoc 3: Pose camera (ArUco hoac COLMAP SfM) + Dense (COLMAP MVS hoac splatfacto) -> point cloud .ply.

Vao : data/samples/<mau>/frames_sharp/*.jpg, data/samples/<mau>/masks/*.png
Ra  : data/samples/<mau>/nerfstudio/         (colmap/, transforms.json, images/, masks/)
      outputs/<mau>/point_cloud.ply
"""
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path
import cv2
import numpy as np

from . import paths


def _run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _promote_largest_model(sparse_dir: Path) -> None:
    """COLMAP co the sinh nhieu model (0, 1, 2...). nerfstudio chi doc sparse/0 -> dua model lon nhat ve 0."""
    def size(d: Path) -> int:
        f = d / "images.bin"
        if not f.exists():
            f = d / "images.txt"
        return f.stat().st_size if f.exists() else 0

    models = sorted((d for d in sparse_dir.iterdir() if d.is_dir()), key=size, reverse=True)
    if not models:
        raise RuntimeError("COLMAP mapper khong tao ra model nao (khong dung duoc cap anh khoi tao).")
    best = models[0]
    print(f"[Buoc 3] COLMAP tao {len(models)} model; chon '{best.name}'")
    if best.name != "0":
        zero, tmp = sparse_dir / "0", sparse_dir / "_swap"
        if zero.exists():
            zero.rename(tmp)
        best.rename(zero)
        if tmp.exists():
            tmp.rename(sparse_dir / best.name)


def _finish_process_data(sdir: Path, nf_dir: Path, cfg: dict | None = None) -> Path:
    """Goi ns-process-data --skip-colmap (doc colmap/sparse/0) -> transforms.json, roi copy anh."""
    images = sdir / "frames_sharp"
    method = (cfg or {}).get("reconstruction", {}).get("method", "colmap_mvs")
    if method != "colmap_mvs" or shutil.which("ns-process-data"):
        try:
            _run([
                "ns-process-data", "images",
                "--data", str(images),
                "--output-dir", str(nf_dir),
                "--skip-colmap",
                "--skip-image-processing",
            ])
        except FileNotFoundError:
            if method != "colmap_mvs":
                raise
    images_out = nf_dir / "images"
    images_out.mkdir(exist_ok=True)
    for p in images.glob("frame_*.jpg"):
        shutil.copy(p, images_out / p.name)
    return nf_dir


def _process_data_colmap_sfm(sdir: Path, nf_dir: Path, cfg: dict) -> Path:
    """Nhanh COLMAP tu do pose bang SfM (turntable: chi trich dac trung trong mask)."""
    rc = cfg["reconstruction"]
    use_mask = rc.get("capture_mode", "orbit") == "turntable"

    colmap_dir = nf_dir / "colmap"
    (colmap_dir / "sparse").mkdir(parents=True)
    db = colmap_dir / "database.db"
    images = sdir / "frames_sharp"

    peak_thresh = str(rc.get("sift_peak_threshold", 0.001))
    max_features = str(rc.get("sift_max_features", 16384))

    extract = [
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(images),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
        "--FeatureExtraction.use_gpu", "1",
        "--SiftExtraction.peak_threshold", peak_thresh,
        "--SiftExtraction.max_num_features", max_features,
    ]
    if use_mask:
        mask_dir = colmap_dir / "masks"
        mask_dir.mkdir()
        for m in (sdir / "masks").glob("frame_*.png"):
            shutil.copy(m, mask_dir / f"{m.stem}.jpg.png")
        extract += ["--ImageReader.mask_path", str(mask_dir)]
    _run(extract)

    # Chay sequential matcher de lien ket cac frame lien tiep theo thoi gian
    overlap = str(rc.get("sequential_overlap", 20))
    min_inliers = str(rc.get("matching_min_inliers", 8))
    seq_cmd = [
        "colmap", "sequential_matcher",
        "--database_path", str(db),
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.guided_matching", "1",
        "--SequentialMatching.overlap", overlap,
        "--SequentialMatching.quadratic_overlap", "1",
        "--TwoViewGeometry.min_num_inliers", min_inliers,
    ]
    _run(seq_cmd)

    # Chay exhaustive matcher de ket noi them cac frame doi dien & cheo goc
    exh_cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", str(db),
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.guided_matching", "1",
        "--TwoViewGeometry.min_num_inliers", min_inliers,
    ]
    _run(exh_cmd)

    _run([
        "colmap", "mapper",
        "--database_path", str(db),
        "--image_path", str(images),
        "--output_path", str(colmap_dir / "sparse"),
        "--Mapper.min_num_matches", str(rc.get("mapper_min_num_matches", 8)),
        "--Mapper.init_min_num_inliers", str(rc.get("mapper_init_min_inliers", 20)),
        "--Mapper.abs_pose_min_num_inliers", str(rc.get("mapper_abs_pose_min_inliers", 8)),
        "--Mapper.filter_max_reproj_error", str(rc.get("mapper_filter_max_reproj_error", 6.0)),
        "--Mapper.tri_ignore_two_view_tracks", "0",
    ])
    _promote_largest_model(colmap_dir / "sparse")
    return _finish_process_data(sdir, nf_dir, cfg)

def _process_data(sdir: Path, cfg: dict) -> Path:
    rc = cfg["reconstruction"]
    nf_dir = sdir / "nerfstudio"
    if nf_dir.exists():
        shutil.rmtree(nf_dir)

    pose_method = rc.get("pose_method", "colmap")  # default colmap

    return _process_data_colmap_sfm(sdir, nf_dir, cfg)  # tên hàm COLMAP của bạn


def _read_colmap_sparse_positions(sparse_0: Path) -> list[np.ndarray]:
    """Doc toa do tam camera 3D truc tiep tu colmap/sparse/0/images.bin."""
    bin_file = sparse_0 / "images.bin"
    positions: list[np.ndarray] = []
    if bin_file.exists():
        try:
            with open(bin_file, "rb") as f:
                num_images = int(struct.unpack("<Q", f.read(8))[0])
                for _ in range(num_images):
                    _image_id = struct.unpack("<i", f.read(4))[0]
                    qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
                    tx, ty, tz = struct.unpack("<3d", f.read(24))
                    _camera_id = struct.unpack("<i", f.read(4))[0]
                    while True:
                        c = f.read(1)
                        if not c or c == b"\x00":
                            break
                    num_pts2d = struct.unpack("<Q", f.read(8))[0]
                    f.seek(num_pts2d * 24, 1)
                    R = np.array([
                        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                    ])
                    t = np.array([tx, ty, tz])
                    positions.append(-R.T @ t)
        except Exception:
            pass
    return positions


def _count_colmap_images(sparse_0: Path) -> int:
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


def _check_poses(nf_dir: Path, sdir: Path, min_ratio: float, warn_collapse: bool = True) -> None:
    tj = nf_dir / "transforms.json"
    n_all = len(list((sdir / "frames_sharp").glob("frame_*.jpg")))

    positions = []
    if tj.exists():
        data = json.loads(tj.read_text(encoding="utf-8"))
        frames = data.get("frames", [])
        n_pose = len(frames)
        for f in frames:
            m = np.array(f.get("transform_matrix", []))
            if m.shape == (4, 4):
                positions.append(m[:3, 3])
    else:
        # Neu transforms.json chua co (vi du mode colmap_mvs), doc truc tiep tu colmap sparse model
        sparse_0 = nf_dir / "colmap" / "sparse" / "0"
        n_pose = _count_colmap_images(sparse_0)
        positions = _read_colmap_sparse_positions(sparse_0)

    print(f"[Buoc 3] Dinh vi duoc {n_pose}/{n_all} anh")
    if n_pose / max(n_all, 1) < min_ratio:
        raise RuntimeError(
            f"Chi dinh vi duoc {n_pose}/{n_all} anh (< {min_ratio:.0%}). "
            "Dung lai: khong du du lieu de dung dense."
        )

    # Phan tich quy dao camera de phat hien Pose Collapse (gay ra hien tuong song bien)
    if warn_collapse and len(positions) >= 5:
        pos_arr = np.array(positions)
        center = pos_arr.mean(axis=0)
        centered = pos_arr - center
        cov = np.cov(centered, rowvar=False)
        w, v = np.linalg.eigh(cov)
        p2d = centered @ v[:, 1:]
        angles = np.arctan2(p2d[:, 1], p2d[:, 0]) * 180 / np.pi
        radii = np.linalg.norm(p2d, axis=1)

        angle_span = float(angles.max() - angles.min())
        r_mean = float(radii.mean())
        r_std = float(radii.std())
        r_var = r_std / max(r_mean, 1e-5)

        print(f"  quy dao camera: goc quet={angle_span:.1f}° | ban kinh={r_mean:.2f} (std={r_std:.2f})")
        if angle_span < 160.0 or r_var > 0.45:
            print("\n" + "=" * 70)
            print("  [CANH BAO SfM: NGUY CO POSE COLLAPSE / SONG BIEN]")
            print(f"  - Camera chi bao phu {angle_span:.1f}° quanh vat the (can ~360°).")
            print(f"  - Do bien thien ban kinh rat lon ({r_var:.0%}), camera bi keo lech.")
            print("  => Voi vat the tron xoay/it texture (lo, coc, chai), cac camera de bi ket")
            print("     vao 1 cung hep va chong cheo thanh nhieu lop vo (song bien).")
            print("  => Giai phap:")
            print("     1. Dat vat the len de co hoa van tren ban xoay (bat base_pad_bottom_frac).")
            print("     2. Dan 2-3 mieng sticker/bang keo danh dau khac nhau quanh than vat the.")
            print("     3. Thu matching_method: sequential trong configs/default.yaml.")
            print("=" * 70 + "\n")


def _attach_masks(nf_dir: Path, masks_dir: Path) -> None:
    tj_path = nf_dir / "transforms.json"
    data = json.loads(tj_path.read_text(encoding="utf-8"))
    masks_out = nf_dir / "masks"
    masks_out.mkdir(exist_ok=True)
    n_matched = 0
    for frame in data["frames"]:
        img_name = Path(frame["file_path"]).name
        mask_src = masks_dir / Path(img_name).with_suffix(".png").name
        if not mask_src.exists():
            continue
        mask_dst = masks_out / mask_src.name
        shutil.copy(mask_src, mask_dst)
        frame["mask_path"] = f"masks/{mask_dst.name}"
        n_matched += 1
    tj_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[Buoc 3] Gan mask cho {n_matched}/{len(data['frames'])} frame")


def _train_splatfacto(nf_dir: Path, sample_name: str) -> Path:
    out_dir = paths.sample_dir(sample_name) / "nerfstudio_output"
    _run([
        "ns-train", "splatfacto",
        "--data", str(nf_dir),
        "--output-dir", str(out_dir),
        "--viewer.quit-on-train-completion", "True",
        "--pipeline.datamanager.cache-images", "cpu",
        "--max-num-iterations", "15000",
    ])
    configs = sorted(out_dir.rglob("config.yml"))
    if not configs:
        raise RuntimeError("Khong tim thay config.yml sau khi train.")
    return configs[-1]


def _export_pointcloud(config_yml: Path, sample_name: str) -> Path:
    export_dir = paths.sample_dir(sample_name) / "nerfstudio_export"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    _run([
        "ns-export", "gaussian-splat",
        "--load-config", str(config_yml),
        "--output-dir", str(export_dir),
    ])
    ply_files = list(export_dir.glob("*.ply"))
    if not ply_files:
        raise RuntimeError(f"ns-export khong tao ra file .ply trong {export_dir}")
    out_ply = paths.output_dir(sample_name) / "point_cloud.ply"
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ply_files[0], out_ply)
    return out_ply


def _dense_mvs(sdir: Path, nf_dir: Path, sample_name: str, cfg: dict) -> Path:
    rc = cfg["reconstruction"]
    dense = nf_dir / "dense"
    if dense.exists():
        shutil.rmtree(dense)

    _run([
        "colmap", "image_undistorter",
        "--image_path", str(sdir / "frames_sharp"),
        "--input_path", str(nf_dir / "colmap" / "sparse" / "0"),
        "--output_path", str(dense),
        "--output_type", "COLMAP",
        "--max_image_size", str(rc.get("mvs_max_image_size", 1200)),
    ])

    geom = rc.get("mvs_geom_consistency", False)
    try:
        _run([
            "colmap", "patch_match_stereo",
            "--workspace_path", str(dense),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true" if geom else "false",
        ])
    except subprocess.CalledProcessError as err:
        # Tren Windows, CUDA DLL doi khi tra ve ma loi (nhu 3221226505) khi giai phong bo nho o cuoi lenh
        # Neu da ghi duoc depth maps thi coi nhu thanh cong va tiep tuc
        depth_maps = list((dense / "stereo" / "depth_maps").glob("*.photometric.bin"))
        if len(depth_maps) > 0:
            print(f"  [COLMAP MVS] Da tao {len(depth_maps)} depth maps (bo qua ma loi exit {err.returncode})")
        else:
            raise

    min_pixels = int(rc.get("mvs_min_num_pixels", 2))
    max_reproj = float(rc.get("mvs_max_reproj_error", 2.5))
    fuse = [
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric" if geom else "photometric",
        "--output_type", "PLY",
        "--output_path", str(dense / "fused.ply"),
        "--StereoFusion.min_num_pixels", str(min_pixels),
        "--StereoFusion.max_reproj_error", str(max_reproj),
    ]
    masks = sdir / "masks"
    if any(masks.glob("frame_*.png")):
        mask_dir = dense / "masks"
        mask_dir.mkdir(exist_ok=True)
        for img_p in (dense / "images").glob("frame_*.jpg"):
            m_src = masks / f"{img_p.stem}.png"
            if not m_src.exists():
                continue
            img_u = cv2.imread(str(img_p))
            m_im = cv2.imread(str(m_src), cv2.IMREAD_GRAYSCALE)
            if img_u is not None and m_im is not None:
                hu, wu = img_u.shape[:2]
                if m_im.shape[:2] != (hu, wu):
                    m_im = cv2.resize(m_im, (wu, hu), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(mask_dir / f"{img_p.name}.png"), m_im)
                cv2.imwrite(str(mask_dir / f"{img_p.stem}.png"), m_im)
            else:
                shutil.copy(m_src, mask_dir / f"{img_p.name}.png")
                shutil.copy(m_src, mask_dir / f"{img_p.stem}.png")
        fuse += ["--StereoFusion.mask_path", str(mask_dir)]
    _run(fuse)

    out_ply = paths.output_dir(sample_name) / "point_cloud.ply"
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(dense / "fused.ply", out_ply)
    return out_ply


def run(sample_name: str, cfg: dict) -> None:
    os.environ["MAX_JOBS"] = "1"
    os.environ["NINJA_FLAGS"] = "-j1"
    sdir = paths.sample_dir(sample_name)
    masks_dir = sdir / "masks"
    if not masks_dir.exists() or not any(masks_dir.glob("*.png")):
        raise FileNotFoundError(f"Khong co mask trong {masks_dir}, chay Buoc 2 truoc.")

    nf_dir = _process_data(sdir, cfg)
    _check_poses(
        nf_dir,
        sdir,
        float(cfg["reconstruction"].get("min_pose_ratio", 0.5)),
        bool(cfg["reconstruction"].get("warn_pose_collapse", True)),
    )

    if cfg["reconstruction"].get("method", "splatfacto") == "colmap_mvs":
        print("[Buoc 3] Dense reconstruction bang COLMAP MVS...")
        out_ply = _dense_mvs(sdir, nf_dir, sample_name, cfg)
    else:
        print("[Buoc 3] Gan mask vao transforms.json...")
        _attach_masks(nf_dir, masks_dir)
        print("[Buoc 3] Train splatfacto...")
        config_yml = _train_splatfacto(nf_dir, sample_name)
        print("[Buoc 3] Xuat point cloud...")
        out_ply = _export_pointcloud(config_yml, sample_name)
    print(f"[Buoc 3] Xong -> {out_ply}")