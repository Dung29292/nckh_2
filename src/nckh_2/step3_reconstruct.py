"""Buoc 3: COLMAP (chay thu cong, co the co mask) + Nerfstudio splatfacto -> point cloud .ply.

Vao : data/samples/<mau>/frames_sharp/*.jpg, data/samples/<mau>/masks/*.png
Ra  : data/samples/<mau>/nerfstudio/         (colmap/, transforms.json, images/, masks/)
      data/samples/<mau>/nerfstudio_output/  (checkpoint + config.yml cua ns-train)
      outputs/<mau>/point_cloud.ply
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

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


def _process_data(sdir: Path, cfg: dict) -> Path:
    """Chay COLMAP thu cong (turntable: chi trich dac trung trong mask) roi dua sang nerfstudio."""
    rc = cfg["reconstruction"]
    use_mask = rc.get("capture_mode", "orbit") == "turntable"

    nf_dir = sdir / "nerfstudio"
    if nf_dir.exists():
        shutil.rmtree(nf_dir)
    colmap_dir = nf_dir / "colmap"
    (colmap_dir / "sparse").mkdir(parents=True)
    db = colmap_dir / "database.db"
    images = sdir / "frames_sharp"

    extract = [
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(images),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
        "--FeatureExtraction.use_gpu", "1",
    ]
    if use_mask:
        # COLMAP doc mask theo ten "<ten_anh>.png" (vung den = bo qua, vung trang = trich dac trung)
        mask_dir = colmap_dir / "masks"
        mask_dir.mkdir()
        for m in (sdir / "masks").glob("frame_*.png"):
            shutil.copy(m, mask_dir / f"{m.stem}.jpg.png")
        extract += ["--ImageReader.mask_path", str(mask_dir)]
    _run(extract)

    matcher = "sequential_matcher" if rc["matching_method"] == "sequential" else "exhaustive_matcher"
    _run(["colmap", matcher, "--database_path", str(db), "--FeatureMatching.use_gpu", "1"])

    _run([
        "colmap", "mapper",
        "--database_path", str(db),
        "--image_path", str(images),
        "--output_path", str(colmap_dir / "sparse"),
    ])
    _promote_largest_model(colmap_dir / "sparse")

    # doi COLMAP -> transforms.json cho nerfstudio (doc colmap/sparse/0)
    _run([
        "ns-process-data", "images",
        "--data", str(images),
        "--output-dir", str(nf_dir),
        "--skip-colmap",
        "--skip-image-processing",
    ])

    # --skip-image-processing khong copy anh -> tu copy de transforms.json tim thay images/frame_xxxxx.jpg
    images_out = nf_dir / "images"
    images_out.mkdir(exist_ok=True)
    for p in images.glob("frame_*.jpg"):
        shutil.copy(p, images_out / p.name)
    return nf_dir


def _check_poses(nf_dir: Path, sdir: Path, min_ratio: float) -> None:
    """Dung som neu COLMAP chi dinh vi duoc qua it anh -> khong train vo ich."""
    data = json.loads((nf_dir / "transforms.json").read_text(encoding="utf-8"))
    n_pose = len(data["frames"])
    n_all = len(list((sdir / "frames_sharp").glob("frame_*.jpg")))
    print(f"[Buoc 3] COLMAP dinh vi duoc {n_pose}/{n_all} anh")
    if n_pose / max(n_all, 1) < min_ratio:
        raise RuntimeError(
            f"COLMAP chi dinh vi duoc {n_pose}/{n_all} anh (< {min_ratio:.0%}). "
            "Dung lai: khong du du lieu de train. Xem lai clip quay hoac che do quay (orbit/turntable)."
        )


def _attach_masks(nf_dir: Path, masks_dir: Path) -> None:
    """Gan mask_path cho tung frame trong transforms.json de splatfacto dung masked loss."""
    tj_path = nf_dir / "transforms.json"
    data = json.loads(tj_path.read_text(encoding="utf-8"))

    masks_out = nf_dir / "masks"
    masks_out.mkdir(exist_ok=True)

    n_matched = 0
    for frame in data["frames"]:
        img_name = Path(frame["file_path"]).name  # vd images/frame_00001.jpg
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
        # giam VRAM cho GPU 6GB: giu anh o RAM thay vi VRAM
        "--pipeline.datamanager.cache-images", "cpu",
        "--max-num-iterations", "15000",
    ])
    configs = sorted(out_dir.rglob("config.yml"))
    if not configs:
        raise RuntimeError("Khong tim thay config.yml sau khi train, kiem tra log ns-train o tren.")
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
    """Dense bang COLMAP MVS (khong can gsplat): undistort -> patch_match_stereo -> stereo_fusion."""
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
    _run([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true" if geom else "false",
    ])

    fuse = [
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric" if geom else "photometric",
        "--output_path", str(dense / "fused.ply"),
    ]
    # mask cua Buoc 2: chi giu diem thuoc vat the, bo nen
    masks = sdir / "masks"
    if any(masks.glob("frame_*.png")):
        mask_dir = dense / "masks"
        mask_dir.mkdir()
        for m in masks.glob("frame_*.png"):
            shutil.copy(m, mask_dir / f"{m.stem}.jpg.png")
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

    print("[Buoc 3] Chay COLMAP...")
    nf_dir = _process_data(sdir, cfg)
    _check_poses(nf_dir, sdir, cfg["reconstruction"].get("min_pose_ratio", 0.5))

    if cfg["reconstruction"].get("method", "splatfacto") == "colmap_mvs":
        print("[Buoc 3] Dense reconstruction bang COLMAP MVS...")
        out_ply = _dense_mvs(sdir, nf_dir, sample_name, cfg)
    else:
        print("[Buoc 3] Gan mask vao transforms.json...")
        _attach_masks(nf_dir, masks_dir)
        print("[Buoc 3] Train splatfacto (co the mat vai - vai chuc phut tuy so frame)...")
        config_yml = _train_splatfacto(nf_dir, sample_name)
        print("[Buoc 3] Xuat point cloud...")
        out_ply = _export_pointcloud(config_yml, sample_name)
    print(f"[Buoc 3] Xong -> {out_ply}")