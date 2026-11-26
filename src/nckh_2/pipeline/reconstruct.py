"""Buoc 3: Pose camera (COLMAP SfM) + Dense (COLMAP MVS hoac splatfacto) -> point cloud .ply.

Vao : data/samples/<mau>/frames_sharp/*.jpg, data/samples/<mau>/masks/*.png
Ra  : data/samples/<mau>/nerfstudio/         (colmap/, transforms.json, images/, masks/)
      outputs/<mau>/point_cloud.ply
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import cv2
import numpy as np

from nckh_2.backends.colmap_io import count_colmap_images, read_colmap_sparse_positions
from nckh_2.core import paths
from nckh_2.core.types import SceneContext, StageResult


def _run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _promote_largest_model(sparse_dir: Path) -> None:
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


def _finish_process_data(sdir: Path, nf_dir: Path, cfg: dict[str, Any] | None = None) -> Path:
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


def _process_data_colmap_sfm(sdir: Path, nf_dir: Path, cfg: dict[str, Any]) -> Path:
    rc = cfg["reconstruction"]
    mask_files = list((sdir / "masks").glob("frame_*.png"))
    sfm_mask_mode = str(rc.get("sfm_mask_mode", "auto")).strip().lower()
    if sfm_mask_mode == "always":
        use_mask = len(mask_files) > 0
    elif sfm_mask_mode == "never":
        use_mask = False
    else:
        # auto: turntable rotates object against moving background -> MUST mask SfM;
        # orbit moves camera around static desk -> keep static bg for rock-solid SfM poses, then mask Dense MVS!
        use_mask = rc.get("capture_mode", "orbit") == "turntable" and len(mask_files) > 0

    colmap_dir = nf_dir / "colmap"
    (colmap_dir / "sparse").mkdir(parents=True)
    db = colmap_dir / "database.db"
    images = sdir / "frames_sharp"

    peak_thresh = str(rc.get("sift_peak_threshold", 0.001))
    max_features = str(rc.get("sift_max_features", 16384))
    camera_model = str(rc.get("camera_model", "OPENCV"))
    single_camera = "1" if bool(rc.get("single_camera", True)) else "0"

    extract = [
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(images),
        "--ImageReader.single_camera", single_camera,
        "--ImageReader.camera_model", camera_model,
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

    overlap = str(rc.get("sequential_overlap", 20))
    min_inliers = str(rc.get("matching_min_inliers", 12))
    loop_det = bool(rc.get("loop_detection", True))
    loop_period = str(rc.get("loop_detection_period", 10))
    vocab_tree = str(rc.get("vocab_tree_path", "") or "").strip()

    seq_cmd = [
        "colmap", "sequential_matcher",
        "--database_path", str(db),
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.guided_matching", "1",
        "--SequentialMatching.overlap", overlap,
        "--SequentialMatching.quadratic_overlap", "1",
        "--TwoViewGeometry.min_num_inliers", min_inliers,
    ]
    if loop_det and vocab_tree and Path(vocab_tree).exists():
        seq_cmd += [
            "--SequentialMatching.loop_detection", "1",
            "--SequentialMatching.loop_detection_period", loop_period,
            "--SequentialMatching.vocab_tree_path", vocab_tree,
        ]
    _run(seq_cmd)

    # Mandatory offline loop closure: exhaustive all-pairs matching guarantees final 20% <-> initial 20% closure
    if loop_det or str(rc.get("matching_method", "exhaustive")).lower() == "exhaustive":
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
        "--Mapper.init_min_num_inliers", str(rc.get("mapper_init_min_num_inliers", 20)),
        "--Mapper.abs_pose_min_num_inliers", str(rc.get("mapper_abs_pose_min_num_inliers", 8)),
        "--Mapper.filter_max_reproj_error", str(rc.get("mapper_filter_max_reproj_error", 6.0)),
        "--Mapper.tri_ignore_two_view_tracks", "0",
    ])
    _promote_largest_model(colmap_dir / "sparse")
    return _finish_process_data(sdir, nf_dir, cfg)


def _process_data(sdir: Path, cfg: dict[str, Any]) -> Path:
    nf_dir = sdir / "nerfstudio"
    if nf_dir.exists():
        shutil.rmtree(nf_dir)
    return _process_data_colmap_sfm(sdir, nf_dir, cfg)


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
        sparse_0 = nf_dir / "colmap" / "sparse" / "0"
        n_pose = count_colmap_images(sparse_0)
        positions = read_colmap_sparse_positions(sparse_0)

    print(f"[Buoc 3] Dinh vi duoc {n_pose}/{n_all} anh")
    if n_pose / max(n_all, 1) < min_ratio:
        raise RuntimeError(
            f"Chi dinh vi duoc {n_pose}/{n_all} anh (< {min_ratio:.0%}). "
            "Dung lai: khong du du lieu de dung dense."
        )

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


def _dense_mvs(sdir: Path, nf_dir: Path, sample_name: str, cfg: dict[str, Any]) -> Path:
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


def run(sample_name: str, cfg: dict[str, Any]) -> None:
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


class ReconstructStage:
    stage_id = 3
    title = "COLMAP + 3DGS"

    def execute(self, ctx: SceneContext) -> StageResult:
        run(ctx.sample_name, ctx.config)
        ctx.artifacts["point_cloud"] = ctx.output_dir / "point_cloud.ply"
        return StageResult(stage_id=self.stage_id, stage_name=self.title, success=True, artifacts=ctx.artifacts)
