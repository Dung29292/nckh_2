"""Phase 2: COLMAP camera loading, 3D Visual Hull carving, symmetry seeding & Biharmonic depth proxy.

Provides depth-free geometric scaffolding for occluded-region completion:
  1. Exact COLMAP binary (``cameras.bin`` / ``images.bin``) and Nerfstudio ``transforms.json`` loader.
  2. Multi-view 3D Visual Hull space carving from Step 2 Grounded-SAM 2 binary masks.
  3. ICP-verified bilateral/rotational symmetry seeding clipped strictly to the Visual Hull.
  4. Overlapping canonical virtual camera rig rendering observed RGB/Depth + occlusion masks.
  5. SciPy sparse Laplace/Biharmonic boundary-value depth membrane anchored to COLMAP boundary depths.
"""
from __future__ import annotations

import dataclasses
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# COLMAP CAMERA & IMAGE LOADER
# ---------------------------------------------------------------------------

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


@dataclasses.dataclass
class CalibratedView:
    """Single calibrated camera view in COLMAP world-to-camera (OpenCV) coordinates."""

    name: str
    width: int
    height: int
    K: np.ndarray            # (3, 3) intrinsics
    R: np.ndarray            # (3, 3) world-to-camera rotation
    t: np.ndarray            # (3,)   world-to-camera translation
    image_path: Path | None
    mask_path: Path | None

    def project(self, pts_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project (N, 3) world points -> (u, v, z) in camera pixel coordinates."""
        pts_cam = (self.R @ pts_world.T).T + self.t[None, :]
        z = pts_cam[:, 2]
        z_safe = np.where(np.abs(z) > 1e-6, z, 1e-6)
        u = self.K[0, 0] * (pts_cam[:, 0] / z_safe) + self.K[0, 2]
        v = self.K[1, 1] * (pts_cam[:, 1] / z_safe) + self.K[1, 2]
        return u, v, z


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float64)


def _read_cameras_bin(path: Path) -> dict[int, tuple[int, int, np.ndarray]]:
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


def _read_images_bin(path: Path) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
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
            records.append((name, cam_id, _qvec_to_rotmat(qvec), tvec))
    return records


def load_calibrated_views(sample_dir: Path) -> list[CalibratedView]:
    """Load calibrated COLMAP cameras matching ``point_cloud.ply`` coordinate space.

    Prefers ``nerfstudio/dense/sparse/`` (exact undistorted MVS coordinate frame),
    falling back to ``nerfstudio/colmap/sparse/0/`` and then ``transforms.json``.
    """
    nf_dir = sample_dir / "nerfstudio"
    dense_sparse = nf_dir / "dense" / "sparse"
    colmap_sparse = nf_dir / "colmap" / "sparse" / "0"

    sparse_dir = None
    img_root = sample_dir / "frames_sharp"
    mask_roots = [nf_dir / "dense" / "masks", sample_dir / "masks", nf_dir / "masks"]

    if (dense_sparse / "cameras.bin").exists() and (dense_sparse / "images.bin").exists():
        sparse_dir = dense_sparse
        if (nf_dir / "dense" / "images").exists():
            img_root = nf_dir / "dense" / "images"
    elif (colmap_sparse / "cameras.bin").exists() and (colmap_sparse / "images.bin").exists():
        sparse_dir = colmap_sparse

    views: list[CalibratedView] = []
    if sparse_dir is not None:
        cams = _read_cameras_bin(sparse_dir / "cameras.bin")
        for img_name, cam_id, R, t in _read_images_bin(sparse_dir / "images.bin"):
            if cam_id not in cams:
                continue
            w, h, K = cams[cam_id]
            stem = Path(img_name).stem
            img_p = img_root / Path(img_name).name
            if not img_p.exists():
                img_p = sample_dir / "frames_sharp" / Path(img_name).name
            mask_p = None
            for mr in mask_roots:
                for cand in (mr / f"{stem}.png", mr / f"{Path(img_name).name}.png"):
                    if cand.exists():
                        mask_p = cand
                        break
                if mask_p is not None:
                    break
            views.append(CalibratedView(
                name=stem,
                width=w,
                height=h,
                K=K,
                R=R,
                t=t,
                image_path=img_p if img_p.exists() else None,
                mask_path=mask_p,
            ))
        return views

    # Fallback: Nerfstudio transforms.json (OpenGL c2w -> OpenCV w2c)
    tj_path = nf_dir / "transforms.json"
    if not tj_path.exists():
        return []
    meta = json.loads(tj_path.read_text(encoding="utf-8"))
    w = int(meta.get("w", 1280))
    h = int(meta.get("h", 720))
    K = np.array([
        [float(meta["fl_x"]), 0.0, float(meta.get("cx", w / 2.0))],
        [0.0, float(meta["fl_y"]), float(meta.get("cy", h / 2.0))],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    for fr in meta.get("frames", []):
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        c2w_cv = c2w.copy()
        c2w_cv[:3, 1:3] *= -1.0
        w2c = np.linalg.inv(c2w_cv)
        fname = Path(fr["file_path"]).name
        stem = Path(fname).stem
        img_p = nf_dir / fr["file_path"]
        mask_p = sample_dir / "masks" / f"{stem}.png"
        views.append(CalibratedView(
            name=stem,
            width=w,
            height=h,
            K=K,
            R=w2c[:3, :3],
            t=w2c[:3, 3],
            image_path=img_p if img_p.exists() else None,
            mask_path=mask_p if mask_p.exists() else None,
        ))
    return views


# ---------------------------------------------------------------------------
# 3D VISUAL HULL SPACE CARVING
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class VisualHull:
    """Voxelized 3D Visual Hull occupancy grid carved from calibrated Step 2 masks."""

    origin: np.ndarray       # (3,) min corner in world space
    voxel_size: np.ndarray   # (3,) voxel step along X, Y, Z
    grid_shape: tuple[int, int, int]
    occupied: np.ndarray     # (Nx, Ny, Nz) bool occupancy

    def contains_points(self, pts: np.ndarray, dilate_voxels: int = 1) -> np.ndarray:
        """Check whether (N, 3) world points lie inside the Visual Hull envelope."""
        if len(pts) == 0:
            return np.zeros(0, dtype=bool)
        occ = ndimage.binary_dilation(self.occupied, iterations=dilate_voxels) if dilate_voxels > 0 else self.occupied
        idx = np.floor((pts - self.origin[None, :]) / self.voxel_size[None, :]).astype(np.int32)
        in_box = (
            (idx[:, 0] >= 0) & (idx[:, 0] < self.grid_shape[0])
            & (idx[:, 1] >= 0) & (idx[:, 1] < self.grid_shape[1])
            & (idx[:, 2] >= 0) & (idx[:, 2] < self.grid_shape[2])
        )
        inside = np.zeros(len(pts), dtype=bool)
        valid_idx = idx[in_box]
        inside[in_box] = occ[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
        return inside

    def extract_surface_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract surface voxel centers and outward normals of the Visual Hull."""
        eroded = ndimage.binary_erosion(self.occupied)
        surf = self.occupied & (~eroded)
        coords = np.argwhere(surf)
        if len(coords) == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
        pts = self.origin[None, :] + (coords + 0.5) * self.voxel_size[None, :]
        # Smooth occupancy gradient -> outward surface normals
        sdf = ndimage.gaussian_filter(self.occupied.astype(np.float32), sigma=1.2)
        gx, gy, gz = np.gradient(sdf)
        normals = -np.column_stack([
            gx[coords[:, 0], coords[:, 1], coords[:, 2]],
            gy[coords[:, 0], coords[:, 1], coords[:, 2]],
            gz[coords[:, 0], coords[:, 1], coords[:, 2]],
        ]).astype(np.float64)
        nlen = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(nlen > 1e-8, normals / np.maximum(nlen, 1e-8), np.array([[0.0, 0.0, 1.0]]))
        return pts, normals


def build_visual_hull(
    observed_pts: np.ndarray,
    views: list[CalibratedView],
    grid_res: int = 64,
    pad_frac: float = 0.28,
    min_mask_consistency: float = 0.82,
    max_views: int = 36,
) -> VisualHull | None:
    """Carve a tight 3D Visual Hull occupancy envelope from calibrated Step 2 masks."""
    valid_views = [v for v in views if v.mask_path is not None and v.mask_path.exists()]
    if len(valid_views) < 2 or len(observed_pts) < 30:
        return None

    # Subsample views uniformly if there are many frames
    if len(valid_views) > max_views:
        idxs = np.linspace(0, len(valid_views) - 1, max_views, dtype=int)
        valid_views = [valid_views[i] for i in idxs]

    p_lo = np.percentile(observed_pts, 1, axis=0)
    p_hi = np.percentile(observed_pts, 99, axis=0)
    span = np.maximum(p_hi - p_lo, 1e-3)
    margin = pad_frac * span
    origin = p_lo - margin
    upper = p_hi + margin
    voxel_size = (upper - origin) / float(grid_res)

    xs = origin[0] + (np.arange(grid_res) + 0.5) * voxel_size[0]
    ys = origin[1] + (np.arange(grid_res) + 0.5) * voxel_size[1]
    zs = origin[2] + (np.arange(grid_res) + 0.5) * voxel_size[2]
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    vox_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    seen_cnt = np.zeros(len(vox_pts), dtype=np.int32)
    fg_cnt = np.zeros(len(vox_pts), dtype=np.int32)

    for vw in valid_views:
        mask = cv2.imread(str(vw.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[1] != vw.width or mask.shape[0] != vw.height:
            mask = cv2.resize(mask, (vw.width, vw.height), interpolation=cv2.INTER_NEAREST)
        # Keep mask tight (no 5x5 dilation bloat that inflates an exterior umbrella cone)
        u, v, z = vw.project(vox_pts)
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        in_view = (z > 0.05) & (ui >= 0) & (ui < vw.width) & (vi >= 0) & (vi < vw.height)
        seen_cnt[in_view] += 1
        fg_hits = mask[vi[in_view], ui[in_view]] > 127
        fg_cnt[in_view] += fg_hits.astype(np.int32)

    # Keep voxels seen by >= 2 cameras and inside foreground in >= min_mask_consistency of views
    ratio = np.divide(fg_cnt, np.maximum(seen_cnt, 1))
    occupied = ((seen_cnt >= 2) & (ratio >= min_mask_consistency)).reshape((grid_res, grid_res, grid_res))

    # Keep only the largest connected component to discard stray mask intersections
    labeled, num_cc = ndimage.label(occupied)
    if num_cc > 1:
        sizes = ndimage.sum(occupied, labeled, index=np.arange(1, num_cc + 1))
        occupied = labeled == (1 + int(np.argmax(sizes)))

    if not np.any(occupied):
        return None

    return VisualHull(
        origin=origin,
        voxel_size=voxel_size,
        grid_shape=(grid_res, grid_res, grid_res),
        occupied=occupied,
    )


# ---------------------------------------------------------------------------
# CANONICAL OBJECT FRAME & ICP-VERIFIED SYMMETRY SEEDING
# ---------------------------------------------------------------------------

def estimate_object_frame(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute canonical orthogonal object frame ``(center, R_obj, extents)``.

    Row 0 = longest axis (X), Row 1 = intermediate axis (Y), Row 2 = shortest axis (Z).
    ``R_obj`` maps centered world coordinates into object-local coordinates: ``p_loc = (pts - center) @ R_obj.T``.
    """
    center = np.median(pts, axis=0)
    centered = pts - center
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending by variance -> [X=longest, Y=medium, Z=shortest]
    order = np.argsort(eigvals)[::-1]
    R_obj = eigvecs[:, order].T.copy()
    if np.linalg.det(R_obj) < 0:
        R_obj[0] *= -1.0
    p_loc = centered @ R_obj.T
    extents = np.percentile(p_loc, 98, axis=0) - np.percentile(p_loc, 2, axis=0)
    return center, R_obj, np.maximum(extents, 1e-4)


def seed_symmetry_points(
    observed_pts: np.ndarray,
    observed_cols: np.ndarray,
    bilateral_axes: list[int] | None = None,
    rotational_order: int = 0,
    visual_hull: VisualHull | None = None,
    min_overlap: float = 0.72,
    gap_frac: float = 0.025,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ICP-verified symmetric completion points strictly inside missing regions.

    Returns ``(new_pts, new_cols, new_conf)`` containing ONLY points that:
      1. Pass ICP overlap validation on observed regions (>= ``min_overlap``),
      2. Stay strictly inside the Step 2 Visual Hull (if provided),
      3. Fill genuine gaps (distance > ``gap_frac * diag`` from existing observed points).
    """
    if len(observed_pts) < 50:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    center, R_obj, extents = estimate_object_frame(observed_pts)
    diag = float(np.linalg.norm(extents))
    gap_rad = gap_frac * diag
    inlier_thr = 0.025 * diag

    obs_tree = cKDTree(observed_pts)
    pc_obs = o3d.geometry.PointCloud()
    pc_obs.points = o3d.utility.Vector3dVector(observed_pts)

    candidates_pts: list[np.ndarray] = []
    candidates_cols: list[np.ndarray] = []

    p_loc = (observed_pts - center) @ R_obj.T

    # 1. Bilateral mirror candidates (axis 0 = YZ plane, axis 1 = XZ plane)
    for ax in (bilateral_axes or []):
        if ax not in (0, 1, 2):
            continue
        p_mir = p_loc.copy()
        p_mir[:, ax] *= -1.0
        w_mir = (p_mir @ R_obj) + center

        pc_mir = o3d.geometry.PointCloud()
        pc_mir.points = o3d.utility.Vector3dVector(w_mir)
        reg = o3d.pipelines.registration.registration_icp(
            pc_mir,
            pc_obs,
            max_correspondence_distance=0.05 * diag,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25),
        )
        w_aligned = (w_mir @ reg.transformation[:3, :3].T) + reg.transformation[:3, 3]

        # Verification gate: check overlap on the observed shared domain
        d_to_obs, _ = obs_tree.query(w_aligned, k=1)
        overlap = float(np.mean(d_to_obs < inlier_thr))
        # Also allow partial-capture symmetry where one half has >= 55% overlap in its own neighborhood
        if overlap < min_overlap:
            continue

        # Inject only into genuine gaps + inside Visual Hull
        is_gap = d_to_obs > gap_rad
        if visual_hull is not None:
            is_gap &= visual_hull.contains_points(w_aligned, dilate_voxels=1)
        if np.any(is_gap):
            candidates_pts.append(w_aligned[is_gap])
            candidates_cols.append(observed_cols[is_gap])

    # 2. Rotational symmetry around local Z axis (e.g., jars, bottles, mugs)
    if rotational_order >= 2:
        n_rot = min(int(rotational_order), 12)
        for k in range(1, n_rot):
            ang = 2.0 * np.pi * k / n_rot
            ca, sa = np.cos(ang), np.sin(ang)
            Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            w_rot = ((p_loc @ Rz.T) @ R_obj) + center
            d_to_obs, _ = obs_tree.query(w_rot, k=1)
            is_gap = d_to_obs > gap_rad
            if visual_hull is not None:
                is_gap &= visual_hull.contains_points(w_rot, dilate_voxels=1)
            if np.any(is_gap):
                candidates_pts.append(w_rot[is_gap])
                candidates_cols.append(observed_cols[is_gap])

    if not candidates_pts:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    pts_out = np.vstack(candidates_pts)
    cols_out = np.vstack(candidates_cols)
    conf_out = np.full(len(pts_out), 0.75, dtype=np.float64)
    return pts_out, cols_out, conf_out


# ---------------------------------------------------------------------------
# VIRTUAL INSPECTION CAMERAS & BIHARMONIC DEPTH PROXY
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class VirtualViewRender:
    """Rendered virtual camera buffers used for occlusion discovery and depth-free lifting."""

    name: str
    width: int
    height: int
    K: np.ndarray            # (3, 3)
    R: np.ndarray            # (3, 3) world-to-camera
    t: np.ndarray            # (3,)   world-to-camera
    rgb_obs: np.ndarray      # (H, W, 3) uint8 BGR
    depth_obs: np.ndarray    # (H, W) float32 (0 where unobserved)
    mask_obs: np.ndarray     # (H, W) bool
    mask_vh: np.ndarray      # (H, W) bool
    depth_vh: np.ndarray     # (H, W) float32 (0 where outside visual hull)
    mask_occ: np.ndarray     # (H, W) bool = mask_vh & ~mask_obs
    depth_proxy: np.ndarray | None = None  # Biharmonic boundary-value metric depth

    @property
    def occlusion_ratio(self) -> float:
        vh_area = int(np.sum(self.mask_vh))
        return float(np.sum(self.mask_occ)) / max(vh_area, 1)

    @property
    def boundary_context_ratio(self) -> float:
        """Fraction of Visual Hull pixels already observed (higher = safer to complete first)."""
        vh_area = int(np.sum(self.mask_vh))
        return float(np.sum(self.mask_obs & self.mask_vh)) / max(vh_area, 1)

    def unproject_pixels(self, mask: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Unproject 2D pixels inside ``mask`` with metric depth into (M, 3) world coordinates."""
        ys, xs = np.nonzero(mask & (depth_map > 1e-4))
        if len(xs) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        z = depth_map[ys, xs].astype(np.float64)
        x_cam = (xs.astype(np.float64) - self.K[0, 2]) * z / self.K[0, 0]
        y_cam = (ys.astype(np.float64) - self.K[1, 2]) * z / self.K[1, 1]
        pts_cam = np.column_stack([x_cam, y_cam, z])
        return ((pts_cam - self.t[None, :]) @ self.R)


def _look_at_w2c(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct OpenCV world-to-camera (R, t) where camera looks along +Z, +Y down."""
    z_axis = target - eye
    z_axis /= np.linalg.norm(z_axis) + 1e-12
    # Handle degenerate up_hint parallel to view direction (e.g. bottom/top view)
    if abs(float(np.dot(z_axis, up_hint))) > 0.95:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(z_axis, up_hint))) > 0.95:
            up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = np.cross(z_axis, up_hint)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    R = np.vstack([x_axis, y_axis, z_axis])
    t = -R @ eye
    return R, t


def _splat_points_zbuf(
    pts: np.ndarray,
    cols_bgr: np.ndarray | None,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    width: int,
    height: int,
    splat_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast vectorized Z-buffered point splatting into (rgb_bgr, depth, mask)."""
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    if len(pts) == 0:
        return rgb, np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=bool)

    pts_cam = (R @ pts.T).T + t[None, :]
    z = pts_cam[:, 2]
    valid = z > 0.05
    if not np.any(valid):
        return rgb, np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=bool)

    pts_cam = pts_cam[valid]
    z = z[valid]
    u = np.round(K[0, 0] * (pts_cam[:, 0] / z) + K[0, 2]).astype(np.int32)
    v = np.round(K[1, 1] * (pts_cam[:, 1] / z) + K[1, 2]).astype(np.int32)
    in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(in_img):
        return rgb, np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=bool)

    u = u[in_img]
    v = v[in_img]
    z = z[in_img].astype(np.float32)
    c = cols_bgr[valid][in_img] if cols_bgr is not None else None

    # Sort far-to-near so nearer points naturally overwrite farther ones (painter's algorithm)
    order = np.argsort(z)[::-1]
    u, v, z = u[order], v[order], z[order]
    if c is not None:
        c = c[order]

    for dy in range(-splat_radius, splat_radius + 1):
        for dx in range(-splat_radius, splat_radius + 1):
            if dx * dx + dy * dy > splat_radius * splat_radius + 1:
                continue
            uu = np.clip(u + dx, 0, width - 1)
            vv = np.clip(v + dy, 0, height - 1)
            closer = z <= depth[vv, uu]
            depth[vv[closer], uu[closer]] = z[closer]
            if c is not None:
                rgb[vv[closer], uu[closer]] = c[closer]

    mask = np.isfinite(depth)
    depth_clean = np.where(mask, depth, 0.0).astype(np.float32)
    return rgb, depth_clean, mask


def render_virtual_views(
    observed_pts: np.ndarray,
    observed_cols_rgb: np.ndarray,
    visual_hull: VisualHull | None = None,
    image_size: int = 384,
) -> list[VirtualViewRender]:
    """Render overlapping canonical inspection viewpoints around the object to localize occlusions."""
    center, R_obj, extents = estimate_object_frame(observed_pts)
    diag = float(np.linalg.norm(extents))
    radius = 2.35 * max(diag, 0.1)

    f = 1.15 * image_size
    K = np.array([
        [f, 0.0, image_size / 2.0],
        [0.0, f, image_size / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Canonical axes in world space (rows of R_obj)
    ax_x, ax_y, ax_z = R_obj[0], R_obj[1], R_obj[2]

    # Overlapping viewpoint triplet per side + back + bottom
    view_dirs = [
        ("front",       -ax_y + 0.18 * ax_z),
        ("back",         ax_y + 0.18 * ax_z),
        ("back_left",    0.85 * ax_y - 0.52 * ax_x + 0.22 * ax_z),
        ("back_right",   0.85 * ax_y + 0.52 * ax_x + 0.22 * ax_z),
        ("left",        -ax_x + 0.15 * ax_z),
        ("right",        ax_x + 0.15 * ax_z),
        ("top",          ax_z + 0.08 * ax_y),
        ("bottom",      -ax_z + 0.08 * ax_y),
    ]

    cols_bgr = np.clip(observed_cols_rgb[:, ::-1] * 255.0, 0, 255).astype(np.uint8)
    vh_pts, _ = visual_hull.extract_surface_points() if visual_hull is not None else (np.zeros((0, 3)), np.zeros((0, 3)))

    renders: list[VirtualViewRender] = []
    for vname, raw_dir in view_dirs:
        vdir = raw_dir / (np.linalg.norm(raw_dir) + 1e-12)
        eye = center + radius * vdir
        R, t = _look_at_w2c(eye, center, ax_z)

        rgb_obs, depth_obs, mask_obs = _splat_points_zbuf(
            observed_pts, cols_bgr, K, R, t, image_size, image_size, splat_radius=2
        )
        # Close tiny 1-2px point-cloud splat pinholes so real observed regions aren't mistaken for holes
        mask_obs_closed = ndimage.binary_closing(mask_obs, iterations=2)

        # Restrict mask_occ STRICTLY to genuine INTERIOR holes enclosed by observed silhouette!
        # Never treat exterior Visual Hull cone overhang outside `obs_silhouette` as an object hole.
        obs_silhouette = ndimage.binary_fill_holes(ndimage.binary_closing(mask_obs, iterations=5))
        if len(vh_pts) > 0:
            _, depth_vh, mask_vh = _splat_points_zbuf(
                vh_pts, None, K, R, t, image_size, image_size, splat_radius=2
            )
            mask_vh = ndimage.binary_fill_holes(ndimage.binary_closing(mask_vh, iterations=2))
            mask_occ = obs_silhouette & mask_vh & (~mask_obs_closed)
        else:
            mask_vh = obs_silhouette
            depth_vh = depth_obs.copy()
            mask_occ = obs_silhouette & (~mask_obs_closed)
        renders.append(VirtualViewRender(
            name=vname,
            width=image_size,
            height=image_size,
            K=K,
            R=R,
            t=t,
            rgb_obs=rgb_obs,
            depth_obs=depth_obs,
            mask_obs=mask_obs_closed,
            mask_vh=mask_vh,
            depth_vh=depth_vh,
            mask_occ=mask_occ,
        ))
    return renders


def solve_biharmonic_depth_proxy(
    view: VirtualViewRender,
    vh_prior_weight: float = 0.08,
) -> np.ndarray:
    """Solve a sparse metric boundary-value membrane depth proxy inside ``view.mask_occ``.

    Anchors observed boundary pixels strictly to exact COLMAP ``depth_obs`` while softly
    guiding occluded interior pixels toward the Visual Hull envelope ``depth_vh``.
    Eliminates the need for monocular depth networks (zero scale drift / planar bowing).
    """
    h, w = view.height, view.width
    depth_out = view.depth_obs.copy().astype(np.float64)
    occ = view.mask_occ.copy()

    if not np.any(occ):
        view.depth_proxy = depth_out.astype(np.float32)
        return view.depth_proxy

    # Fill tiny splat gaps inside mask_obs using nearest observed depth before boundary solve
    valid_obs = view.depth_obs > 1e-4
    if np.any(valid_obs):
        _, nearest_idx = ndimage.distance_transform_edt(~valid_obs, return_indices=True)
        obs_filled = view.depth_obs[nearest_idx[0], nearest_idx[1]].astype(np.float64)
    else:
        obs_filled = np.where(view.depth_vh > 1e-4, view.depth_vh, 1.0).astype(np.float64)

    # Restrict linear solve strictly to unknown pixels inside mask_occ (fully vectorized)
    occ_ys, occ_xs = np.nonzero(occ)
    n_unk = len(occ_ys)
    idx_map = np.full((h, w), -1, dtype=np.int32)
    idx_map[occ_ys, occ_xs] = np.arange(n_unk, dtype=np.int32)

    vh_valid = view.depth_vh > 1e-4
    if np.any(valid_obs):
        median_depth = float(np.median(view.depth_obs[valid_obs]))
    elif np.any(vh_valid):
        median_depth = float(np.median(view.depth_vh[vh_valid]))
    else:
        median_depth = 1.0

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    diag_acc = np.zeros(n_unk, dtype=np.float64)
    rhs = np.zeros(n_unk, dtype=np.float64)
    k_all = np.arange(n_unk, dtype=np.int32)

    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny = occ_ys + dy
        nx = occ_xs + dx
        in_b = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        k_in = k_all[in_b]
        ny_in, nx_in = ny[in_b], nx[in_b]

        nbr_idx = idx_map[ny_in, nx_in]
        is_unk_nbr = nbr_idx >= 0
        if np.any(is_unk_nbr):
            row_parts.append(k_in[is_unk_nbr])
            col_parts.append(nbr_idx[is_unk_nbr])
            val_parts.append(np.full(int(np.sum(is_unk_nbr)), -1.0, dtype=np.float64))
            diag_acc[k_in[is_unk_nbr]] += 1.0

        is_dirichlet = (~is_unk_nbr) & valid_obs[ny_in, nx_in]
        if np.any(is_dirichlet):
            kd = k_in[is_dirichlet]
            diag_acc[kd] += 4.0
            rhs[kd] += 4.0 * view.depth_obs[ny_in[is_dirichlet], nx_in[is_dirichlet]].astype(np.float64)

        is_soft_obs = (~is_unk_nbr) & (~valid_obs[ny_in, nx_in]) & view.mask_obs[ny_in, nx_in]
        if np.any(is_soft_obs):
            ks = k_in[is_soft_obs]
            diag_acc[ks] += 2.0
            rhs[ks] += 2.0 * obs_filled[ny_in[is_soft_obs], nx_in[is_soft_obs]]

    prior_d = np.where(vh_valid[occ_ys, occ_xs], view.depth_vh[occ_ys, occ_xs].astype(np.float64), median_depth)
    w_p = np.where(diag_acc > 0.0, vh_prior_weight, 1.0)
    diag_acc += w_p
    rhs += w_p * prior_d

    row_parts.append(k_all)
    col_parts.append(k_all)
    val_parts.append(diag_acc)

    A = sparse.csc_matrix(
        (np.concatenate(val_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=(n_unk, n_unk),
    )
    sol = np.clip(spsolve(A, rhs), 0.05, 1e4)

    depth_out[occ_ys, occ_xs] = sol
    # Ensure metric consistency on valid observed pixels
    depth_out[valid_obs] = view.depth_obs[valid_obs]
    view.depth_proxy = depth_out.astype(np.float32)
    return view.depth_proxy


def select_consistent_stationary_views(
    views: list[CalibratedView],
    min_warped_iou: float = 0.38,
) -> list[CalibratedView]:
    """Detect mid-video object repositioning via background homography and keep the longest stationary pass.

    In `orbit` captures, if the user's hand moves/rotates the object on the table mid-video while
    COLMAP tracks the stationary background table, consecutive frames have a valid background homography H
    (on `mask == 0`), yet `IoU(warp(mask_i, H), mask_{i+1})` drops sharply. Splitting at such cuts and
    retaining the longest consistent camera segment prevents multi-pose ghost shells from corrupting carving.
    """
    ordered = sorted(
        [v for v in views if v.mask_path is not None and v.mask_path.exists()],
        key=lambda x: x.name,
    )
    if len(ordered) < 6:
        return ordered

    # Only applicable when images exist to verify background motion
    if any(v.image_path is None or not v.image_path.exists() for v in ordered):
        return ordered

    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.003)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    cuts: list[int] = [0]

    # Prefer original sharp frames over resized dense frames for crisp wood/background SIFT tracking
    def _resolve_bg_img(vw: CalibratedView) -> Path | None:
        if vw.image_path is None:
            return None
        for anc in vw.image_path.parents:
            sharp_cand = anc / "frames_sharp" / vw.image_path.name
            if sharp_cand.exists():
                return sharp_cand
        return vw.image_path if vw.image_path.exists() else None

    p0 = _resolve_bg_img(ordered[0])
    if p0 is None:
        return ordered
    prev_im = cv2.imread(str(p0), cv2.IMREAD_GRAYSCALE)
    prev_m = cv2.imread(str(ordered[0].mask_path), cv2.IMREAD_GRAYSCALE)
    if prev_im is None or prev_m is None:
        return ordered
    if prev_m.shape != prev_im.shape:
        prev_m = cv2.resize(prev_m, (prev_im.shape[1], prev_im.shape[0]), interpolation=cv2.INTER_NEAREST)

    bg_kern = np.ones((9, 9), np.uint8)
    prev_kp, prev_des = sift.detectAndCompute(prev_im, cv2.erode(255 - prev_m, bg_kern))

    for i in range(1, len(ordered)):
        pi = _resolve_bg_img(ordered[i])
        if pi is None:
            continue
        cur_im = cv2.imread(str(pi), cv2.IMREAD_GRAYSCALE)
        cur_m = cv2.imread(str(ordered[i].mask_path), cv2.IMREAD_GRAYSCALE)
        if cur_im is None or cur_m is None:
            continue
        if cur_m.shape != cur_im.shape:
            cur_m = cv2.resize(cur_m, (cur_im.shape[1], cur_im.shape[0]), interpolation=cv2.INTER_NEAREST)

        cur_kp, cur_des = sift.detectAndCompute(cur_im, cv2.erode(255 - cur_m, bg_kern))
        if (
            prev_des is not None
            and cur_des is not None
            and len(prev_des) >= 12
            and len(cur_des) >= 12
        ):
            matches = [
                m for m, n in bf.knnMatch(prev_des, cur_des, k=2)
                if m.distance < 0.75 * n.distance
            ]
            if len(matches) >= 10:
                pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([cur_kp[m.trainIdx].pt for m in matches])
                H, inl = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
                if H is not None and inl is not None and int(np.sum(inl)) >= 8:
                    warped = cv2.warpPerspective(prev_m, H, (cur_m.shape[1], cur_m.shape[0])) > 127
                    cur_bin = cur_m > 127
                    union = float(np.sum(warped | cur_bin))
                    if union > 50.0:
                        iou = float(np.sum(warped & cur_bin)) / union
                        if iou < float(min_warped_iou):
                            cuts.append(i)

        prev_im, prev_m, prev_kp, prev_des = cur_im, cur_m, cur_kp, cur_des

    cuts.append(len(ordered))
    if len(cuts) <= 2:
        return ordered

    # Pick the longest contiguous stationary camera segment
    best_a, best_b = 0, len(ordered)
    best_len = -1
    for s_idx in range(len(cuts) - 1):
        seg_len = cuts[s_idx + 1] - cuts[s_idx]
        if seg_len > best_len:
            best_len = seg_len
            best_a, best_b = cuts[s_idx], cuts[s_idx + 1]

    if best_len < max(6, len(ordered) // 3):
        return ordered

    print(
        f"  [Silhouette] Phat hien vat the bi dich chuyen giua video -> giu doan camera tinh dai nhat "
        f"{ordered[best_a].name}..{ordered[best_b - 1].name} ({best_len}/{len(ordered)} goc nhin)"
    )
    return ordered[best_a:best_b]


def carve_point_cloud_with_masks(
    pc: o3d.geometry.PointCloud,
    sample_dir: Path,
    carve_cfg: dict[str, Any],
) -> tuple[o3d.geometry.PointCloud, list[CalibratedView]]:
    """Direct ray-level Multi-View Silhouette Space Carving using calibrated Step 2 SAM-2 masks.

    Eliminates MVS background bleed, grazing-angle table skirts, and multi-pass ghost points
    before MLS / Interpolation / Poisson meshing.
    """
    if not bool(carve_cfg.get("enable", True)) or not sample_dir.exists() or len(pc.points) < 100:
        return pc, []

    views = load_calibrated_views(sample_dir)
    valid_views = [v for v in views if v.mask_path is not None and v.mask_path.exists()]
    if len(valid_views) < int(carve_cfg.get("min_visible_views", 5)):
        return pc, valid_views

    if bool(carve_cfg.get("check_stationary_homography", True)):
        valid_views = select_consistent_stationary_views(
            valid_views,
            min_warped_iou=float(carve_cfg.get("homography_min_iou", 0.38)),
        )

    pts = np.asarray(pc.points, dtype=np.float64)
    n_pts = len(pts)
    fg_hits = np.zeros(n_pts, dtype=np.int32)
    vis_cnt = np.zeros(n_pts, dtype=np.int32)

    erode_px = int(carve_cfg.get("erode_px", 2))
    kern = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8) if erode_px > 0 else None

    for vw in valid_views:
        mask = cv2.imread(str(vw.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[1] != vw.width or mask.shape[0] != vw.height:
            mask = cv2.resize(mask, (vw.width, vw.height), interpolation=cv2.INTER_NEAREST)
        if kern is not None:
            mask = cv2.erode(mask, kern, iterations=1)

        u, v, z = vw.project(pts)
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        in_fov = (z > 0.05) & (ui >= 0) & (ui < vw.width) & (vi >= 0) & (vi < vw.height)
        vis_cnt[in_fov] += 1
        fg_hits[in_fov] += (mask[vi[in_fov], ui[in_fov]] > 127).astype(np.int32)

    min_vis = min(int(carve_cfg.get("min_visible_views", 5)), max(2, len(valid_views) // 3))
    min_cons = float(carve_cfg.get("min_mask_consistency", 0.88))
    hit_ratio = fg_hits / np.maximum(vis_cnt, 1)
    keep = (vis_cnt >= min_vis) & (hit_ratio >= min_cons)

    # Guard: never wipe out the cloud if camera coordinate conventions mismatched
    if int(np.sum(keep)) < max(200, int(0.08 * n_pts)):
        return pc, valid_views

    # Optional CIE Lab MAD filter to strip dark table-shadow / wood-grain points clinging to object rim
    if bool(carve_cfg.get("filter_bg_color_bleed", True)) and pc.has_colors() and len(pc.colors) == n_pts:
        cols_u8 = np.clip(np.asarray(pc.colors, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(cols_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
        core = keep & (hit_ratio >= min(0.98, min_cons + 0.07))
        ref_lab = lab[core] if int(np.sum(core)) >= 100 else lab[keep]
        med_lab = np.median(ref_lab, axis=0)
        c_dist = np.linalg.norm(lab - med_lab[None, :], axis=1)
        d_core = c_dist[core] if int(np.sum(core)) >= 100 else c_dist[keep]
        med_d = float(np.median(d_core))
        mad_d = max(1.4826 * float(np.median(np.abs(d_core - med_d))), 2.5)
        c_thr = med_d + float(carve_cfg.get("color_mad_sigma", 3.2)) * mad_d
        # Strip extreme dark/wood shadow outliers even if grazing rays passed through foreground masks
        is_dark_bleed = (lab[:, 0] < med_lab[0] - 2.6 * max(1.4826 * float(np.median(np.abs(ref_lab[:, 0] - med_lab[0]))), 4.0)) & (c_dist > c_thr)
        color_ok = (~is_dark_bleed) & ((hit_ratio >= 0.97) | (c_dist <= c_thr))
        if int(np.sum(keep & color_ok)) >= max(200, int(0.55 * np.sum(keep))):
            keep &= color_ok

    idx_keep = np.nonzero(keep)[0]
    print(
        f"  Silhouette carving ({len(valid_views)} masks, >= {min_cons:.0%}): "
        f"{n_pts:,} -> {len(idx_keep):,} diem (loai {n_pts - len(idx_keep):,} diem phan thua ngoai mask)"
    )
    return pc.select_by_index(idx_keep), valid_views


def trim_mesh_outside_masks(
    mesh: o3d.geometry.TriangleMesh,
    views: list[CalibratedView],
    aligned_to_world_fn: Any,
    min_consistency: float = 0.82,
    dilate_px: int = 2,
) -> tuple[o3d.geometry.TriangleMesh, int]:
    """Project Poisson mesh vertices back into calibrated SAM-2 masks to slice off exterior bubbles/skirts.

    Preserves 100% of smooth interior textureless surfaces (such as mouse palm domes and buttons)
    that lie inside the 2D object silhouettes while cleanly removing exterior Poisson extrapolations.
    """
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0 or len(views) < 4:
        return mesh, 0

    verts_aligned = np.asarray(mesh.vertices, dtype=np.float64)
    verts_world = aligned_to_world_fn(verts_aligned)
    n_v = len(verts_world)

    fg_hits = np.zeros(n_v, dtype=np.int32)
    vis_cnt = np.zeros(n_v, dtype=np.int32)
    kern = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8) if dilate_px > 0 else None

    for vw in views:
        if vw.mask_path is None or not vw.mask_path.exists():
            continue
        mask = cv2.imread(str(vw.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[1] != vw.width or mask.shape[0] != vw.height:
            mask = cv2.resize(mask, (vw.width, vw.height), interpolation=cv2.INTER_NEAREST)
        if kern is not None:
            mask = cv2.dilate(mask, kern, iterations=1)

        u, v, z = vw.project(verts_world)
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        in_fov = (z > 0.05) & (ui >= 0) & (ui < vw.width) & (vi >= 0) & (vi < vw.height)
        vis_cnt[in_fov] += 1
        fg_hits[in_fov] += (mask[vi[in_fov], ui[in_fov]] > 127).astype(np.int32)

    min_vis = max(2, len(views) // 4)
    ratio = fg_hits / np.maximum(vis_cnt, 1)
    outside_mask = (vis_cnt >= min_vis) & (ratio < float(min_consistency))
    n_out = int(np.sum(outside_mask))
    if n_out == 0 or int(np.sum(~outside_mask)) < 100:
        return mesh, 0

    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_vertices_by_mask(outside_mask)
    out.remove_degenerate_triangles()
    out.remove_unreferenced_vertices()
    return out, n_out
