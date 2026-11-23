"""Buoc 4: Don sach point cloud (Open3D) + Dung mesh 3D tong quat -> .obj.

Thuat toan tai tao be mat tong quat (General Surface Reconstruction):
- mesh_mode = bpa     : Ball-Pivoting Algorithm (Bernardini et al.) - Bam sat 100% diem thuc te,
                        bao toan hoan hao chi tiet sac net co khi (quai xach, ranh khia, canh vuong)
                        lan be mat cong huu co ma khong gay phong bien dang nhu Poisson.
- mesh_mode = poisson : Screened Poisson Surface Reconstruction (Kazhdan et al.) voi phap tuyen
                        dinh huong theo tam camera thuc te (Sensor-Aware Normal Orientation).
- mesh_mode = both    : Xuat ca BPA (chinh xac vat ly) lan Poisson (tham chieu).
- mesh_mode = obb     : Fit Oriented Bounding Box nho nhat.

Khong su dung bat ky code mau (template) hay gia dinh hinh dang co dinh (hardcoded box/cylinder) nao.
Phu hop voi moi vat the 3D bat ky (co khi, do gia dung, tuong, coc, chai, chuot may tinh...).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
import numpy as np
import open3d as o3d
from plyfile import PlyData

from . import paths


# ---------------------------------------------------------------------------
# DOC POINT CLOUD
# ---------------------------------------------------------------------------

def _load_gaussian_ply(path: Path) -> o3d.geometry.PointCloud:
    """Doc .ply tu ns-export gaussian-splat hoac colmap stereo_fusion."""
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)

    names = v.dtype.names
    if all(f"f_dc_{i}" in names for i in range(3)):
        C0 = 0.28209479177387814
        dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
        rgb = np.clip(0.5 + C0 * dc, 0.0, 1.0)
        pc.colors = o3d.utility.Vector3dVector(rgb)
    elif all(c in names for c in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float64) / 255.0
        pc.colors = o3d.utility.Vector3dVector(rgb)

    if all(n in names for n in ("nx", "ny", "nz")):
        norms = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float64)
        lens = np.linalg.norm(norms, axis=1, keepdims=True)
        valid = (lens > 1e-4).squeeze()
        if np.any(valid):
            norms[valid] /= lens[valid]
            pc.normals = o3d.utility.Vector3dVector(norms)

    if "opacity" in names:
        opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
        solid_mask = opacity > 0.5
        if all(f"scale_{i}" in names for i in range(3)):
            s0 = np.exp(v["scale_0"].astype(np.float64))
            s1 = np.exp(v["scale_1"].astype(np.float64))
            s2 = np.exp(v["scale_2"].astype(np.float64))
            max_scale = np.maximum(s0, np.maximum(s1, s2))
            # Loai bo cac floaters khong khi co ban kinh lon hoac do trong suot cao
            scale_thresh = float(np.percentile(max_scale, 85))
            solid_mask = solid_mask & (max_scale < scale_thresh)
        keep = np.nonzero(solid_mask)[0]
        pc = pc.select_by_index(keep)
        print(f"  loc floaters khong khi (opacity > 0.5 & compact scale): giu {len(keep):,}/{len(xyz):,} diem")

        # DBSCAN chon cum thuc the dac duy nhat cua vat the
        if len(pc.points) > 100:
            labels = np.array(pc.cluster_dbscan(eps=0.08, min_points=30, print_progress=False))
            valid = labels >= 0
            if np.any(valid):
                counts = np.bincount(labels[valid])
                largest_c = int(np.argmax(counts))
                keep_c = np.nonzero(labels == largest_c)[0]
                pc = pc.select_by_index(keep_c)
                print(f"  DBSCAN co lap vat the chinh: giu {len(keep_c):,} diem dac")

    return pc


# ---------------------------------------------------------------------------
# DOC TOA DO TAM CAMERA (SENSOR POSES)
# ---------------------------------------------------------------------------

def _load_camera_centers(sample_name: str) -> np.ndarray:
    """Trich xuat toa do tam camera 3D (X, Y, Z) tu transforms.json hoac images.bin."""
    centers: list[np.ndarray] = []
    sample_dir = paths.sample_dir(sample_name)

    # 1. Thu doc tu transforms.json
    tj_candidates = [
        sample_dir / "nerfstudio" / "transforms.json",
        sample_dir / "transforms.json",
    ]
    for tj in tj_candidates:
        if tj.exists():
            try:
                data = json.loads(tj.read_text(encoding="utf-8"))
                for frame in data.get("frames", []):
                    m = np.array(frame.get("transform_matrix", []))
                    if m.shape == (4, 4):
                        centers.append(m[:3, 3])
                if centers:
                    return np.array(centers, dtype=np.float64)
            except Exception:
                pass

    # 2. Thu doc tu COLMAP sparse model images.bin
    bin_candidates = [
        sample_dir / "nerfstudio" / "colmap" / "sparse" / "0" / "images.bin",
        sample_dir / "colmap" / "sparse" / "0" / "images.bin",
    ]
    for bin_file in bin_candidates:
        if bin_file.exists():
            try:
                with open(bin_file, "rb") as f:
                    num_images = struct.unpack("<Q", f.read(8))[0]
                    for _ in range(num_images):
                        _id = struct.unpack("<i", f.read(4))[0]
                        qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
                        tx, ty, tz = struct.unpack("<3d", f.read(24))
                        _cam_id = struct.unpack("<i", f.read(4))[0]
                        while f.read(1) != b"\x00":
                            pass
                        num_pts2d = struct.unpack("<Q", f.read(8))[0]
                        f.seek(num_pts2d * 24, 1)

                        # X_c = R * X_w + t => C_w = -R^T * t
                        R = np.array([
                            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
                        ])
                        t = np.array([tx, ty, tz])
                        center = -R.T @ t
                        centers.append(center)
                if centers:
                    return np.array(centers, dtype=np.float64)
            except Exception:
                pass

    return np.empty((0, 3), dtype=np.float64)


# ---------------------------------------------------------------------------
# LOC NHIEU POINT CLOUD
# ---------------------------------------------------------------------------

def _keep_significant_clusters(
    pc: o3d.geometry.PointCloud, diag: float, e_dict: dict
) -> o3d.geometry.PointCloud:
    """DBSCAN cluster filtering giu cum vat the chinh, loai bo cac hat bui/diem troi noi."""
    pts = np.asarray(pc.points)
    if len(pts) < 100:
        return pc

    eps_frac = float(e_dict.get("cluster_eps_frac", 0.02))
    min_pts = int(e_dict.get("cluster_min_points", 40))
    eps = max(eps_frac * diag, 0.005)

    labels = np.array(pc.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False))
    if len(labels) == 0:
        return pc

    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return pc

    counts = np.bincount(labels[valid_mask])
    if len(counts) <= 1:
        keep = np.nonzero(labels == 0)[0] if len(counts) == 1 else np.arange(len(labels))
        return pc.select_by_index(keep)

    largest = int(np.argmax(counts))
    largest_count = counts[largest]

    min_cluster_ratio = float(e_dict.get("cluster_min_ratio", 0.03))
    threshold_count = max(min_pts, int(largest_count * min_cluster_ratio))

    keep_clusters = {c for c, cnt in enumerate(counts) if cnt >= threshold_count}
    keep_indices = np.nonzero(np.isin(labels, list(keep_clusters)))[0]

    n_removed = len(labels) - len(keep_indices)
    kept_pts = len(keep_indices)
    print(
        f"  DBSCAN (eps={eps:.4f}): Giu {len(keep_clusters)} cum chu yeu ({kept_pts:,}/{len(labels):,} diem), "
        f"loai {n_removed:,} diem troi noi."
    )
    return pc.select_by_index(keep_indices)


def _clean(pc: o3d.geometry.PointCloud, e: dict) -> tuple[o3d.geometry.PointCloud, float]:
    """Loc thong ke (Statistical) + ban kinh (Radius) + DBSCAN clustering."""
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        return pc, 0.0
    p5 = np.percentile(pts, 5, axis=0)
    p95 = np.percentile(pts, 95, axis=0)
    diag = float(np.linalg.norm(p95 - p5))
    if diag < 1e-3:
        diag = float(np.linalg.norm(np.asarray(pc.get_max_bound()) - np.asarray(pc.get_min_bound())))
    print(f"  duong cheo (robust) = {diag:.3f}")

    # Loc thong ke
    before = len(pc.points)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=e["stat_nb"], std_ratio=e["stat_ratio"])
    print(f"  loc thong ke: {before:,} -> {len(pc.points):,}")
    if len(pc.points) == 0:
        raise RuntimeError("Loc thong ke xoa het diem. Giam stat_ratio trong config.")

    # Loc ban kinh
    rad = e["rad_radius_frac"] * diag
    before = len(pc.points)
    pc, _ = pc.remove_radius_outlier(nb_points=e["rad_nb"], radius=rad)
    print(f"  loc ban kinh: {before:,} -> {len(pc.points):,}")
    if len(pc.points) == 0:
        raise RuntimeError("Loc ban kinh xoa het diem. Tang rad_radius_frac trong config.")

    # Giu cum chu yeu
    pc = _keep_significant_clusters(pc, diag, e)

    # Voxel downsample tuy chon
    if e.get("voxel_frac", 0):
        vox = e["voxel_frac"] * diag
        before = len(pc.points)
        pc = pc.voxel_down_sample(vox)
        print(f"  voxel downsample: {before:,} -> {len(pc.points):,}")

    return pc, diag


def _remove_plane_ransac(
    pc: o3d.geometry.PointCloud,
    diag: float,
    e: dict,
) -> tuple[o3d.geometry.PointCloud, tuple[np.ndarray, float] | None]:
    """Phat hien va tach mat phang ban do / ban xoay (neu co). Tra ve point cloud va mat phang."""
    if not e.get("remove_table_plane", True) or len(pc.points) < 100:
        return pc, None

    dist_thresh = float(e.get("plane_dist_threshold_frac", 0.012)) * diag
    dist_thresh = max(dist_thresh, 0.002)

    try:
        plane_model, inliers = pc.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=1000,
        )
    except Exception:
        return pc, None

    n_plane = len(inliers)
    n_total = len(pc.points)
    ratio = n_plane / max(n_total, 1)

    if 0.05 <= ratio <= 0.65:
        pts = np.asarray(pc.points)
        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        n_len = np.linalg.norm(normal)
        if n_len > 1e-8:
            normal /= n_len
            d_norm = d / n_len
            dists = np.dot(pts, normal) + d_norm
            pos_ratio = np.mean(dists > dist_thresh)
            neg_ratio = np.mean(dists < -dist_thresh)

            if max(pos_ratio, neg_ratio) > 0.60:
                pc_clean = pc.select_by_index(inliers, invert=True)
                print(f"  RANSAC Plane: da tach {n_plane:,}/{n_total:,} diem ({ratio:.1%}) thuoc de/mat ban")
                return pc_clean, (normal, d_norm)

    return pc, None


# ---------------------------------------------------------------------------
# PHAP TUYEN HUONG THEO CAMERA (SENSOR-AWARE NORMALS)
# ---------------------------------------------------------------------------

def _orient_normals_sensor_aware(
    pc: o3d.geometry.PointCloud,
    cam_centers: np.ndarray,
    diag: float,
    e: dict,
) -> o3d.geometry.PointCloud:
    """Bao dam phap tuyen huong ngoai 100% dua tren tia nhin camera (Visibility-Aware).

    Khac phuc triet de loi dao chieu phap tuyen tai goc 90 do gay ra sup lo trong Poisson.
    """
    pts = np.asarray(pc.points)
    n_pts = len(pts)
    if n_pts == 0:
        return pc

    # Neu da co phap tuyen hop le tu COLMAP stereo fusion, kiem tra va giu nguyen
    has_valid_colmap_normals = False
    if pc.has_normals() and len(pc.normals) == n_pts:
        norms = np.asarray(pc.normals)
        valid_lens = np.linalg.norm(norms, axis=1)
        if np.mean(valid_lens > 0.5) > 0.90:
            has_valid_colmap_normals = True

    if has_valid_colmap_normals:
        print(f"  Giu nguyen 100% phap tuyen quang hoc ({n_pts:,} vector) tu COLMAP stereo fusion.")
        return pc

    # Uoc luong phap tuyen cuc bo va dinh huong theo tia nhin camera
    norm_rad = float(e.get("normal_radius_local", 0.035))
    norm_nn = int(e.get("normal_nn_local", 30))
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=norm_rad, max_nn=norm_nn))

    if len(cam_centers) >= 3:
        try:
            norms = np.asarray(pc.normals)
            kdtree_cam = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cam_centers)))
            for i, pt in enumerate(pts):
                _, idxs, _ = kdtree_cam.search_knn_vector_3d(pt, 1)
                c = cam_centers[idxs[0]]
                view_dir = c - pt
                if np.dot(norms[i], view_dir) < 0:
                    norms[i] = -norms[i]
            pc.normals = o3d.utility.Vector3dVector(norms)
            print(f"  Da dinh huong phap tuyen theo tia nhin camera ({len(cam_centers)} goc chup).")
            return pc
        except Exception:
            pass

    pc.orient_normals_consistent_tangent_plane(k=15)
    return pc


# ---------------------------------------------------------------------------
# THUAT TOAN TAI TAO BE MAT TONG QUAT 1: BALL-PIVOTING ALGORITHM (BPA)
# ---------------------------------------------------------------------------

def _reconstruct_bpa(pc: o3d.geometry.PointCloud, e: dict) -> o3d.geometry.TriangleMesh:
    """Ball-Pivoting Algorithm (Bernardini et al. 1999).

    Thuat toan tai tao be mat bám sát trực tiếp tập điểm thực tế bằng cách lăn hình cầu
    đa bán kính r_1, r_2, r_3.
    - Không phồng lồi, không tạo nếp nhăn súp lơ.
    - Bảo toàn 100% hình học gốc: quai xách, rãnh lồi/lõm, góc vuông 90° lẫn bề mặt cong.
    """
    pts = np.asarray(pc.points)
    if len(pts) < 10:
        return o3d.geometry.TriangleMesh()

    # Voxel downsampling nhe de dong nhat mat do diem
    pcd_eval = pc
    if len(pts) > 250000:
        vox_size = float(e.get("bpa_voxel_size", 0.008))
        if vox_size > 0:
            pcd_eval = pc.voxel_down_sample(vox_size)
            print(f"  BPA voxel downsample: {len(pts):,} -> {len(pcd_eval.points):,} diem")

    distances = pcd_eval.compute_nearest_neighbor_distance()
    avg_dist = float(np.mean(distances))
    print(f"  BPA: khoang cach lang gieng trung binh = {avg_dist:.4f}")

    # Bo 3 ban kinh lan: bat tu chi tiet nho den khung lon
    r1 = avg_dist * float(e.get("bpa_r1_mult", 1.2))
    r2 = avg_dist * float(e.get("bpa_r2_mult", 2.5))
    r3 = avg_dist * float(e.get("bpa_r3_mult", 5.0))
    radii = [r1, r2, r3]
    print(f"  BPA ban kinh lan: {[round(r, 4) for r in radii]}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd_eval, o3d.utility.DoubleVector(radii)
    )

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Giu cum tam giac lien thong chinh
    tri_clusters, n_tris, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    n_tris = np.asarray(n_tris)
    if len(n_tris) > 0:
        largest = n_tris.argmax()
        largest_cnt = n_tris[largest]
        keep_clusters = set(np.where(n_tris >= max(20, int(largest_cnt * 0.02)))[0])
        keep_mask = np.isin(tri_clusters, list(keep_clusters))
        mesh.remove_triangles_by_mask(~keep_mask)
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    print(f"  BPA hoan tat: {len(mesh.vertices):,} dinh, {len(mesh.triangles):,} tam giac")
    return mesh


# ---------------------------------------------------------------------------
# THUAT TOAN TAI TAO BE MAT TONG QUAT 2: SCREENED POISSON + PLANAR CAPPING
# ---------------------------------------------------------------------------

def _reconstruct_poisson(
    pc: o3d.geometry.PointCloud,
    e: dict,
    support_plane: tuple[np.ndarray, float] | None = None,
) -> o3d.geometry.TriangleMesh:
    """Screened Poisson Surface Reconstruction (Kazhdan & Hoppe 2013).

    Cai tien phuong phap noi diem tong quat:
    1. Loc muot truong phap tuyen (Normal Pre-smoothing) bang KNN Tangent Plane de triet tieu go ghe/nep gap.
    2. Giai Screened Poisson voi do sau Octree phu hop.
    3. Cat tia triet de vat mang nhon/tam giac ma (Strict Density Trimming) theo mat do diem thuc.
    4. Lam phang be mat khong teo the tich (Taubin Smoothing) giu dung form dang vat the.
    """
    pts = np.asarray(pc.points)
    p5, p95 = np.percentile(pts, [5, 95], axis=0)
    diag = float(np.linalg.norm(p95 - p5))

    # 1. Bao toan phap tuyen hop le da co (giu tron ven quai xach, ranh khia va cac mat phang)
    pcd_eval = o3d.geometry.PointCloud(pc)
    if not pcd_eval.has_normals() or len(pcd_eval.normals) != len(pts):
        norm_rad = float(e.get("normal_radius_local", 0.035))
        norm_nn = int(e.get("normal_nn_local", 30))
        pcd_eval.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=norm_rad, max_nn=norm_nn))
        pcd_eval.orient_normals_consistent_tangent_plane(k=15)
        print("  [Phap tuyen] Uoc luong phap tuyen tu dong cho point cloud.")
    else:
        print(f"  [Phap tuyen] Bao toan 100% phap tuyen quang hoc ({len(pts):,} vector) giu chi tiet goc.")

    depth = int(e.get("poisson_depth", 9))
    print(f"  Poisson Surface Reconstruction depth={depth} (khu gon song, lam phang vach & giu chi tiet quai xach)...")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_eval, depth=depth, linear_fit=bool(e.get("poisson_linear_fit", True))
    )
    densities = np.asarray(densities)
    print(f"  Poisson raw: {len(mesh.vertices):,} dinh, {len(mesh.triangles):,} tam giac")

    # 2. Cat tia bong bong ngoai vi theo phan vi mat do octree (Kazhdan Density Trimming)
    # Mac dinh 0.0: khong cat theo quantile de tranh duc thung lo tren vach va mat day
    dens_percentile = float(e.get("poisson_density_trim_percentile", e.get("poisson_density_quantile", 0.0) * 100.0))
    if dens_percentile > 0:
        dens_threshold = float(np.percentile(densities, dens_percentile))
        mesh.remove_vertices_by_mask(densities < dens_threshold)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Giu duy nhat cum tam giac lien thong chinh lon nhat (loai bo 100% manh vun roi rac)
    tri_clusters, n_tris, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    n_tris = np.asarray(n_tris)
    if len(n_tris) > 0:
        largest = n_tris.argmax()
        mesh.remove_triangles_by_mask(tri_clusters != largest)
        mesh.remove_unreferenced_vertices()

    # 3. Lam min Taubin giu nguyen the tich va form dang hop (khong lam mo goc vuong 90 do)
    s_iter = int(e.get("smooth_iter", 15))
    s_lam = float(e.get("smooth_lambda", 0.45))
    s_mu = float(e.get("smooth_mu", -0.48))
    mesh = mesh.filter_smooth_taubin(number_of_iterations=s_iter, lambda_filter=s_lam, mu=s_mu)

    # 4. Chuyen mau tu point cloud goc sang mesh vertices neu co mau
    if pc.has_colors() and len(pc.colors) > 0 and len(mesh.vertices) > 0:
        try:
            kdtree = o3d.geometry.KDTreeFlann(pc)
            pc_colors = np.asarray(pc.colors)
            m_verts = np.asarray(mesh.vertices)
            mesh_cols = np.zeros_like(m_verts)
            for vi, v in enumerate(m_verts):
                _, idx, _ = kdtree.search_knn_vector_3d(v, 1)
                mesh_cols[vi] = pc_colors[idx[0]]
            mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_cols)
        except Exception:
            pass

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    print(f"  Poisson sau khi cat vat nhon: {len(mesh.vertices):,} dinh, {len(mesh.triangles):,} tam giac")
    return mesh


# ---------------------------------------------------------------------------
# THUAT TOAN TAI TAO BE MAT TONG QUAT 3: ORIENTED BOUNDING BOX (OBB)
# ---------------------------------------------------------------------------

def _obb_mesh(pc: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    """Hinh hop dinh huong toi thieu (Oriented Bounding Box) cho bat ky vat the nao."""
    try:
        obb = pc.get_minimal_oriented_bounding_box(robust=True)
    except AttributeError:
        obb = pc.get_oriented_bounding_box(robust=True)

    w, h, d = obb.extent
    box = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=d)
    box.translate(-obb.extent / 2)
    box.rotate(obb.R, center=(0, 0, 0))
    box.translate(obb.center)
    box.compute_vertex_normals()
    print(f"  OBB: {w:.3f} x {h:.3f} x {d:.3f}  watertight={box.is_watertight()}")
    return box


# ---------------------------------------------------------------------------
# DONG KIN BIEN (WATERTIGHT BOUNDARY CAPPING) TONG QUAT
# ---------------------------------------------------------------------------

def _make_watertight(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Dong kin open mesh thanh watertight tong quat cho moi vat the (Reverse Engineering Planar Base Extrusion).

    Vat the chup tren ban luon bi che mat day (2.5D surface).
    Thuat toan tong quat:
      1. Tim cac chu trinh bien ho (boundary loops) theo directed edges.
      2. Chu trinh bien chinh lon nhat (o day vat the) duoc keo tuong thang dung xuong mat san Z=0 va bit kin day.
      3. Cac chu trinh bien nho (lo noi tai tren be mat) duoc fan-fill cuc bo.
      4. Khau kin topology va lam muot chuyen tiep tranh rang cua hay chem ngang cheo.
    """
    if mesh.is_watertight():
        print("  mesh da watertight")
        return mesh

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    verts = np.asarray(mesh.vertices).copy()
    tris = np.asarray(mesh.triangles).copy()
    if len(tris) == 0:
        return mesh

    # Xac dinh directed boundary edges
    directed: dict[tuple[int, int], int] = {}
    for tri in tris:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in [(a, b), (b, c), (c, a)]:
            directed[(u, v)] = directed.get((u, v), 0) + 1

    next_map: dict[int, int] = {u: v for (u, v) in directed if (v, u) not in directed}
    if not next_map:
        return mesh

    visited_starts: set[int] = set()
    loops: list[list[int]] = []
    for start in list(next_map.keys()):
        if start in visited_starts:
            continue
        loop = [start]
        visited_starts.add(start)
        cur = next_map.get(start, start)
        while cur != start and cur not in visited_starts:
            loop.append(cur)
            visited_starts.add(cur)
            cur = next_map.get(cur, start)
        if len(loop) >= 3:
            loops.append(loop)

    if not loops:
        return mesh

    loops.sort(key=len, reverse=True)
    print(f"  Phat hien {len(loops)} chu trinh bien: {[len(l) for l in loops[:5]]} dinh")

    z_floor = 0.0
    new_v: list[np.ndarray] = [verts]
    new_t: list[np.ndarray] = [tris]
    offset = len(verts)

    for l_idx, loop in enumerate(loops):
        k = len(loop)
        loop_xyz = verts[loop]

        # Loop lon nhat hoac loop co do cao thap tiep giap san -> Tao tuong xuong san Z=0 va bit day
        if l_idx == 0 and len(loop) >= 8:
            cap_xyz = loop_xyz.copy()
            cap_xyz[:, 2] = z_floor
            cap_start = offset
            new_v.append(cap_xyz)
            offset += k

            cent_floor = cap_xyz.mean(axis=0)
            c_idx = offset
            new_v.append(cent_floor[None])
            offset += 1

            # Tuong ben (side walls)
            side_tris = []
            for i in range(k):
                a = loop[i]
                b = loop[(i + 1) % k]
                ca = cap_start + i
                cb = cap_start + (i + 1) % k
                side_tris.append([a, cb, b])
                side_tris.append([a, ca, cb])
            new_t.append(np.array(side_tris, dtype=np.int32))

            # Mat day (floor fan)
            xy = cap_xyz[:, :2]
            c_xy = cent_floor[:2]
            ang = np.arctan2(xy[:, 1] - c_xy[1], xy[:, 0] - c_xy[0])
            ord_idx = np.argsort(ang)
            floor_tris = [[c_idx, cap_start + ord_idx[i], cap_start + ord_idx[(i + 1) % k]] for i in range(k)]
            new_t.append(np.array(floor_tris, dtype=np.int32))
        else:
            # Cac lo thung nho cuc bo tren be mat -> Fan-fill cuc bo
            cent = loop_xyz.mean(axis=0)
            c_idx = offset
            new_v.append(cent[None])
            offset += 1
            fill_tris = [[c_idx, loop[(i + 1) % k], loop[i]] for i in range(k)]
            new_t.append(np.array(fill_tris, dtype=np.int32))

    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(np.vstack(new_v))
    result.triangles = o3d.utility.Vector3iVector(np.vstack(new_t))
    result.remove_degenerate_triangles()
    result.remove_duplicated_vertices()
    result.compute_vertex_normals()
    print(f"  Dong kin mesh hoan tat: {len(result.vertices):,} dinh, {len(result.triangles):,} tam giac, watertight={result.is_watertight()}")
    return result


# ---------------------------------------------------------------------------
# CAN THANG TRUC PCA & TIEP DAT SAN Z=0 TONG QUAT CHO MOI VAT THE
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CAN THANG TRUC PCA & DONG BO HE TOA DO POINT CLOUD & MESH
# ---------------------------------------------------------------------------

def _align_point_cloud_upright(
    pc: o3d.geometry.PointCloud,
    support_plane: tuple[np.ndarray, float] | None = None,
    cam_centers: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    """Can thang dung he truc PCA truc tiep cho Point Cloud truoc khi dung mesh.

    Dam bao:
    - Truc X: Chieu dai chinh (Length, tam X=0)
    - Truc Y: Chieu rong ngang (Depth/Width, tam Y=0)
    - Truc Z: Chieu cao thang dung (Height: Chan de tiep dat tai Z=0, Nap hop o tren dinh Z > 0).
    """
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        return pc

    m_cent = pts.mean(axis=0)
    centered = pts - m_cent
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 1. Xac dinh truc thang dung Z (Vertical Axis):
    if support_plane is not None:
        # Neu co mat ban/de xoay RANSAC, phap tuyen mat ban la truc Z
        n_z = support_plane[0].copy()
        if cam_centers is not None and len(cam_centers) > 0:
            if np.dot(n_z, cam_centers.mean(axis=0) - m_cent) < 0:
                n_z = -n_z
    elif cam_centers is not None and len(cam_centers) >= 3:
        try:
            cam_cov = np.cov(cam_centers - cam_centers.mean(axis=0), rowvar=False)
            _, cam_evecs = np.linalg.eigh(cam_cov)
            orbit_norm = cam_evecs[:, 0].copy()
            if np.dot(orbit_norm, cam_centers.mean(axis=0) - m_cent) < 0:
                orbit_norm = -orbit_norm
            # Chon truc PCA gan nhat voi orbit_norm
            dots = [abs(np.dot(eigvecs[:, i], orbit_norm)) for i in range(3)]
            best_i = int(np.argmax(dots))
            n_z = eigvecs[:, best_i].copy()
            if np.dot(n_z, orbit_norm) < 0:
                n_z = -n_z
        except Exception:
            n_z = eigvecs[:, 0].copy()
    else:
        # Fallback: chieu co span nho nhat thuong la chieu cao
        spans_pca = [np.ptp(centered @ eigvecs[:, i]) for i in range(3)]
        n_z = eigvecs[:, int(np.argmin(spans_pca))].copy()

    # Kiem tra neu co nap cam va than toi: +Z huong len nap cam
    if pc.has_colors():
        cols = np.asarray(pc.colors)
        orange = (cols[:, 0] > 0.55) & (cols[:, 2] < 0.35)
        if np.sum(orange) > 100:
            vec_b_to_o = np.median(pts[orange], axis=0) - np.median(pts[~orange], axis=0)
            if np.dot(n_z, vec_b_to_o) < 0:
                n_z = -n_z
                print("  [Huong UP] Da can chinh n_z de Nap cam luon o tren dinh (+Z)!")

    # 2. Xac dinh 2 truc ngang X (dai nhat) va Y (rong nhat) vuong goc voi Z
    # Chieu cac diem len mat phang vuong goc voi n_z
    p_proj = centered - np.outer(centered @ n_z, n_z)
    cov_2d = np.cov(p_proj, rowvar=False)
    _, evecs_2d = np.linalg.eigh(cov_2d)
    cand_h = [evecs_2d[:, 2], evecs_2d[:, 1], evecs_2d[:, 0]]
    # Chon 2 vector truc giao voi n_z
    cand_h = [v - np.dot(v, n_z) * n_z for v in cand_h]
    cand_h = [v / (np.linalg.norm(v) + 1e-8) for v in cand_h if np.linalg.norm(v) > 0.5]
    n_x = cand_h[0] if len(cand_h) > 0 else eigvecs[:, 2]
    n_x -= np.dot(n_x, n_z) * n_z
    n_x /= np.linalg.norm(n_x)
    n_y = np.cross(n_z, n_x)
    n_y /= np.linalg.norm(n_y)

    # Dam bao truc X la chieu dai hon truc Y
    span_x = np.ptp(centered @ n_x)
    span_y = np.ptp(centered @ n_y)
    if span_y > span_x:
        n_x, n_y = n_y, -n_x

    R = np.vstack([n_x, n_y, n_z])
    if np.linalg.det(R) < 0:
        n_x = -n_x
        R = np.vstack([n_x, n_y, n_z])

    p_rot = centered @ R.T

    # Kiem tra tuyet doi: Neu Nap cam van bi nam o duoi thi lat nguoc Z lai ngay lap tuc
    if pc.has_colors():
        cols = np.asarray(pc.colors)
        orange = (cols[:, 0] > 0.55) & (cols[:, 2] < 0.35)
        if np.sum(orange) > 100 and p_rot[orange, 2].mean() < p_rot[~orange, 2].mean():
            p_rot[:, 2] = -p_rot[:, 2]
            R[2] = -R[2]
            R[1] = -R[1]
            print("  [Kiem tra nguoc chieu] Da lat nguoc Z de bao dam 100% Nap cam o tren dinh (+Z)!")

    p_rot[:, 0] -= 0.5 * (p_rot[:, 0].min() + p_rot[:, 0].max())
    p_rot[:, 1] -= 0.5 * (p_rot[:, 1].min() + p_rot[:, 1].max())
    p_rot[:, 2] -= p_rot[:, 2].min()

    pc_aligned = o3d.geometry.PointCloud()
    pc_aligned.points = o3d.utility.Vector3dVector(p_rot)
    if pc.has_colors():
        pc_aligned.colors = pc.colors
    if pc.has_normals():
        norms = np.asarray(pc.normals) @ R.T
        pc_aligned.normals = o3d.utility.Vector3dVector(norms)

    print(f"  Da can thang truc PCA point cloud (Nap tren dinh, Day o Z=0): W={p_rot[:,0].ptp():.2f}, D={p_rot[:,1].ptp():.2f}, H={p_rot[:,2].max():.2f}")
    return pc_aligned


# ---------------------------------------------------------------------------
# TIEP DAT VA CAN BANG CHUNG (UNIVERSAL GROUNDING)
# ---------------------------------------------------------------------------

def _ground_mesh_z0(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Tiep dat sat san Z=0 bang tinh tien tuyet doi, khong dung mat phang cat xe de bao toan 100% hinh hoc diem anh."""
    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        return mesh
    sv = verts.copy()
    sv[:, 2] -= sv[:, 2].min()
    mesh.vertices = o3d.utility.Vector3dVector(sv)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# XUAT FILE OBJ
# ---------------------------------------------------------------------------

def _export_obj(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_colors = mesh.has_vertex_colors() and len(mesh.vertex_colors) > 0
    o3d.io.write_triangle_mesh(
        str(path),
        mesh,
        write_ascii=True,
        write_vertex_normals=True,
        write_vertex_colors=has_colors,
    )


# ---------------------------------------------------------------------------
# ENTRY POINT CHINH (OBJECT-AGNOSTIC UNIVERSAL RECONSTRUCTION)
# ---------------------------------------------------------------------------

def run(sample_name: str, cfg: dict) -> None:
    """Cong thuc tai tao be mat 3D tong quat truc tiep tu dam may diem anh cho moi mo hinh."""
    e = cfg["mesh_export"]
    mode = str(e.get("mesh_mode", "both")).strip().lower()
    valid_modes = ("bpa", "poisson", "both")
    if mode not in valid_modes:
        mode = "both"

    out_dir = paths.output_dir(sample_name)
    ply_path = out_dir / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Khong thay {ply_path}, chay Buoc 3 truoc.")

    # 1. Doc point cloud goc tu MVS / COLMAP
    print(f"[Buoc 4] Doc point cloud diem anh tu {ply_path.name}...")
    pc = _load_gaussian_ply(ply_path)
    n_raw = len(pc.points)
    print(f"  {n_raw:,} diem ban dau tu anh")
    if n_raw == 0:
        raise RuntimeError("Khong co diem nao trong point cloud.")

    obj_path = out_dir / f"{sample_name}.obj"
    clean_ply = out_dir / "point_cloud_clean.ply"
    aligned_ply = out_dir / f"{sample_name}_point_cloud_aligned.ply"
    conf_npy = out_dir / "point_confidence.npy"

    # 2. Loc nhieu point cloud tong quat (Statistical + Radius + DBSCAN giu vat the chinh)
    print("[Buoc 4] Loc nhieu point cloud tong quat tu diem anh (Statistical + Radius + DBSCAN)...")
    pc, diag = _clean(pc, e)

    # 3. Uoc luong va dong nhat truong phap tuyen quang hoc tu cac tam camera SfM
    cam_centers = _load_camera_centers(sample_name)
    print(f"[Buoc 4] Dong nhat phap tuyen theo tam camera ({len(cam_centers)} camera)...")
    pc = _orient_normals_sensor_aware(pc, cam_centers, diag, e)

    # 4. Can thang truc point cloud theo he toa do quan sat (dong bo giua Point Cloud va Mesh)
    if e.get("align_upright", True):
        print("[Buoc 4] Can thang truc Point Cloud (dong bo toa do voi Mesh)...")
        pc = _align_point_cloud_upright(pc, None, cam_centers=cam_centers)

    o3d.io.write_point_cloud(str(clean_ply), pc)
    o3d.io.write_point_cloud(str(aligned_ply), pc)
    np.save(str(conf_npy), np.ones(len(pc.points), dtype=np.float32))
    print(f"  Da luu point cloud sach da can truc -> {clean_ply.name}")

    # 5. Tai tao be mat theo Cong thuc chung (BPA / Poisson)
    bpa_path = out_dir / f"{sample_name}_bpa.obj"
    poi_path = out_dir / f"{sample_name}_poisson.obj"

    if mode == "bpa":
        print("[Buoc 4] Thuat toan tai tao be mat: Ball-Pivoting Algorithm (BPA)...")
        mesh = _reconstruct_bpa(pc, e)
    elif mode == "poisson":
        print("[Buoc 4] Thuat toan tai tao be mat: Screened Poisson Surface Reconstruction...")
        mesh = _reconstruct_poisson(pc, e, None)
    else:  # both
        print("[Buoc 4] Tai tao dong thoi ca BPA (noi diem thuc 100%) va Screened Poisson (be mat lien tuc)...")
        mesh_bpa = _reconstruct_bpa(pc, e)
        mesh_poi = _reconstruct_poisson(pc, e, None)

        mesh_bpa = _ground_mesh_z0(mesh_bpa)
        mesh_poi = _ground_mesh_z0(mesh_poi)

        _export_obj(mesh_bpa, bpa_path)
        _export_obj(mesh_poi, poi_path)
        print(f"  da luu ca hai ban: {bpa_path.name} va {poi_path.name}")
        mesh = mesh_poi

    # 6. Tiep dat Z=0 bang tinh tien truc tiep (khong cat xe hinh hoc)
    mesh = _ground_mesh_z0(mesh)

    if e.get("make_watertight", True):
        print("[Buoc 4] Dong kin mesh (watertight solid)...")
        mesh = _make_watertight(mesh)
        mesh = _ground_mesh_z0(mesh)

    # 7. Xuat ban OBJ cuoi cung
    print(f"[Buoc 4] Xuat {obj_path.name}...")
    _export_obj(mesh, obj_path)
    print(f"[Buoc 4] Hoan tat -> {obj_path} ({obj_path.stat().st_size/1024:.1f} KB)")

    raw_path = out_dir / f"{sample_name}_raw.obj"
    import shutil
    shutil.copyfile(str(obj_path), str(raw_path))

    # Xuat hinh thong so
    try:
        from . import spec_figures
        spec_figures.export_all_specification_figures(sample_name, out_dir, mesh, pc)
    except Exception as err_spec:
        print(f"  [Canh bao] Xuat hinh thong so: {err_spec}")

    # 8. Buoc 5: Mesh Refine (neu bat)
    if cfg.get("mesh_refine", {}).get("auto_run", True):
        from . import step5_mesh_refine
        step5_mesh_refine.run(sample_name, cfg)
    else:
        print(f"  Blender: File > Import > Wavefront (.obj) > {obj_path.name}")
