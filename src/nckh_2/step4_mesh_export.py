"""Buoc 4: Don sach point cloud (Open3D) + Dung mesh 3D tong quat -> .obj.

Pipeline tong quat (khong hard-code hinh dang vat the):
  load -> clean -> tach mat ban (RANSAC) -> dinh huong phap tuyen -> can truc
       -> BPA / Poisson -> cat day tai mat ban + bit kin -> xuat .obj

Prior duy nhat: vat duoc dat tren ban/ban xoay nen co DAY PHANG nam tren mat ban.
Mat ban duoc RANSAC tim ra tu chinh du lieu, nen khong phu thuoc loai vat the.

mesh_mode = bpa     : Ball-Pivoting (bam sat diem that)
mesh_mode = poisson : Screened Poisson (be mat lien tuc, kin)
mesh_mode = both    : xuat ca hai, file chinh = Poisson
"""
from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

from . import paths

try:
    from scipy.spatial import cKDTree
except Exception:  # scipy khong bat buoc
    cKDTree = None


# ---------------------------------------------------------------------------
# TIEN ICH
# ---------------------------------------------------------------------------

def _nn_query(ref: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lang gieng gan nhat (k=1) cua moi diem query trong ref. Tra ve (dist, idx)."""
    if cKDTree is not None:
        d, i = cKDTree(ref).query(query, k=1)
        return np.asarray(d), np.asarray(i)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref))
    tree = o3d.geometry.KDTreeFlann(pcd)
    idx = np.zeros(len(query), dtype=np.int64)
    dist = np.zeros(len(query), dtype=np.float64)
    for n, q in enumerate(query):
        _, ii, dd = tree.search_knn_vector_3d(q, 1)
        idx[n] = ii[0]
        dist[n] = np.sqrt(dd[0])
    return dist, idx


def _robust_diag(pts: np.ndarray) -> float:
    p5 = np.percentile(pts, 5, axis=0)
    p95 = np.percentile(pts, 95, axis=0)
    diag = float(np.linalg.norm(p95 - p5))
    if diag < 1e-6:
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    return diag


# ---------------------------------------------------------------------------
# DOC POINT CLOUD
# ---------------------------------------------------------------------------

def _load_gaussian_ply(path: Path) -> o3d.geometry.PointCloud:
    """Doc .ply tu ns-export gaussian-splat hoac colmap stereo_fusion."""
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = v.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    # Loc opacity truoc (chi gaussian-splat) de cac mang sau cung kich thuoc
    keep = np.ones(len(xyz), dtype=bool)
    if "opacity" in names:
        opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
        keep = opacity > 0.2
        print(f"  loc opacity > 0.2: giu {int(keep.sum()):,}/{len(xyz):,} diem")

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz[keep])

    if all(f"f_dc_{i}" in names for i in range(3)):
        C0 = 0.28209479177387814
        dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float64)
        rgb = np.clip(0.5 + C0 * dc, 0.0, 1.0)
        pc.colors = o3d.utility.Vector3dVector(rgb[keep])
    elif all(c in names for c in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float64) / 255.0
        pc.colors = o3d.utility.Vector3dVector(rgb[keep])

    # Phap tuyen: chi tin khi GAN NHU TAT CA hop le (gaussian-splat thuong ghi nx,ny,nz = 0)
    if all(n in names for n in ("nx", "ny", "nz")):
        norms = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float64)[keep]
        lens = np.linalg.norm(norms, axis=1)
        valid = lens > 1e-4
        if len(valid) and valid.mean() > 0.99:
            norms[valid] /= lens[valid, None]
            norms[~valid] = [0.0, 0.0, 1.0]
            pc.normals = o3d.utility.Vector3dVector(norms)
        else:
            print("  phap tuyen trong ply khong hop le -> se uoc luong lai")

    return pc


# ---------------------------------------------------------------------------
# DOC TOA DO TAM CAMERA
# ---------------------------------------------------------------------------

def _read_colmap_images_bin(bin_file: Path) -> np.ndarray:
    centers = []
    with open(bin_file, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            f.read(4)                                         # image_id
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            f.read(4)                                         # camera_id
            while f.read(1) != b"\x00":                       # ten anh
                pass
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(n2d * 24, 1)
            R = np.array([
                [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
            ])
            centers.append(-R.T @ np.array([tx, ty, tz]))     # C = -R^T t
    return np.array(centers, dtype=np.float64)


def _load_camera_centers(sample_name: str, pc_center: np.ndarray) -> np.ndarray:
    """Lay tam camera, tu chon he toa do (goc / da normalize cua nerfstudio) khop point cloud.

    ns-export xuat point cloud trong he da normalize (dataparser_transforms.json),
    con transforms.json o he goc -> thu ca hai, chon he nao co tam camera gan vat hon.
    """
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
                # Thu them ban da ap dataparser_transforms.json
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
                    c = _read_colmap_images_bin(bf)
                    if len(c):
                        cands.append(c)
                        break
                except Exception:
                    pass

    if not cands:
        return np.empty((0, 3), dtype=np.float64)

    best = min(cands, key=lambda c: np.linalg.norm(c.mean(axis=0) - pc_center))
    if len(cands) > 1:
        print(f"  chon he toa do camera #{cands.index(best)} (khop point cloud hon)")
    return best


# ---------------------------------------------------------------------------
# LOC NHIEU
# ---------------------------------------------------------------------------

def _keep_significant_clusters(pc, diag: float, e: dict):
    """DBSCAN: giu cum chinh + cac cum du lon, bo diem troi noi."""
    if len(pc.points) < 100:
        return pc
    min_pts = int(e.get("cluster_min_points", 40))
    eps = max(float(e.get("cluster_eps_frac", 0.02)) * diag, 1e-6)

    labels = np.array(pc.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False))
    if len(labels) == 0 or not np.any(labels >= 0):
        return pc

    counts = np.bincount(labels[labels >= 0])
    largest_count = counts.max()
    ratio = float(e.get("cluster_min_ratio", 0.10))
    thr = max(min_pts, int(largest_count * ratio))
    keep_clusters = [c for c, cnt in enumerate(counts) if cnt >= thr]
    keep = np.nonzero(np.isin(labels, keep_clusters))[0]
    print(f"  DBSCAN (eps={eps:.4f}): giu {len(keep_clusters)}/{len(counts)} cum "
          f"({len(keep):,}/{len(labels):,} diem)")
    return pc.select_by_index(keep)


def _clean(pc, e: dict):
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        return pc, 0.0
    diag = _robust_diag(pts)
    print(f"  duong cheo (robust) = {diag:.4f}")

    before = len(pc.points)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=e["stat_nb"], std_ratio=e["stat_ratio"])
    print(f"  loc thong ke: {before:,} -> {len(pc.points):,}")
    if len(pc.points) == 0:
        raise RuntimeError("Loc thong ke xoa het diem. Giam stat_ratio trong config.")

    rad = e["rad_radius_frac"] * diag
    before = len(pc.points)
    pc, _ = pc.remove_radius_outlier(nb_points=e["rad_nb"], radius=rad)
    print(f"  loc ban kinh: {before:,} -> {len(pc.points):,}")
    if len(pc.points) == 0:
        raise RuntimeError("Loc ban kinh xoa het diem. Tang rad_radius_frac trong config.")

    pc = _keep_significant_clusters(pc, diag, e)

    if e.get("voxel_frac", 0):
        vox = e["voxel_frac"] * diag
        before = len(pc.points)
        pc = pc.voxel_down_sample(vox)
        print(f"  voxel downsample: {before:,} -> {len(pc.points):,}")
    return pc, diag


# ---------------------------------------------------------------------------
# TACH MAT BAN (RANSAC) - tra ve mat phang (n, d) voi n huong ve phia vat
# ---------------------------------------------------------------------------

def _orbit_normal(cam_centers: np.ndarray) -> np.ndarray | None:
    if cam_centers is None or len(cam_centers) < 3:
        return None
    cov = np.cov(cam_centers - cam_centers.mean(axis=0), rowvar=False)
    _, ev = np.linalg.eigh(cov)
    return ev[:, 0].copy()          # phap tuyen mat phang quy dao camera


def _remove_plane_ransac(pc, diag: float, e: dict, cam_centers: np.ndarray | None = None):
    """Tim & loai mat ban/ban xoay. Tra ve (pc, (normal, d)) voi n.x + d = 0, n huong len phia vat."""
    if not e.get("remove_table_plane", True) or len(pc.points) < 100:
        return pc, None

    dist_thresh = max(float(e.get("plane_dist_threshold_frac", 0.012)) * diag, 1e-6)
    try:
        model, inliers = pc.segment_plane(distance_threshold=dist_thresh, ransac_n=3,
                                          num_iterations=1000)
    except Exception:
        return pc, None

    ratio = len(inliers) / max(len(pc.points), 1)
    if not (0.05 <= ratio <= 0.65):
        return pc, None

    a, b, c, d = model
    n = np.array([a, b, c], dtype=np.float64)
    nl = np.linalg.norm(n)
    if nl < 1e-8:
        return pc, None
    n /= nl
    d /= nl

    pts = np.asarray(pc.points)
    dists = pts @ n + d
    pos = np.mean(dists > dist_thresh)
    neg = np.mean(dists < -dist_thresh)
    if max(pos, neg) <= 0.60:
        return pc, None                                   # vat nam 2 ben -> khong phai mat ban

    # Chong nham mat ben cua vat voi mat ban: mat ban phai ~vuong goc truc quy dao camera
    orbit = _orbit_normal(cam_centers)
    if orbit is not None and abs(float(n @ orbit)) < 0.8:
        print("  RANSAC Plane: bo qua (mat phang khong song song quy dao camera)")
        return pc, None

    # n huong ve phia vat (phia co nhieu diem hon)
    if neg > pos:
        n, d = -n, -d

    print(f"  RANSAC Plane: tach {len(inliers):,}/{len(pc.points):,} diem ({ratio:.1%}) mat ban")
    return pc.select_by_index(inliers, invert=True), (n, float(d))


# ---------------------------------------------------------------------------
# PHAP TUYEN
# ---------------------------------------------------------------------------

def _orient_normals(pc, cam_centers: np.ndarray, diag: float, e: dict):
    """Uoc luong (neu chua co) va dinh huong phap tuyen ra ngoai theo tia nhin camera."""
    pts = np.asarray(pc.points)
    n_pts = len(pts)
    if n_pts == 0:
        return pc

    if pc.has_normals() and len(pc.normals) == n_pts:
        lens = np.linalg.norm(np.asarray(pc.normals), axis=1)
        if np.mean(lens > 0.5) > 0.99:
            print(f"  giu phap tuyen co san ({n_pts:,} vector)")
            return pc

    rad = float(e.get("normal_radius_frac", 0.02)) * diag
    nn = int(e.get("normal_nn", 30))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=nn))
    norms = np.asarray(pc.normals).copy()

    if len(cam_centers) >= 3:
        _, idx = _nn_query(cam_centers, pts)
        view = cam_centers[idx] - pts
        flip = np.einsum("ij,ij->i", norms, view) < 0
        print(f"  dinh huong phap tuyen theo {len(cam_centers)} camera (lat {int(flip.sum()):,})")
    else:
        view = pts - pts.mean(axis=0)                     # fallback: huong ra ngoai tam vat
        flip = np.einsum("ij,ij->i", norms, view) < 0
        print("  dinh huong phap tuyen ra ngoai tam vat (khong co camera)")
    norms[flip] *= -1
    pc.normals = o3d.utility.Vector3dVector(norms)
    return pc


# ---------------------------------------------------------------------------
# CAN THANG TRUC (Z-up, mat ban = Z=0)
# ---------------------------------------------------------------------------

def _align_point_cloud_upright(pc, support_plane=None, cam_centers=None):
    """Z = phap tuyen mat ban (hoac truc quy dao camera); X = chieu ngang dai nhat; mat ban tai Z=0."""
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        return pc
    m_cent = pts.mean(axis=0)
    centered = pts - m_cent

    # 1. Truc Z
    if support_plane is not None:
        n_z = support_plane[0].copy()
    else:
        n_z = _orbit_normal(cam_centers)
        if n_z is not None:
            if np.dot(n_z, cam_centers.mean(axis=0) - m_cent) < 0:
                n_z = -n_z
        else:
            _, ev = np.linalg.eigh(np.cov(centered, rowvar=False))
            spans = [np.ptp(centered @ ev[:, i]) for i in range(3)]
            n_z = ev[:, int(np.argmin(spans))].copy()
    n_z /= np.linalg.norm(n_z)

    # 2. Truc X, Y nam ngang
    proj = centered - np.outer(centered @ n_z, n_z)
    _, ev2 = np.linalg.eigh(np.cov(proj, rowvar=False))
    n_x = ev2[:, 2] - np.dot(ev2[:, 2], n_z) * n_z
    n_x /= np.linalg.norm(n_x)
    n_y = np.cross(n_z, n_x)
    n_y /= np.linalg.norm(n_y)

    R = np.vstack([n_x, n_y, n_z])                         # det = +1
    p_rot = centered @ R.T

    # 3. Can tam XY, dat Z=0
    p_rot[:, 0] -= 0.5 * (p_rot[:, 0].min() + p_rot[:, 0].max())
    p_rot[:, 1] -= 0.5 * (p_rot[:, 1].min() + p_rot[:, 1].max())
    if support_plane is not None:
        n, d = support_plane
        z_plane = -(float(n @ m_cent) + d)                 # cao do mat ban trong he moi
        p_rot[:, 2] -= z_plane
    else:
        p_rot[:, 2] -= p_rot[:, 2].min()

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(p_rot)
    if pc.has_colors():
        out.colors = pc.colors
    if pc.has_normals():
        out.normals = o3d.utility.Vector3dVector(np.asarray(pc.normals) @ R.T)

    print(f"  can truc: X={np.ptp(p_rot[:,0]):.3f} Y={np.ptp(p_rot[:,1]):.3f} "
          f"Z=[{p_rot[:,2].min():.3f}, {p_rot[:,2].max():.3f}] "
          f"({'theo mat ban' if support_plane is not None else 'theo quy dao/PCA'})")
    return out


# ---------------------------------------------------------------------------
# TAI TAO BE MAT 1: BPA
# ---------------------------------------------------------------------------

def _clean_mesh_topology(mesh, keep_ratio: float = 0.02):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    tri_clusters, n_tris, _ = mesh.cluster_connected_triangles()
    tri_clusters, n_tris = np.asarray(tri_clusters), np.asarray(n_tris)
    if len(n_tris) > 0:
        thr = max(20, int(n_tris.max() * keep_ratio))
        keep = np.nonzero(n_tris >= thr)[0]
        mesh.remove_triangles_by_mask(~np.isin(tri_clusters, keep))
        mesh.remove_unreferenced_vertices()
    return mesh


def _reconstruct_bpa(pc, diag: float, e: dict):
    """Ball-Pivoting da ban kinh (ban kinh theo khoang cach lang gieng)."""
    if len(pc.points) < 10:
        return o3d.geometry.TriangleMesh()

    pcd = pc
    if len(pc.points) > 250000:
        vox = float(e.get("bpa_voxel_frac", 0.004)) * diag
        if vox > 0:
            pcd = pc.voxel_down_sample(vox)
            print(f"  BPA voxel downsample: {len(pc.points):,} -> {len(pcd.points):,}")
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(e.get("normal_radius_frac", 0.02)) * diag, max_nn=30))

    avg = float(np.mean(pcd.compute_nearest_neighbor_distance()))
    radii = [avg * float(e.get(k, m)) for k, m in
             (("bpa_r1_mult", 1.2), ("bpa_r2_mult", 2.5), ("bpa_r3_mult", 5.0))]
    print(f"  BPA: avg_dist={avg:.5f} radii={[round(r, 5) for r in radii]}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    mesh = _clean_mesh_topology(mesh)
    mesh.compute_vertex_normals()
    print(f"  BPA: {len(mesh.vertices):,} dinh, {len(mesh.triangles):,} tam giac")
    return mesh


# ---------------------------------------------------------------------------
# TAI TAO BE MAT 2: SCREENED POISSON
# ---------------------------------------------------------------------------

def _reconstruct_poisson(pc, diag: float, e: dict):
    pcd = o3d.geometry.PointCloud(pc)
    if not pcd.has_normals() or len(pcd.normals) != len(pcd.points):
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(e.get("normal_radius_frac", 0.02)) * diag, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)

    depth = int(e.get("poisson_depth", 8))
    print(f"  Poisson depth={depth}...")
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=bool(e.get("poisson_linear_fit", True)))
    dens = np.asarray(dens)
    print(f"  Poisson raw: {len(mesh.vertices):,} dinh, {len(mesh.triangles):,} tam giac")

    # Cat vung mat do thap (bong bong ngoai suy) - mac dinh 4 phan tram
    pct = float(e.get("poisson_density_trim_percentile", 4.0))
    if pct > 0:
        mesh.remove_vertices_by_mask(dens < np.percentile(dens, pct))

    # Crop theo bbox point cloud + pad
    bbox = pcd.get_axis_aligned_bounding_box()
    pad = float(e.get("poisson_hull_pad_frac", 0.01)) * float(np.linalg.norm(bbox.get_extent()))
    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(bbox.min_bound - pad, bbox.max_bound + pad))

    mesh = _clean_mesh_topology(mesh, keep_ratio=1.0)      # chi giu cum lon nhat

    s_iter = int(e.get("smooth_iter", 12))
    if s_iter > 0 and len(mesh.vertices) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=s_iter,
                                         lambda_filter=float(e.get("smooth_lambda", 0.5)),
                                         mu=float(e.get("smooth_mu", -0.53)))

    # Chuyen mau
    if pc.has_colors() and len(mesh.vertices) > 0:
        _, idx = _nn_query(np.asarray(pc.points), np.asarray(mesh.vertices))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pc.colors)[idx])

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    print(f"  Poisson cuoi: {len(mesh.vertices):,} dinh, {len(mesh.triangles):,} tam giac")
    return mesh


# ---------------------------------------------------------------------------
# DONG KIN MESH
# ---------------------------------------------------------------------------

def _make_watertight(mesh):
    """Bit cac chu trinh bien (PyMeshLab neu co, khong thi fan-fill tai centroid)."""
    if len(mesh.triangles) == 0 or mesh.is_watertight():
        return mesh

    has_col = mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices)

    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
                                   face_matrix=np.asarray(mesh.triangles, dtype=np.int32)))
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_close_holes(maxholesize=10000)
        res = ms.current_mesh()
        out = o3d.geometry.TriangleMesh()
        out.vertices = o3d.utility.Vector3dVector(res.vertex_matrix())
        out.triangles = o3d.utility.Vector3iVector(res.face_matrix())
        out.remove_degenerate_triangles()
        out.compute_vertex_normals()
        print(f"  PyMeshLab close holes: watertight={out.is_watertight()}")
        return out
    except Exception as err:
        print(f"  PyMeshLab khong kha dung ({err}) -> fan-fill")

    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    verts = np.asarray(mesh.vertices).copy()
    tris = np.asarray(mesh.triangles).copy()
    cols = np.asarray(mesh.vertex_colors).copy() if has_col else None

    directed: dict[tuple[int, int], int] = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            directed[(int(u), int(v))] = directed.get((int(u), int(v)), 0) + 1
    nxt = {u: v for (u, v) in directed if (v, u) not in directed}
    if not nxt:
        return mesh

    loops, remaining = [], dict(nxt)
    while remaining:
        start = next(iter(remaining))
        cur = remaining.pop(start)
        loop = [start]
        while cur != start:
            loop.append(cur)
            if cur not in remaining:
                break
            cur = remaining.pop(cur)
        if cur == start and len(loop) >= 3:
            loops.append(loop)
    if not loops:
        return mesh

    print(f"  fan-fill {len(loops)} chu trinh bien: {[len(l) for l in loops]} dinh")
    new_v, new_t, new_c = [verts], [tris], ([cols] if has_col else None)
    offset = len(verts)
    for loop in loops:
        k = len(loop)
        new_v.append(verts[loop].mean(axis=0)[None])
        if has_col:
            new_c.append(cols[loop].mean(axis=0)[None])
        new_t.append(np.array([[offset, loop[(i + 1) % k], loop[i]] for i in range(k)], dtype=np.int32))
        offset += 1

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(np.vstack(new_v))
    out.triangles = o3d.utility.Vector3iVector(np.vstack(new_t))
    if has_col:
        out.vertex_colors = o3d.utility.Vector3dVector(np.vstack(new_c))
    out.remove_degenerate_triangles()
    out.compute_vertex_normals()
    print(f"  fan-fill: watertight={out.is_watertight()}")
    return out


# ---------------------------------------------------------------------------
# DAY PHANG TREN MAT BAN
# ---------------------------------------------------------------------------

def _cap_bottom(mesh, diag: float, e: dict):
    """Mat ban = Z=0. Ep cac dinh sat/duoi san ve Z=0, bo tam giac nam han tren san, roi bit day."""
    if len(mesh.vertices) == 0:
        return mesh
    tol = float(e.get("floor_snap_frac", 0.02)) * diag
    v = np.asarray(mesh.vertices).copy()
    v[v[:, 2] < tol, 2] = 0.0
    mesh.vertices = o3d.utility.Vector3dVector(v)

    tris = np.asarray(mesh.triangles)
    on_floor = np.all(v[tris, 2] <= 0.0, axis=1)           # tam giac nam bet tren san
    mesh.remove_triangles_by_mask(on_floor)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    return _make_watertight(mesh)


def _ground_mesh_z0(mesh):
    v = np.asarray(mesh.vertices)
    if len(v) == 0:
        return mesh
    v = v.copy()
    v[:, 2] -= v[:, 2].min()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def _finalize_mesh(mesh, diag: float, e: dict, has_plane: bool):
    if has_plane:
        mesh = _cap_bottom(mesh, diag, e)
    else:
        mesh = _ground_mesh_z0(mesh)
        if e.get("make_watertight", True):
            mesh = _make_watertight(mesh)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    print(f"  mesh cuoi: {len(mesh.vertices):,} dinh, {len(mesh.triangles):,} tam giac "
          f"watertight={mesh.is_watertight()}")
    return mesh


# ---------------------------------------------------------------------------
# XUAT OBJ
# ---------------------------------------------------------------------------

def _export_obj(mesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_colors = mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices)
    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=True,
                               write_vertex_normals=True, write_vertex_colors=has_colors)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run(sample_name: str, cfg: dict) -> None:
    e = cfg["mesh_export"]
    mode = str(e.get("mesh_mode", "both")).strip().lower()
    if mode not in ("bpa", "poisson", "both"):
        print(f"  mesh_mode '{mode}' khong hop le -> dung 'both'")
        mode = "both"

    out_dir = paths.output_dir(sample_name)
    ply_path = out_dir / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Khong thay {ply_path}, chay Buoc 3 truoc.")

    obj_path = out_dir / f"{sample_name}.obj"
    clean_ply = out_dir / "point_cloud_clean.ply"
    aligned_ply = out_dir / f"{sample_name}_point_cloud_aligned.ply"
    bpa_path = out_dir / f"{sample_name}_bpa.obj"
    poi_path = out_dir / f"{sample_name}_poisson.obj"

    # 1. Doc
    print(f"[Buoc 4] Doc point cloud tu {ply_path.name}...")
    pc = _load_gaussian_ply(ply_path)
    print(f"  {len(pc.points):,} diem ban dau")
    if len(pc.points) == 0:
        raise RuntimeError("Khong co diem nao trong point cloud.")

    # 2. Loc nhieu
    print("[Buoc 4] Loc nhieu (Statistical + Radius + DBSCAN)...")
    pc, diag = _clean(pc, e)

    # 3. Camera + tach mat ban
    cam_centers = _load_camera_centers(sample_name, np.asarray(pc.points).mean(axis=0))
    print(f"[Buoc 4] Camera: {len(cam_centers)}")
    print("[Buoc 4] Tach mat ban (RANSAC)...")
    pc, support_plane = _remove_plane_ransac(pc, diag, e, cam_centers)

    # 4. Phap tuyen
    print("[Buoc 4] Dinh huong phap tuyen...")
    pc = _orient_normals(pc, cam_centers, diag, e)

    # 5. Can truc (mat ban -> Z=0)
    if e.get("align_upright", True):
        print("[Buoc 4] Can thang truc point cloud...")
        pc = _align_point_cloud_upright(pc, support_plane, cam_centers)
    else:
        support_plane = None                               # khong can truc -> khong co mat Z=0

    o3d.io.write_point_cloud(str(clean_ply), pc)
    o3d.io.write_point_cloud(str(aligned_ply), pc)
    print(f"  da luu {clean_ply.name}")
    has_plane = support_plane is not None

    # 6. Tai tao be mat
    if mode == "bpa":
        print("[Buoc 4] BPA...")
        mesh = _finalize_mesh(_reconstruct_bpa(pc, diag, e), diag, e, has_plane)
    elif mode == "poisson":
        print("[Buoc 4] Poisson...")
        mesh = _finalize_mesh(_reconstruct_poisson(pc, diag, e), diag, e, has_plane)
    else:
        print("[Buoc 4] BPA + Poisson...")
        mesh_bpa = _finalize_mesh(_reconstruct_bpa(pc, diag, e), diag, e, has_plane)
        mesh_poi = _finalize_mesh(_reconstruct_poisson(pc, diag, e), diag, e, has_plane)
        _export_obj(mesh_bpa, bpa_path)
        _export_obj(mesh_poi, poi_path)
        print(f"  da luu {bpa_path.name} va {poi_path.name}")
        mesh = mesh_poi

    # 7. Xuat
    print(f"[Buoc 4] Xuat {obj_path.name}...")
    _export_obj(mesh, obj_path)
    print(f"[Buoc 4] Hoan tat -> {obj_path} ({obj_path.stat().st_size/1024:.1f} KB)")
    shutil.copyfile(str(obj_path), str(out_dir / f"{sample_name}_raw.obj"))

    try:
        from . import spec_figures
        spec_figures.export_all_specification_figures(sample_name, out_dir, mesh, pc)
    except Exception as err:
        print(f"  [Canh bao] Xuat hinh thong so: {err}")

    # 8. Buoc 5
    if cfg.get("mesh_refine", {}).get("auto_run", True):
        from . import step5_mesh_refine
        step5_mesh_refine.run(sample_name, cfg)
    else:
        print(f"  Blender: File > Import > Wavefront (.obj) > {obj_path.name}")