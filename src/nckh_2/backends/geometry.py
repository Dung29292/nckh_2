"""Cac ham xu ly hinh hoc Open3D / SciPy (Point Cloud & Mesh) tong quat cho moi file .obj."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from plyfile import PlyData

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


# ---------------------------------------------------------------------------
# DIAGNOSTICS & ATTRIBUTE-SAFE HELPERS
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MeshDiagnostics:
    """Thong so kiem dinh hinh hoc cua mesh truoc/sau khi xu ly."""

    num_vertices: int
    num_triangles: int
    num_components: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    bbox_extents: tuple[float, float, float]
    surface_area: float
    is_watertight: bool
    has_vertex_colors: bool
    has_vertex_normals: bool
    has_triangle_uvs: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _spatial_connected_components(
    verts: np.ndarray,
    tris: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phan cum tam giac lien thong dua tren toa do khong gian (weld dinh trung vi tri).

    Tranh tinh trang cac file .obj co UV seam / split normals bi tach thanh hang tram
    manh 1-tam-giac gia tao khi dung Open3D cluster_connected_triangles() truc tiep.
    Tra ve (tri_cluster_ids, tri_counts_per_comp, area_per_comp).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_t = len(tris)
    if n_t == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    # Dien tich tung tam giac
    e1 = verts[tris[:, 1]] - verts[tris[:, 0]]
    e2 = verts[tris[:, 2]] - verts[tris[:, 0]]
    tri_areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    # Weld dinh theo toa do khong gian (do chinh xac tuong doi 1e-7 * diag)
    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    quant = max(diag * 1e-7, 1e-9)
    q_verts = np.round(verts / quant).astype(np.int64)
    _, inv_v = np.unique(q_verts, axis=0, return_inverse=True)
    w_tris = inv_v[tris]
    n_wv = int(inv_v.max()) + 1

    # Do thi lien thong dinh-tam giac (vertex-sharing components tren luoi da weld)
    v_flat = w_tris.ravel()
    f_flat = np.repeat(np.arange(n_t, dtype=np.int32), 3)
    vf = coo_matrix((np.ones(len(v_flat), dtype=np.int8), (v_flat, f_flat)), shape=(n_wv, n_t)).tocsr()
    ff = vf.T @ vf
    n_comp, labels = connected_components(ff, directed=False)
    labels = labels.astype(np.int32)

    tri_counts = np.bincount(labels, minlength=n_comp).astype(np.int64)
    comp_areas = np.bincount(labels, weights=tri_areas, minlength=n_comp).astype(np.float64)
    return labels, tri_counts, comp_areas


def compute_mesh_diagnostics(mesh: o3d.geometry.TriangleMesh) -> MeshDiagnostics:
    """Tinh toan nhanh cac chi so hinh hoc & thuoc tinh cua mesh (an toan voi mesh rong)."""
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)
    if n_v == 0:
        zeros = (0.0, 0.0, 0.0)
        return MeshDiagnostics(
            num_vertices=0,
            num_triangles=0,
            num_components=0,
            bbox_min=zeros,
            bbox_max=zeros,
            bbox_extents=zeros,
            surface_area=0.0,
            is_watertight=False,
            has_vertex_colors=False,
            has_vertex_normals=False,
            has_triangle_uvs=False,
        )

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    ext = bmax - bmin

    if n_t > 0:
        tris = np.asarray(mesh.triangles, dtype=np.int32)
        _, n_tris_per_comp, areas_per_comp = _spatial_connected_components(verts, tris)
        num_comp = len(n_tris_per_comp)
        area = float(np.sum(areas_per_comp))
        watertight = bool(mesh.is_watertight())
    else:
        num_comp = 0
        area = 0.0
        watertight = False

    has_colors = bool(mesh.has_vertex_colors() and len(mesh.vertex_colors) == n_v)
    has_normals = bool(mesh.has_vertex_normals() and len(mesh.vertex_normals) == n_v)
    has_uvs = bool(mesh.has_triangle_uvs() and len(mesh.triangle_uvs) == 3 * n_t and n_t > 0)

    return MeshDiagnostics(
        num_vertices=n_v,
        num_triangles=n_t,
        num_components=num_comp,
        bbox_min=tuple(round(float(x), 6) for x in bmin),
        bbox_max=tuple(round(float(x), 6) for x in bmax),
        bbox_extents=tuple(round(float(x), 6) for x in ext),
        surface_area=round(area, 6),
        is_watertight=watertight,
        has_vertex_colors=has_colors,
        has_vertex_normals=has_normals,
        has_triangle_uvs=has_uvs,
    )


def remove_triangles_preserving_attributes(
    mesh: o3d.geometry.TriangleMesh,
    remove_mask: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Xoa tam giac theo mask dong thoi dong bo hoa triangle_uvs va triangle_material_ids."""
    n_t = len(mesh.triangles)
    if n_t == 0:
        return mesh
    remove_mask = np.asarray(remove_mask, dtype=bool)
    if len(remove_mask) != n_t or not np.any(remove_mask):
        return mesh

    keep_mask = ~remove_mask

    uvs_kept = None
    if mesh.has_triangle_uvs() and len(mesh.triangle_uvs) == 3 * n_t:
        uvs = np.asarray(mesh.triangle_uvs, dtype=np.float64).reshape(n_t, 3, 2)
        uvs_kept = uvs[keep_mask].reshape(-1, 2)

    mat_ids_kept = None
    if mesh.has_triangle_material_ids() and len(mesh.triangle_material_ids) == n_t:
        mat_ids = np.asarray(mesh.triangle_material_ids, dtype=np.int32)
        mat_ids_kept = mat_ids[keep_mask]

    mesh.remove_triangles_by_mask(remove_mask)
    if uvs_kept is not None:
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs_kept)
    if mat_ids_kept is not None:
        mesh.triangle_material_ids = o3d.utility.IntVector(mat_ids_kept.tolist())
    return mesh


def clean_mesh_global(
    mesh: o3d.geometry.TriangleMesh,
    min_component_tri_ratio: float = 0.005,
    min_component_area_ratio: float = 0.002,
    min_component_triangles: int = 15,
    remove_non_manifold: bool = False,
    orient_consistent: bool = False,
) -> tuple[o3d.geometry.TriangleMesh, dict[str, int]]:
    """Lam sach topo & loai bo manh vun troi noi toan cuc theo ty le tuong doi (khong hardcode vat the).

    Giu lai tat ca cac thanh phan/vat the hop le trong canh (multi-object safe):
    Mot cum lien thong chi bi loai bo khi NO VUA nho hon nguong ty le tam giac
    VUA nho hon nguong ty le dien tich be mat (hoac la bui < min_component_triangles).
    Luon bao toan it nhat 1 cum lon nhat de khong bao gio xoa trang mesh.
    """
    stats = {
        "components_before": 0,
        "components_removed": 0,
        "zero_area_tris_removed": 0,
    }
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return mesh, stats

    # 1. Xoa tam giac dien tich ~0 (degenerate) co bao toan UVs/material IDs
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    e1 = verts[tris[:, 1]] - verts[tris[:, 0]]
    e2 = verts[tris[:, 2]] - verts[tris[:, 0]]
    dbl_area = np.linalg.norm(np.cross(e1, e2), axis=1)
    # Nguong tuong doi theo binh phuong duong cheo bbox de khong phu thuoc don vi (mm vs m)
    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    eps_area = max(1e-12, 1e-10 * diag * diag)
    zero_mask = (dbl_area <= eps_area) | (tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2]) | (tris[:, 0] == tris[:, 2])
    if np.any(zero_mask):
        stats["zero_area_tris_removed"] = int(np.sum(zero_mask))
        mesh = remove_triangles_preserving_attributes(mesh, zero_mask)

    # 2. Don trung lap dinh / tam giac & canh phi da tap (neu bat)
    # Luu y: chi goi remove_degenerate/duplicated khi khong co UV seam de tranh Open3D bao canh bao & lam lech UV
    has_uvs = mesh.has_triangle_uvs() and len(mesh.triangle_uvs) == 3 * len(mesh.triangles)
    if not has_uvs:
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
    if remove_non_manifold:
        mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if len(mesh.triangles) == 0:
        return mesh, stats

    # 3. Loc manh vun troi noi theo ty le tuong doi (scene-agnostic, UV-seam safe, multi-object safe)
    verts_cur = np.asarray(mesh.vertices, dtype=np.float64)
    tris_cur = np.asarray(mesh.triangles, dtype=np.int32)
    tri_clusters, n_tris_arr, areas_arr = _spatial_connected_components(verts_cur, tris_cur)
    num_comp = len(n_tris_arr)
    stats["components_before"] = num_comp

    if num_comp > 1 and (min_component_tri_ratio > 0 or min_component_area_ratio > 0 or min_component_triangles > 0):
        total_tris = max(int(n_tris_arr.sum()), 1)
        total_area = max(float(areas_arr.sum()), 1e-12)

        tri_ratios = n_tris_arr / float(total_tris)
        area_ratios = areas_arr / total_area

        # Giu cum neu du lon theo ty le tam giac HOAC ty le dien tich, va >= nguong bui toi thieu
        keep_comp = (n_tris_arr >= int(min_component_triangles)) & (
            (tri_ratios >= float(min_component_tri_ratio)) | (area_ratios >= float(min_component_area_ratio))
        )
        # Luon giu it nhat cum lon nhat (tranh xoa trang toan bo canh gom nhieu vat nho)
        if not np.any(keep_comp):
            keep_comp[int(np.argmax(n_tris_arr))] = True

        n_removed = int(np.sum(~keep_comp))
        if n_removed > 0:
            remove_tri_mask = ~keep_comp[tri_clusters]
            mesh = remove_triangles_preserving_attributes(mesh, remove_tri_mask)
            mesh.remove_unreferenced_vertices()
            stats["components_removed"] = n_removed

    # 4. Dong nhat chieu cuon tam giac (winding order) neu la luoi manifold va duoc yeu cau
    if orient_consistent and len(mesh.triangles) > 0:
        try:
            if mesh.is_edge_manifold(allow_boundary_edges=True):
                mesh.orient_triangles()
        except Exception:
            pass

    return mesh, stats


def peel_boundary_slivers(
    mesh: o3d.geometry.TriangleMesh,
    aspect_ratio_thr: float = 8.0,
    max_passes: int = 2,
) -> tuple[o3d.geometry.TriangleMesh, int]:
    """Cat tia nhe cac tam giac ria/vay qua det (overhanging boundary slivers) o bien mo.

    Chi loai bo tam giac nam tren canh bien (co it nhat 1 canh chi thuoc 1 tam giac) VA co
    ty le canh (aspect ratio) > ``aspect_ratio_thr`` de khong bao gio cat vao chi tiet mong hop le.
    """
    if aspect_ratio_thr <= 0 or max_passes <= 0 or len(mesh.triangles) == 0:
        return mesh, 0

    total_removed = 0
    for _ in range(int(max_passes)):
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        tris = np.asarray(mesh.triangles, dtype=np.int32)
        n_t = len(tris)
        if n_t == 0:
            break

        # Tim cac canh bien (canh vo huong chi xuat hien dung 1 lan)
        e01 = np.sort(tris[:, [0, 1]], axis=1)
        e12 = np.sort(tris[:, [1, 2]], axis=1)
        e20 = np.sort(tris[:, [2, 0]], axis=1)
        all_edges = np.vstack([e01, e12, e20])
        # Ma hoa canh thanh 1 so int64 duy nhat
        max_v = int(len(verts)) + 1
        keys = all_edges[:, 0].astype(np.int64) * max_v + all_edges[:, 1].astype(np.int64)
        uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        edge_is_boundary = counts[inv] == 1
        tri_is_boundary = (
            edge_is_boundary[:n_t]
            | edge_is_boundary[n_t : 2 * n_t]
            | edge_is_boundary[2 * n_t :]
        )
        if not np.any(tri_is_boundary):
            break

        # Tinh aspect ratio cua cac tam giac bien
        v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        a = np.linalg.norm(v1 - v0, axis=1)
        b = np.linalg.norm(v2 - v1, axis=1)
        c = np.linalg.norm(v0 - v2, axis=1)
        s = 0.5 * (a + b + c)
        denom = 8.0 * np.maximum(s - a, 0.0) * np.maximum(s - b, 0.0) * np.maximum(s - c, 0.0)
        ar = np.where(denom > 1e-18, (a * b * c) / np.maximum(denom, 1e-18), np.inf)

        remove_mask = tri_is_boundary & (ar > float(aspect_ratio_thr))
        n_rem = int(np.sum(remove_mask))
        if n_rem == 0 or n_rem >= n_t:
            break

        mesh = remove_triangles_preserving_attributes(mesh, remove_mask)
        mesh.remove_unreferenced_vertices()
        total_removed += n_rem

    return mesh, total_removed


def prune_excess_mesh_extrapolation(
    mesh: o3d.geometry.TriangleMesh,
    ref_points: np.ndarray,
    ref_confidence: np.ndarray | None = None,
    max_dist_frac: float = 0.045,
    floor_tol_frac: float = 0.015,
) -> tuple[o3d.geometry.TriangleMesh, int]:
    """Cat bo phan luoi thua / bong bong ngoai suy Poisson nam xa tap diem MVS quan sat thuc.

    Uu tien chi dung cac diem quan sat thuc (``ref_confidence >= 0.90``) lam neo hinh hoc;
    bao ve mat day phang tiep dat (Z ~ z_min) neu vat the da duoc bit kin day watertight.
    """
    if max_dist_frac <= 0.0 or len(mesh.triangles) == 0 or len(ref_points) < 20 or cKDTree is None:
        return mesh, 0

    obs_pts = np.asarray(ref_points, dtype=np.float64)
    if ref_confidence is not None and len(ref_confidence) == len(obs_pts):
        high_conf = obs_pts[np.asarray(ref_confidence, dtype=np.float64) >= 0.90]
        if len(high_conf) >= 50:
            obs_pts = high_conf

    obs_diag = max(robust_diag(obs_pts), 1.0e-6)
    dist_thr = float(max_dist_frac) * obs_diag

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    v_dists, _ = cKDTree(obs_pts).query(verts, k=1)

    # Mot tam giac la ngoai suy thua khi trung binh hoac cuc dai dinh cua no vuot nguong ho tro
    tri_max_d = np.max(v_dists[tris], axis=1)
    tri_mean_d = np.mean(v_dists[tris], axis=1)
    unsupported = (tri_mean_d > dist_thr) | (tri_max_d > 1.35 * dist_thr)

    # Bao ve nap bit phang day (neu tat ca 3 dinh nam sat mat san z_min cua diem quan sat)
    z_floor = float(obs_pts[:, 2].min())
    floor_tol = float(floor_tol_frac) * obs_diag
    tri_z = verts[tris, 2]
    xy_in_bbox = (
        (verts[tris, 0].mean(axis=1) >= obs_pts[:, 0].min() - floor_tol)
        & (verts[tris, 0].mean(axis=1) <= obs_pts[:, 0].max() + floor_tol)
        & (verts[tris, 1].mean(axis=1) >= obs_pts[:, 1].min() - floor_tol)
        & (verts[tris, 1].mean(axis=1) <= obs_pts[:, 1].max() + floor_tol)
    )
    is_bottom_cap = np.all(np.abs(tri_z - z_floor) <= floor_tol, axis=1) & xy_in_bbox
    remove_mask = unsupported & (~is_bottom_cap)

    n_rem = int(np.sum(remove_mask))
    if n_rem == 0 or n_rem >= int(0.85 * len(tris)):
        return mesh, 0

    mesh = remove_triangles_preserving_attributes(mesh, remove_mask)
    mesh.remove_unreferenced_vertices()
    return mesh, n_rem


def extract_obj_mtl_headers(obj_path: Path) -> list[str]:
    """Trich xuat cac dong tham chieu `mtllib` tu file .obj goc (neu co)."""
    headers: list[str] = []
    if not obj_path.exists():
        return headers
    try:
        with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i > 100:
                    break
                stripped = line.strip()
                if stripped.startswith("mtllib "):
                    headers.append(stripped)
    except Exception:
        pass
    return headers


# ---------------------------------------------------------------------------
# POINT CLOUD & MESH UTILITIES
# ---------------------------------------------------------------------------

def nn_query(ref: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def robust_diag(pts: np.ndarray) -> float:
    p5 = np.percentile(pts, 5, axis=0)
    p95 = np.percentile(pts, 95, axis=0)
    diag = float(np.linalg.norm(p95 - p5))
    if diag < 1e-6:
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    return diag


def load_gaussian_ply(path: Path, opacity_min: float = 0.20) -> o3d.geometry.PointCloud:
    """Doc .ply tu ns-export gaussian-splat hoac colmap stereo_fusion."""
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = v.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    keep = np.all(np.isfinite(xyz), axis=1)
    if "opacity" in names:
        opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
        keep &= opacity > float(opacity_min)
        print(f"  loc opacity > {opacity_min:.2f}: giu {int(keep.sum()):,}/{len(xyz):,} diem")

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


def mls_smooth_point_cloud(
    pc: o3d.geometry.PointCloud,
    iterations: int = 2,
    k: int = 30,
    alpha: float = 0.80,
    outlier_sigma: float = 3.0,
    polynomial_order: int = 2,
    radius: float = 0.0,
    bandwidth: float = 0.0,
    reg_eps: float = 1.0e-6,
    recompute_normals: bool = True,
    chunk_size: int = 40000,
) -> o3d.geometry.PointCloud:
    """Chieu diem Moving Least Squares (MLS) ve tiep dien cuc bo (Eq. 6 quadratic Monge hoac planar).

    Khi ``polynomial_order == 2`` (mac dinh theo Mục 2.3 & Eq. 6 trong nckh.pdf), khop paraboloid
    vi phan bac hai z = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2 tren tung lan can k-NN / ban kinh R,
    bao ton dong thoi vung lom yen ngua (K < 0) va vom loi (K > 0).
    """
    if iterations <= 0 or len(pc.points) < max(k + 1, 20) or cKDTree is None:
        return pc

    if int(polynomial_order) >= 2:
        from nckh_2.backends.surfaces import mls_monge_project_point_cloud

        n_before = len(pc.points)
        out_monge = mls_monge_project_point_cloud(
            pc,
            iterations=int(iterations),
            k_neighbors=int(k),
            alpha=float(alpha),
            outlier_sigma=float(outlier_sigma),
            radius=float(radius),
            bandwidth=float(bandwidth),
            reg_eps=float(reg_eps),
            recompute_normals=bool(recompute_normals),
            chunk_size=int(chunk_size),
        )
        removed_m = n_before - len(out_monge.points)
        print(
            f"  MLS Monge deg-2 ({iterations}x, k={min(int(k), n_before - 1)}, alpha={float(alpha):.2f}): "
            f"loai {removed_m:,} diem ngoai lai -> {len(out_monge.points):,} diem"
        )
        return out_monge

    pts = np.asarray(pc.points, dtype=np.float64).copy()
    has_norms = pc.has_normals() and len(pc.normals) == len(pts)
    norms = np.asarray(pc.normals, dtype=np.float64).copy() if has_norms else None
    has_cols = pc.has_colors() and len(pc.colors) == len(pts)
    cols = np.asarray(pc.colors, dtype=np.float64).copy() if has_cols else None

    k_eff = min(int(k), len(pts) - 1)
    a = float(np.clip(alpha, 0.0, 1.0))

    # Xu ly theo lo (chunk) de tiet kiem RAM tren point cloud > 200k diem
    chunk_size = 40000
    for _ in range(int(iterations)):
        tree = cKDTree(pts)
        new_pts = np.empty_like(pts)
        signed_dists = np.zeros(len(pts), dtype=np.float64)

        for start in range(0, len(pts), chunk_size):
            end = min(start + chunk_size, len(pts))
            _, idx = tree.query(pts[start:end], k=k_eff, workers=-1)
            nbrs = pts[idx]  # (B, k, 3)
            centroids = nbrs.mean(axis=1)  # (B, 3)

            if norms is not None:
                n_loc = norms[start:end]
            else:
                # Uoc luong phap tuyen cuc bo bang SVD/PCA neu chua co
                diff = nbrs - centroids[:, None, :]
                cov = np.einsum("bki,bkj->bij", diff, diff) / max(k_eff - 1, 1)
                _, ev = np.linalg.eigh(cov)
                n_loc = ev[:, :, 0]

            n_len = np.maximum(np.linalg.norm(n_loc, axis=1, keepdims=True), 1e-12)
            n_loc = n_loc / n_len

            sd = np.sum((pts[start:end] - centroids) * n_loc, axis=1)
            signed_dists[start:end] = sd
            new_pts[start:end] = pts[start:end] - (a * sd)[:, None] * n_loc

        pts = new_pts

    # Loai bo diem ngoai lai nang nam xa be mat tiep dien (vo troi / ghost arcs)
    keep = np.ones(len(pts), dtype=bool)
    if outlier_sigma > 0:
        abs_sd = np.abs(signed_dists)
        med = float(np.median(abs_sd))
        mad = float(np.median(np.abs(abs_sd - med)))
        thr = med + float(outlier_sigma) * max(1.4826 * mad, 1e-6)
        keep = abs_sd <= thr
        if int(np.sum(keep)) < max(100, int(0.5 * len(pts))):
            keep = np.ones(len(pts), dtype=bool)

    removed = int(len(pts) - np.sum(keep))
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[keep])
    if norms is not None:
        out.normals = o3d.utility.Vector3dVector(norms[keep])
    if cols is not None:
        out.colors = o3d.utility.Vector3dVector(cols[keep])
    print(f"  MLS thin-sheet ({iterations}x, k={k_eff}, alpha={a:.2f}): loai {removed:,} diem ngoai lai -> {len(out.points):,} diem")
    return out


def smooth_point_cloud_normals(
    pc: o3d.geometry.PointCloud,
    iterations: int = 3,
    k: int = 25,
    sigma_angle_deg: float = 45.0,
) -> o3d.geometry.PointCloud:
    """Lam muot truong phap tuyen k-NN kieu Bilateral (giu canh sac, triet tieu be mat da/nham thach)."""
    if iterations <= 0 or not pc.has_normals() or len(pc.points) < max(k + 1, 20) or cKDTree is None:
        return pc

    pts = np.asarray(pc.points, dtype=np.float64)
    norms = np.asarray(pc.normals, dtype=np.float64).copy()
    n_len = np.maximum(np.linalg.norm(norms, axis=1, keepdims=True), 1e-12)
    norms /= n_len

    k_eff = min(int(k), len(pts) - 1)
    sigma_rad = np.radians(max(float(sigma_angle_deg), 5.0))
    inv_2s2 = 1.0 / (2.0 * sigma_rad * sigma_rad)
    tree = cKDTree(pts)
    chunk_size = 40000

    for _ in range(int(iterations)):
        next_norms = np.empty_like(norms)
        for start in range(0, len(pts), chunk_size):
            end = min(start + chunk_size, len(pts))
            _, idx = tree.query(pts[start:end], k=k_eff, workers=-1)
            n_cur = norms[start:end]  # (B, 3)
            n_nbr = norms[idx]        # (B, k, 3)
            dots = np.clip(np.sum(n_cur[:, None, :] * n_nbr, axis=2), -1.0, 1.0)
            angles = np.arccos(dots)
            w = np.exp(-(angles * angles) * inv_2s2)  # (B, k)
            # Bo qua phap tuyen nguoc chieu (> 90 do)
            w = np.where(dots > 0.0, w, 0.0)
            acc = np.sum(w[:, :, None] * n_nbr, axis=1) + 0.25 * n_cur
            acc_len = np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
            next_norms[start:end] = acc / acc_len
        norms = next_norms

    out = o3d.geometry.PointCloud(pc)
    out.normals = o3d.utility.Vector3dVector(norms)
    print(f"  Bilateral Normal Smooth ({iterations}x, k={k_eff}, sigma={sigma_angle_deg:.0f} deg): xong")
    return out


def clean_mesh_topology(mesh: o3d.geometry.TriangleMesh, keep_ratio: float = 0.02) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    tri_clusters, n_tris, _ = mesh.cluster_connected_triangles()
    tri_clusters, n_tris = np.asarray(tri_clusters), np.asarray(n_tris)
    if len(n_tris) > 0:
        thr = max(20, int(n_tris.max() * keep_ratio))
        keep = np.nonzero(n_tris >= thr)[0]
        mesh = remove_triangles_preserving_attributes(mesh, ~np.isin(tri_clusters, keep))
        mesh.remove_unreferenced_vertices()
    return mesh


def make_watertight(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Bit cac chu trinh bien (PyMeshLab neu co, khong thi fan-fill tai centroid)."""
    if len(mesh.triangles) == 0 or mesh.is_watertight():
        return mesh

    has_col = mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices)

    try:
        import pymeshlab
        src_verts = np.asarray(mesh.vertices, dtype=np.float64)
        src_cols = np.asarray(mesh.vertex_colors, dtype=np.float64) if has_col else None
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=src_verts,
                                   face_matrix=np.asarray(mesh.triangles, dtype=np.int32)))
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_close_holes(maxholesize=100000, refinehole=True)
        res = ms.current_mesh()
        out = o3d.geometry.TriangleMesh()
        out_verts = np.asarray(res.vertex_matrix(), dtype=np.float64)
        out.vertices = o3d.utility.Vector3dVector(out_verts)
        out.triangles = o3d.utility.Vector3iVector(res.face_matrix())
        if not out.is_watertight():
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_repair_non_manifold_vertices()
            ms.meshing_close_holes(maxholesize=100000, refinehole=False)
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_repair_non_manifold_vertices()
            res = ms.current_mesh()
            out_verts = np.asarray(res.vertex_matrix(), dtype=np.float64)
            out.vertices = o3d.utility.Vector3dVector(out_verts)
            out.triangles = o3d.utility.Vector3iVector(res.face_matrix())
        if src_cols is not None and len(out_verts) > 0:
            _, idx = nn_query(src_verts, out_verts)
            out.vertex_colors = o3d.utility.Vector3dVector(src_cols[idx])
        out.remove_degenerate_triangles()
        out.remove_duplicated_triangles()
        out.remove_non_manifold_edges()
        out.remove_unreferenced_vertices()
        out.compute_triangle_normals()
        out.compute_vertex_normals()
        print(f"  PyMeshLab close holes: watertight={out.is_watertight()}")
        if out.is_watertight():
            return out
        # Continue to fan-fill fallback on `out` if any residual open loops remain
        mesh = out
        has_col = mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices)
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


def ground_mesh_z0(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    v = np.asarray(mesh.vertices)
    if len(v) == 0:
        return mesh
    v = v.copy()
    v[:, 2] -= v[:, 2].min()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def cap_bottom(mesh: o3d.geometry.TriangleMesh, diag: float, e: dict) -> o3d.geometry.TriangleMesh:
    """Mat ban = Z=0. Ep cac dinh sat/duoi san ve Z=0, bo tam giac nam han tren san, roi bit day."""
    if len(mesh.vertices) == 0:
        return mesh
    tol = float(e.get("floor_snap_frac", 0.02)) * diag
    v = np.asarray(mesh.vertices).copy()
    v[v[:, 2] < tol, 2] = 0.0
    mesh.vertices = o3d.utility.Vector3dVector(v)

    tris = np.asarray(mesh.triangles)
    on_floor = np.all(v[tris, 2] <= 0.0, axis=1)
    mesh = remove_triangles_preserving_attributes(mesh, on_floor)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    return make_watertight(mesh)


def export_obj(
    mesh: o3d.geometry.TriangleMesh,
    path: Path,
    mtl_headers: list[str] | None = None,
) -> None:
    """Xuat file .obj chuan, tu dong tao thu muc cha, bao toan mau, phap tuyen, UV va mtllib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)
    has_colors = bool(mesh.has_vertex_colors() and len(mesh.vertex_colors) == n_v)
    has_uvs = bool(mesh.has_triangle_uvs() and len(mesh.triangle_uvs) == 3 * n_t and n_t > 0)

    o3d.io.write_triangle_mesh(
        str(path),
        mesh,
        write_ascii=True,
        write_vertex_normals=True,
        write_vertex_colors=has_colors,
        write_triangle_uvs=has_uvs,
    )

    # Neu file nguon co tham chieu mtllib nhung Open3D bo qua, chen lai an toan vao dau file
    if mtl_headers:
        try:
            text = path.read_text(encoding="utf-8")
            missing = [h for h in mtl_headers if h not in text[:512]]
            if missing:
                path.write_text("\n".join(missing) + "\n" + text, encoding="utf-8")
        except Exception:
            pass
