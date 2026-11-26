"""Buoc 4: Don sach point cloud (Open3D) + Dung mesh 3D tong quat -> .obj."""
from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import numpy as np
import open3d as o3d

from nckh_2.backends.colmap_io import load_camera_centers, orbit_normal
from nckh_2.backends.geometry import (
    clean_mesh_topology,
    export_obj,
    ground_mesh_z0,
    load_gaussian_ply,
    make_watertight,
    mls_smooth_point_cloud,
    nn_query,
    robust_diag,
    smooth_point_cloud_normals,
)
from nckh_2.core import paths
from nckh_2.core.types import SceneContext, StageResult


def _keep_significant_clusters(pc: o3d.geometry.PointCloud, diag: float, e: dict[str, Any]) -> o3d.geometry.PointCloud:
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


def _clean(pc: o3d.geometry.PointCloud, e: dict[str, Any]) -> tuple[o3d.geometry.PointCloud, float]:
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        return pc, 0.0
    diag = robust_diag(pts)
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


def _remove_plane_ransac(
    pc: o3d.geometry.PointCloud,
    diag: float,
    e: dict[str, Any],
    cam_centers: np.ndarray | None = None,
) -> tuple[o3d.geometry.PointCloud, tuple[np.ndarray, float] | None]:
    if not e.get("remove_table_plane", True) or len(pc.points) < 100:
        return pc, None

    dist_thresh = max(float(e.get("plane_dist_threshold_frac", 0.015)) * diag, 1e-6)
    ransac_iters = int(e.get("plane_ransac_iterations", 1500))
    try:
        model, inliers = pc.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=ransac_iters)
    except Exception:
        return pc, None

    ratio = len(inliers) / max(len(pc.points), 1)
    min_ratio, max_ratio = e.get("plane_inlier_ratio_range", [0.25, 0.70])
    if not (min_ratio <= ratio <= max_ratio):
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
    pos_pts = int(np.sum(dists > dist_thresh))
    neg_pts = int(np.sum(dists < -dist_thresh))
    out_pts = pos_pts + neg_pts
    if out_pts > 0:
        one_sided = max(pos_pts, neg_pts) / float(out_pts)
    else:
        one_sided = 0.0

    min_one_sided = float(e.get("plane_one_sided_min_ratio", 0.75))
    if one_sided < min_one_sided:
        return pc, None

    orbit = orbit_normal(cam_centers)
    min_orbit_cos = float(e.get("plane_orbit_min_cos", 0.65))
    if orbit is not None and abs(float(n @ orbit)) < min_orbit_cos:
        print("  RANSAC Plane: bo qua (mat phang khong song song quy dao camera)")
        return pc, None

    if neg_pts > pos_pts:
        n, d = -n, -d
        dists = -dists

    print(f"  RANSAC Plane: tach {len(inliers):,}/{len(pc.points):,} diem ({ratio:.1%}) mat ban (one_sided={one_sided:.1%})")

    # Loc sach diem mat ban va vung nhieu duoi mat ban (table clearance)
    clearance = float(e.get("table_clearance_frac", 0.006)) * diag
    above_table = dists > clearance
    out_pc = pc.select_by_index(np.where(above_table)[0])
    if len(out_pc.points) >= 50:
        out_pc = _keep_significant_clusters(out_pc, diag, e)
    else:
        out_pc = pc.select_by_index(inliers, invert=True)

    return out_pc, (n, float(d))


def _orient_normals(
    pc: o3d.geometry.PointCloud,
    cam_centers: np.ndarray,
    diag: float,
    e: dict[str, Any],
) -> o3d.geometry.PointCloud:
    pts = np.asarray(pc.points)
    n_pts = len(pts)
    if n_pts == 0:
        return pc

    recompute = bool(e.get("recompute_normals", True))
    if not recompute and pc.has_normals() and len(pc.normals) == n_pts:
        lens = np.linalg.norm(np.asarray(pc.normals), axis=1)
        if np.mean(lens > 0.5) > 0.99:
            print(f"  giu phap tuyen co san ({n_pts:,} vector)")
            return pc

    rad = max(float(e.get("normal_radius_frac", 0.02)) * diag, 1e-6)
    nn = int(e.get("normal_nn", 30))
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=nn))

    orient_k = min(int(e.get("orient_k", 15)), max(n_pts - 1, 1))
    tangent_oriented = False
    if orient_k >= 3:
        try:
            pc.orient_normals_consistent_tangent_plane(k=orient_k)
            tangent_oriented = True
            print(f"  dong bo phap tuyen theo mat tiep dien (k={orient_k})")
        except Exception as err:
            print(f"  [Canh bao] orient_normals_consistent_tangent_plane(k={orient_k}) loi: {err}")

    norms = np.asarray(pc.normals).copy()
    if len(cam_centers) == 1:
        pc.orient_normals_towards_camera_location(camera_location=cam_centers[0])
        print("  dinh huong phap tuyen ve phia camera")
    elif len(cam_centers) >= 2:
        _, idx = nn_query(cam_centers, pts)
        view = cam_centers[idx] - pts
        inward = np.einsum("ij,ij->i", norms, view) < 0
        if tangent_oriented:
            if float(np.mean(inward)) > 0.5:
                norms *= -1.0
                pc.normals = o3d.utility.Vector3dVector(norms)
                print(f"  lat toan cuc phap tuyen huong ve {len(cam_centers)} camera")
            else:
                print(f"  phap tuyen da huong dung ve {len(cam_centers)} camera ({1.0 - float(np.mean(inward)):.1%})")
        else:
            norms[inward] *= -1.0
            pc.normals = o3d.utility.Vector3dVector(norms)
            print(f"  dinh huong phap tuyen theo {len(cam_centers)} camera (lat {int(inward.sum()):,})")
    else:
        view = pts - pts.mean(axis=0)
        inward = np.einsum("ij,ij->i", norms, view) < 0
        if tangent_oriented:
            if float(np.mean(inward)) > 0.5:
                norms *= -1.0
                pc.normals = o3d.utility.Vector3dVector(norms)
                print("  lat toan cuc phap tuyen ra ngoai tam vat")
        else:
            norms[inward] *= -1.0
            pc.normals = o3d.utility.Vector3dVector(norms)
            print("  dinh huong phap tuyen ra ngoai tam vat (khong co camera)")
    return pc


def _align_point_cloud_upright(
    pc: o3d.geometry.PointCloud,
    support_plane: tuple[np.ndarray, float] | None = None,
    cam_centers: np.ndarray | None = None,
) -> tuple[o3d.geometry.PointCloud, Any]:
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        return pc, (lambda x: x)
    m_cent = pts.mean(axis=0).copy()
    centered = pts - m_cent

    if support_plane is not None:
        n_z = support_plane[0].copy()
    else:
        _, ev = np.linalg.eigh(np.cov(centered, rowvar=False))
        spans = [float(np.ptp(centered @ ev[:, i])) for i in range(3)]
        pca_thin_z = ev[:, int(np.argmin(spans))].copy()
        orb_z = orbit_normal(cam_centers)
        if orb_z is not None:
            if np.dot(orb_z, cam_centers.mean(axis=0) - m_cent) < 0:
                orb_z = -orb_z
            # Neu truc mong nhat cua vat (PCA thickness) lech qua xa (>40 do) so voi mat phang quy dao bi nhieu,
            # uu tien truc mong nhat PCA va dinh huong cung ban cau voi camera de vat luon nam phang tren mat ban!
            if abs(float(np.dot(pca_thin_z, orb_z))) < 0.76 and (max(spans) / max(min(spans), 1e-6) >= 1.35):
                n_z = pca_thin_z
                if cam_centers is not None and len(cam_centers) > 0 and np.dot(n_z, cam_centers.mean(axis=0) - m_cent) < 0:
                    n_z = -n_z
            else:
                n_z = orb_z
        else:
            n_z = pca_thin_z
            if cam_centers is not None and len(cam_centers) > 0 and np.dot(n_z, cam_centers.mean(axis=0) - m_cent) < 0:
                n_z = -n_z
    n_z /= np.linalg.norm(n_z)

    proj = centered - np.outer(centered @ n_z, n_z)
    _, ev2 = np.linalg.eigh(np.cov(proj, rowvar=False))
    n_x = ev2[:, 2] - np.dot(ev2[:, 2], n_z) * n_z
    n_x /= np.linalg.norm(n_x)
    n_y = np.cross(n_z, n_x)
    n_y /= np.linalg.norm(n_y)

    R = np.vstack([n_x, n_y, n_z])
    p_rot = centered @ R.T

    shift_x = 0.5 * (p_rot[:, 0].min() + p_rot[:, 0].max())
    shift_y = 0.5 * (p_rot[:, 1].min() + p_rot[:, 1].max())
    if support_plane is not None:
        n, d = support_plane
        shift_z = -(float(n @ m_cent) + d)
    else:
        shift_z = float(p_rot[:, 2].min())

    post_shift = np.array([shift_x, shift_y, shift_z], dtype=np.float64)
    p_rot -= post_shift[None, :]

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


def _reconstruct_bpa(pc: o3d.geometry.PointCloud, diag: float, e: dict[str, Any]) -> o3d.geometry.TriangleMesh:
    if len(pc.points) < 10:
        return o3d.geometry.TriangleMesh()

    pcd = pc
    max_bpa_pts = int(e.get("bpa_max_points", 250000))
    if len(pc.points) > max_bpa_pts:
        vox = float(e.get("bpa_voxel_frac", 0.004)) * diag
        if vox > 0:
            pcd = pc.voxel_down_sample(vox)
            print(f"  BPA voxel downsample: {len(pc.points):,} -> {len(pcd.points):,}")
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(float(e.get("normal_radius_frac", 0.02)) * diag, 1e-6),
            max_nn=int(e.get("normal_nn", 30)),
        ))

    avg = float(np.mean(pcd.compute_nearest_neighbor_distance()))
    radii = [avg * float(e.get(k, m)) for k, m in
             (("bpa_r1_mult", 1.2), ("bpa_r2_mult", 2.5), ("bpa_r3_mult", 5.0))]
    print(f"  BPA: avg_dist={avg:.5f} radii={[round(r, 5) for r in radii]}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    mesh = clean_mesh_topology(mesh)
    mesh.compute_vertex_normals()
    print(f"  BPA: {len(mesh.vertices):,} dinh, {len(mesh.triangles):,} tam giac")
    return mesh


def _reconstruct_poisson(
    pc: o3d.geometry.PointCloud,
    diag: float,
    e: dict[str, Any],
) -> o3d.geometry.TriangleMesh:
    pcd = o3d.geometry.PointCloud(pc)
    if not pcd.has_normals() or len(pcd.normals) != len(pcd.points):
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(float(e.get("normal_radius_frac", 0.02)) * diag, 1e-6),
            max_nn=int(e.get("normal_nn", 30)),
        ))
        orient_k = min(int(e.get("orient_k", 15)), max(len(pcd.points) - 1, 1))
        if orient_k >= 3:
            pcd.orient_normals_consistent_tangent_plane(k=orient_k)

    depth = int(e.get("poisson_depth", 7))
    scale = float(e.get("poisson_scale", 1.1))
    linear_fit = bool(e.get("poisson_linear_fit", True))
    print(f"  Poisson depth={depth}, scale={scale}, linear_fit={linear_fit}...")
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit)
    dens = np.asarray(dens, dtype=np.float64)
    print(f"  Poisson raw: {len(mesh.vertices):,} dinh, {len(mesh.triangles):,} tam giac")

    q = float(e.get("poisson_density_quantile", 0.0))
    if q <= 0.0:
        q = float(e.get("poisson_density_trim_percentile", 0.0)) / 100.0
    if 0.0 < q < 1.0 and len(dens) == len(mesh.vertices) and len(dens) > 0:
        dens_thr = float(np.quantile(dens, q))
        low_dens_mask = dens < dens_thr
        n_pruned = int(np.sum(low_dens_mask))
        if 0 < n_pruned < len(mesh.vertices):
            mesh.remove_vertices_by_mask(low_dens_mask)
            print(f"  Poisson density trim (q={q:.3f}, thr={dens_thr:.4f}): loai {n_pruned:,} dinh mat do thap")

    bbox = pcd.get_axis_aligned_bounding_box()
    pad = float(e.get("poisson_hull_pad_frac", 0.03)) * float(np.linalg.norm(bbox.get_extent()))
    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(bbox.min_bound - pad, bbox.max_bound + pad))

    # Giu moi vat the hop le (>= 5% cum lon nhat) de ho tro canh 2+ vat the khac hinh dang
    keep_ratio = float(e.get("poisson_cluster_keep_ratio", 0.05))
    mesh = clean_mesh_topology(mesh, keep_ratio=keep_ratio)

    s_iter = int(e.get("smooth_iter", 10))
    if s_iter > 0 and len(mesh.vertices) > 0:
        lam = float(e.get("smooth_lambda", 0.35))
        mu = float(e.get("smooth_mu", -0.37))
        if lam <= 0.0 or mu >= -lam:
            print(f"  [Canh bao] Tham so Taubin (lambda={lam}, mu={mu}) khong thoa man lambda > 0 va mu < -lambda")
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=s_iter,
            lambda_filter=lam,
            mu=mu,
        )

    p_iter = int(e.get("polish_iter", 0))
    if p_iter > 0 and len(mesh.vertices) > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=p_iter)

    if pc.has_colors() and len(mesh.vertices) > 0:
        _, idx = nn_query(np.asarray(pc.points), np.asarray(mesh.vertices))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pc.colors)[idx])

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    print(f"  Poisson cuoi: {len(mesh.vertices):,} dinh, {len(mesh.triangles):,} tam giac")
    return mesh


def _triangulate_2d_ear_clipping(pts_2d: np.ndarray, loop_indices: list[int]) -> list[tuple[int, int, int]]:
    """Triangulate a simple 2D polygon using ear-clipping on exact boundary vertex indices."""
    m = len(loop_indices)
    if m < 3:
        return []
    if m == 3:
        return [(loop_indices[0], loop_indices[1], loop_indices[2])]

    # Compute signed 2D area to determine polygon winding (CCW > 0, CW < 0)
    poly = pts_2d[loop_indices]
    x, y = poly[:, 0], poly[:, 1]
    area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    sign = 1.0 if area2 >= 0.0 else -1.0

    rem = list(loop_indices)
    tris: list[tuple[int, int, int]] = []
    max_guard = m * m + 16
    guard = 0

    while len(rem) > 3 and guard < max_guard:
        guard += 1
        ear_found = False
        k = len(rem)
        for i in range(k):
            v_prev = rem[(i - 1) % k]
            v_curr = rem[i]
            v_next = rem[(i + 1) % k]

            a = pts_2d[v_prev]
            b = pts_2d[v_curr]
            c = pts_2d[v_next]

            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if sign * cross <= 1e-12:
                continue

            # Check no other polygon vertex lies strictly inside triangle (a, b, c)
            other_idx = [v for v in rem if v != v_prev and v != v_curr and v != v_next]
            if other_idx:
                p = pts_2d[other_idx]
                c1 = (b[0] - a[0]) * (p[:, 1] - a[1]) - (b[1] - a[1]) * (p[:, 0] - a[0])
                c2 = (c[0] - b[0]) * (p[:, 1] - b[1]) - (c[1] - b[1]) * (p[:, 0] - b[0])
                c3 = (a[0] - c[0]) * (p[:, 1] - c[1]) - (a[1] - c[1]) * (p[:, 0] - c[0])
                inside = (sign * c1 >= -1e-12) & (sign * c2 >= -1e-12) & (sign * c3 >= -1e-12)
                if np.any(inside):
                    continue

            tris.append((v_prev, v_curr, v_next))
            rem.pop(i)
            ear_found = True
            break

        if not ear_found:
            # Relaxed ear fallback for near-collinear / slightly self-touching boundary
            best_i = 0
            best_val = -np.inf
            for i in range(k):
                a = pts_2d[rem[(i - 1) % k]]
                b = pts_2d[rem[i]]
                c = pts_2d[rem[(i + 1) % k]]
                cross = sign * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if cross > best_val:
                    best_val = cross
                    best_i = i
            tris.append((rem[(best_i - 1) % k], rem[best_i], rem[(best_i + 1) % k]))
            rem.pop(best_i)

    if len(rem) == 3:
        tris.append((rem[0], rem[1], rem[2]))
    return tris


def _extract_boundary_loops(tris: np.ndarray) -> list[list[int]]:
    """Extract closed boundary half-edge loops from a triangle mesh."""
    directed: dict[tuple[int, int], int] = {}
    for a, b, c in tris:
        ia, ib, ic = int(a), int(b), int(c)
        for u, v in ((ia, ib), (ib, ic), (ic, ia)):
            directed[(u, v)] = directed.get((u, v), 0) + 1

    adj: dict[int, list[int]] = {}
    for (u, v), cnt in directed.items():
        if cnt == 1 and (v, u) not in directed:
            adj.setdefault(u, []).append(v)

    loops: list[list[int]] = []
    while adj:
        start = next(iter(adj))
        if not adj[start]:
            adj.pop(start)
            continue
        cur = adj[start].pop()
        if not adj[start]:
            adj.pop(start)
        loop = [start]
        visited_pos = {start: 0}
        while cur != start:
            if cur in visited_pos:
                # Split off inner simple sub-loop at pinch vertex
                sub_start = visited_pos[cur]
                sub_loop = loop[sub_start:]
                if len(sub_loop) >= 3:
                    loops.append(sub_loop)
                loop = loop[:sub_start]
                visited_pos = {v: i for i, v in enumerate(loop)}
            visited_pos[cur] = len(loop)
            loop.append(cur)
            if cur not in adj or not adj[cur]:
                break
            nxt = adj[cur].pop()
            if not adj[cur]:
                adj.pop(cur)
            cur = nxt
        if cur == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def _clip_and_cap_planar_base(
    mesh: o3d.geometry.TriangleMesh,
    diag: float,
    e: dict[str, Any],
) -> o3d.geometry.TriangleMesh:
    """Object-agnostic planar base slicing at z = z_cut + 2D ear-clipping planar triangulation + welding."""
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return mesh

    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    tris = np.asarray(mesh.triangles, dtype=np.int32).copy()
    has_col = mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(verts)
    cols = np.asarray(mesh.vertex_colors, dtype=np.float64).copy() if has_col else None

    z_min = float(verts[:, 2].min())
    z_max = float(verts[:, 2].max())
    height = max(z_max - z_min, 1e-6)
    slice_frac = float(e.get("base_slice_frac", 0.035))
    z_cut = z_min + slice_frac * height

    # Nudge any vertex lying exactly on z_cut upward to avoid degenerate on-plane splits
    eps = max(1e-7 * diag, 1e-9)
    on_cut = np.abs(verts[:, 2] - z_cut) <= eps
    verts[on_cut, 2] = z_cut + eps

    above = verts[:, 2] > z_cut
    new_verts = list(verts)
    new_cols = list(cols) if has_col else None
    edge_cache: dict[tuple[int, int], int] = {}

    def _get_split_vertex(u: int, v: int) -> int:
        key = (u, v) if u < v else (v, u)
        if key in edge_cache:
            return edge_cache[key]
        p_u, p_v = verts[u], verts[v]
        t = (z_cut - p_u[2]) / (p_v[2] - p_u[2])
        p_new = p_u + t * (p_v - p_u)
        p_new[2] = z_cut
        idx_new = len(new_verts)
        new_verts.append(p_new)
        if has_col and new_cols is not None:
            new_cols.append(cols[u] + t * (cols[v] - cols[u]))
        edge_cache[key] = idx_new
        return idx_new

    out_tris: list[tuple[int, int, int]] = []
    for a, b, c in tris:
        ia, ib, ic = int(a), int(b), int(c)
        ab_flags = (bool(above[ia]), bool(above[ib]), bool(above[ic]))
        n_above = int(ab_flags[0]) + int(ab_flags[1]) + int(ab_flags[2])
        if n_above == 3:
            out_tris.append((ia, ib, ic))
        elif n_above == 0:
            continue
        elif n_above == 1:
            # Rotate cyclic order so first vertex is above
            if ab_flags[1]:
                ia, ib, ic = ib, ic, ia
            elif ab_flags[2]:
                ia, ib, ic = ic, ia, ib
            v_ab = _get_split_vertex(ia, ib)
            v_ac = _get_split_vertex(ia, ic)
            out_tris.append((ia, v_ab, v_ac))
        else:
            # 2 vertices above: rotate cyclic order so third vertex is below
            if not ab_flags[0]:
                ia, ib, ic = ib, ic, ia
            elif not ab_flags[1]:
                ia, ib, ic = ic, ia, ib
            v_bc = _get_split_vertex(ib, ic)
            v_ca = _get_split_vertex(ic, ia)
            out_tris.append((ia, ib, v_bc))
            out_tris.append((ia, v_bc, v_ca))

    if not out_tris:
        return mesh

    clipped = o3d.geometry.TriangleMesh()
    clipped.vertices = o3d.utility.Vector3dVector(np.asarray(new_verts, dtype=np.float64))
    clipped.triangles = o3d.utility.Vector3iVector(np.asarray(out_tris, dtype=np.int32))
    if has_col and new_cols is not None:
        clipped.vertex_colors = o3d.utility.Vector3dVector(np.asarray(new_cols, dtype=np.float64))

    clipped.remove_degenerate_triangles()
    clipped.remove_duplicated_triangles()
    clipped.remove_duplicated_vertices()
    clipped.remove_non_manifold_edges()
    clipped.remove_unreferenced_vertices()

    # Keep largest consolidated body before capping
    keep_ratio = float(e.get("poisson_cluster_keep_ratio", 0.15))
    clipped = clean_mesh_topology(clipped, keep_ratio=keep_ratio)

    v_arr = np.asarray(clipped.vertices, dtype=np.float64)
    t_arr = np.asarray(clipped.triangles, dtype=np.int32)
    c_arr = (
        np.asarray(clipped.vertex_colors, dtype=np.float64)
        if clipped.has_vertex_colors() and len(clipped.vertex_colors) == len(v_arr)
        else None
    )
    loops = _extract_boundary_loops(t_arr)

    extra_verts: list[np.ndarray] = []
    extra_cols: list[np.ndarray] = []
    cap_tris: list[tuple[int, int, int]] = []
    plane_tol = max(1e-3 * diag, 1e-4)
    v_offset = len(v_arr)

    for loop in loops:
        z_vals = v_arr[loop, 2]
        z_dev = float(np.max(np.abs(z_vals - z_cut)))
        if z_dev <= plane_tol:
            # Loop lies strictly on z_cut: direct 2D ear-clipping cap
            rev_loop = list(reversed(loop))
            cap_tris.extend(_triangulate_2d_ear_clipping(v_arr[:, :2], rev_loop))
        elif float(np.median(z_vals)) <= z_min + 0.35 * height:
            # Uneven bottom rim (e.g. from density trimming): project onto z_cut, stitch side skirt, cap flat base
            m_loop = len(loop)
            proj_indices: list[int] = []
            for vi in loop:
                p_proj = v_arr[vi].copy()
                p_proj[2] = z_cut
                extra_verts.append(p_proj)
                if c_arr is not None:
                    extra_cols.append(c_arr[vi].copy())
                proj_indices.append(v_offset)
                v_offset += 1
            for i in range(m_loop):
                u_top = loop[i]
                v_top = loop[(i + 1) % m_loop]
                u_bot = proj_indices[i]
                v_bot = proj_indices[(i + 1) % m_loop]
                # Opposing half-edge (v_top -> u_top) matches upper shell (u_top -> v_top)
                cap_tris.append((v_top, u_top, u_bot))
                cap_tris.append((v_top, u_bot, v_bot))
            all_xy = np.vstack([v_arr[:, :2], np.asarray(extra_verts, dtype=np.float64)[:, :2]])
            rev_proj = list(reversed(proj_indices))
            cap_tris.extend(_triangulate_2d_ear_clipping(all_xy, rev_proj))

    if cap_tris:
        if extra_verts:
            v_arr = np.vstack([v_arr, np.asarray(extra_verts, dtype=np.float64)])
            clipped.vertices = o3d.utility.Vector3dVector(v_arr)
            if c_arr is not None and extra_cols:
                c_arr = np.vstack([c_arr, np.asarray(extra_cols, dtype=np.float64)])
                clipped.vertex_colors = o3d.utility.Vector3dVector(c_arr)
        all_tris = np.vstack([t_arr, np.asarray(cap_tris, dtype=np.int32)])
        clipped.triangles = o3d.utility.Vector3iVector(all_tris)
        clipped.remove_degenerate_triangles()
        clipped.remove_duplicated_triangles()
        clipped.remove_non_manifold_edges()
        clipped.remove_unreferenced_vertices()
        print(f"  Planar base cap (z_cut={z_cut:.4f}): +{len(cap_tris):,} cap tris, watertight={clipped.is_watertight()}")

    return clipped


def _cap_bottom_mesh(mesh: o3d.geometry.TriangleMesh, diag: float, e: dict[str, Any]) -> o3d.geometry.TriangleMesh:
    if len(mesh.vertices) == 0:
        return mesh
    if bool(e.get("planar_base_cap", True)):
        mesh = _clip_and_cap_planar_base(mesh, diag, e)
        mesh = ground_mesh_z0(mesh)
        if not mesh.is_watertight() and bool(e.get("make_watertight", True)):
            mesh = make_watertight(mesh)
        return mesh

    tol = float(e.get("floor_snap_frac", 0.02)) * diag
    v = np.asarray(mesh.vertices).copy()
    v[v[:, 2] < tol, 2] = 0.0
    mesh.vertices = o3d.utility.Vector3dVector(v)

    tris = np.asarray(mesh.triangles)
    on_floor = np.all(v[tris, 2] <= 0.0, axis=1)
    mesh.remove_triangles_by_mask(on_floor)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    return make_watertight(mesh)


def _finalize_mesh(mesh: o3d.geometry.TriangleMesh, diag: float, e: dict[str, Any], has_plane: bool) -> o3d.geometry.TriangleMesh:
    if has_plane or bool(e.get("planar_base_cap", True)):
        mesh = _cap_bottom_mesh(mesh, diag, e)
    else:
        mesh = ground_mesh_z0(mesh)
        if e.get("make_watertight", True):
            mesh = make_watertight(mesh)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    print(f"  mesh cuoi: {len(mesh.vertices):,} dinh, {len(mesh.triangles):,} tam giac "
          f"watertight={mesh.is_watertight()}")
    return mesh


def run(sample_name: str, cfg: dict[str, Any]) -> None:
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

    print(f"[Buoc 4] Doc point cloud tu {ply_path.name}...")
    pc = load_gaussian_ply(ply_path, opacity_min=float(e.get("gaussian_opacity_min", 0.20)))
    print(f"  {len(pc.points):,} diem ban dau")
    if len(pc.points) == 0:
        raise RuntimeError("Khong co diem nao trong point cloud.")

    print("[Buoc 4] Loc nhieu (Statistical + Radius + DBSCAN)...")
    pc, diag = _clean(pc, e)

    cam_centers = load_camera_centers(sample_name, np.asarray(pc.points).mean(axis=0))
    print(f"[Buoc 4] Camera: {len(cam_centers)}")
    print("[Buoc 4] Tach mat ban (RANSAC)...")
    pc, support_plane = _remove_plane_ransac(pc, diag, e, cam_centers)

    print("[Buoc 4] Dinh huong phap tuyen...")
    pc = _orient_normals(pc, cam_centers, diag, e)

    mls_iter = int(e.get("mls_iter", 2))
    if mls_iter > 0:
        print("[Buoc 4] Lam muot MLS Monge point cloud...")
        num_cfg = cfg.get("numerical", {})
        pc = mls_smooth_point_cloud(
            pc,
            iterations=mls_iter,
            k=int(e.get("mls_neighbors", 30)),
            alpha=float(e.get("mls_alpha", 0.80)),
            outlier_sigma=float(e.get("mls_outlier_sigma", 3.0)),
            polynomial_order=int(e.get("mls_polynomial_order", 2)),
            radius=float(e.get("mls_radius_frac", 0.035)) * diag,
            bandwidth=float(e.get("mls_weight_bandwidth_frac", 0.020)) * diag,
            reg_eps=float(num_cfg.get("svd_reg_eps", 1.0e-6)),
            recompute_normals=bool(e.get("mls_recompute_normals", True)),
            chunk_size=int(e.get("mls_chunk_size", 40000)),
        )

    # Bu khuyet bang noi suy (Per-Component Symmetry + Visual Hull + Biharmonic) trong he toa do COLMAP goc
    point_conf: np.ndarray | None = None
    interp_cfg = e.get("interpolation_fill", {})
    if isinstance(interp_cfg, dict) and bool(interp_cfg.get("enable", False)):
        print("[Buoc 4] Bu khuyet vat the bang noi suy (Symmetry + Visual Hull + Biharmonic)...")
        from nckh_2.backends.interpolation import fill_point_cloud_holes
        pc, point_conf, _interp_stats = fill_point_cloud_holes(
            pc,
            sample_dir=paths.sample_dir(sample_name),
            interp_cfg=interp_cfg,
        )

    ns_iter = int(e.get("normal_smooth_iter", 3))
    if ns_iter > 0:
        print("[Buoc 4] Lam muot truong phap tuyen k-NN...")
        pc = smooth_point_cloud_normals(
            pc,
            iterations=ns_iter,
            k=int(e.get("normal_smooth_k", 25)),
            sigma_angle_deg=float(e.get("normal_smooth_sigma_deg", 45.0)),
        )

    if e.get("align_upright", True):
        print("[Buoc 4] Can thang truc point cloud...")
        pc = _align_point_cloud_upright(pc, support_plane, cam_centers)
    else:
        support_plane = None

    o3d.io.write_point_cloud(str(clean_ply), pc)
    o3d.io.write_point_cloud(str(aligned_ply), pc)
    if point_conf is not None and len(point_conf) == len(pc.points):
        np.save(str(out_dir / "point_confidence.npy"), point_conf)
    print(f"  da luu {clean_ply.name}")
    has_plane = support_plane is not None
    diag = max(robust_diag(np.asarray(pc.points)), 1.0e-6)

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
        export_obj(mesh_bpa, bpa_path)
        export_obj(mesh_poi, poi_path)
        print(f"  da luu {bpa_path.name} va {poi_path.name}")
        mesh = mesh_poi

    print(f"[Buoc 4] Xuat {obj_path.name}...")
    export_obj(mesh, obj_path)
    print(f"[Buoc 4] Hoan tat -> {obj_path} ({obj_path.stat().st_size/1024:.1f} KB)")
    shutil.copyfile(str(obj_path), str(out_dir / f"{sample_name}_raw.obj"))
    shutil.copyfile(str(obj_path), str(out_dir / "mesh_raw.obj"))

    try:
        from nckh_2.evaluation import spec_figures
        spec_figures.export_all_specification_figures(sample_name, out_dir, mesh, pc)
    except Exception as err:
        print(f"  [Canh bao] Xuat hinh thong so: {err}")

    if cfg.get("mesh_refine", {}).get("auto_run", True):
        from nckh_2.pipeline import mesh_refine
        mesh_refine.run(sample_name, cfg)
    else:
        print(f"  Blender: File > Import > Wavefront (.obj) > {obj_path.name}")


class MeshExportStage:
    stage_id = 4
    title = "Dung mesh khoi tao (Poisson)"

    def execute(self, ctx: SceneContext) -> StageResult:
        run(ctx.sample_name, ctx.config)
        ctx.artifacts["mesh"] = ctx.output_dir / f"{ctx.sample_name}.obj"
        return StageResult(stage_id=self.stage_id, stage_name=self.title, success=True, artifacts=ctx.artifacts)
