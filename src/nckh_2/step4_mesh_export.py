"""Buoc 4: don sach point cloud (Open3D) + dung mesh -> .obj.

mesh_mode = obb     : fit oriented bounding box -> hop chu nhat kin 6 mat (demo Pha 1)
mesh_mode = poisson : Poisson surface reconstruction -> mesh cho vat the tu do
mesh_mode = both    : xuat ca hai (obb + poisson) cung luc de tham chieu va tai hien

Vao : outputs/<mau>/point_cloud.ply     (tu Buoc 3, dinh dang gaussian-splat ply)
Ra  : outputs/<mau>/point_cloud_clean.ply
      outputs/<mau>/box.obj             (mode obb hoac both)
      outputs/<mau>/mesh_poisson.obj    (mode poisson hoac both)
"""
import numpy as np
import open3d as o3d
from plyfile import PlyData

from . import paths


# ---------------------------------------------------------------------------
# DOC PLY  (ho tro ca gaussian-splat f_dc va ply thuong tu colmap_mvs)
# ---------------------------------------------------------------------------

def _load_gaussian_ply(path) -> o3d.geometry.PointCloud:
    """Doc .ply tu ns-export gaussian-splat hoac colmap stereo_fusion.

    Gaussian-splat: mau luu dang f_dc_0/1/2 (spherical harmonics bac 0).
    Colmap MVS fused.ply: mau luu dang red/green/blue (uint8).
    o3d.io.read_point_cloud khong doc duoc f_dc nen phai dung plyfile.
    """
    ply = PlyData.read(str(path))
    v   = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)

    names = v.dtype.names
    if all(f"f_dc_{i}" in names for i in range(3)):
        # Gaussian splat: chuyen tu SH DC coefficient sang RGB
        C0  = 0.28209479177387814
        dc  = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
        rgb = np.clip(0.5 + C0 * dc, 0.0, 1.0)
        pc.colors = o3d.utility.Vector3dVector(rgb)
    elif all(c in names for c in ("red", "green", "blue")):
        # Colmap MVS fused.ply: mau uint8 truc tiep
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float64) / 255.0
        pc.colors = o3d.utility.Vector3dVector(rgb)

    # Loc opacity (chi co trong gaussian-splat)
    if "opacity" in names:
        opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
        keep    = np.nonzero(opacity > 0.2)[0]
        pc      = pc.select_by_index(keep)
        print(f"  loc opacity > 0.2: giu {len(keep):,}/{len(xyz):,} diem")

    return pc


# ---------------------------------------------------------------------------
# LOC NHIEU
# ---------------------------------------------------------------------------

def _keep_significant_clusters(
    pc: o3d.geometry.PointCloud,
    diag: float,
    e: dict | None = None,
) -> o3d.geometry.PointCloud:
    """Loc nhieu troi noi bang DBSCAN nhung KHONG lam mat cac phan cong/khac mat phang cua vat the.

    Thay vi chi giu 1 cum duy nhat (khien vat the phi phang bi cut/mat 50-70% so diem),
    thuat toan giu lai cum chinh cung cac cum thanh phan hop le (kich thuoc >= 12% cum chinh),
    chi loai bo nhieu troi noi thuc su.
    """
    e_dict = e or {}
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
    n_clusters = len(counts)
    if n_clusters <= 1:
        keep = np.nonzero(labels == 0)[0] if n_clusters == 1 else np.arange(len(labels))
        return pc.select_by_index(keep)

    largest = int(np.argmax(counts))
    largest_count = counts[largest]

    # Giu cum lon nhat va cac cum con co quy mo >= cluster_min_ratio so voi cum lon nhat
    min_cluster_ratio = float(e_dict.get("cluster_min_ratio", 0.12))
    threshold_count = max(min_pts, int(largest_count * min_cluster_ratio))

    keep_clusters = {c for c, cnt in enumerate(counts) if cnt >= threshold_count}
    keep_indices = np.nonzero(np.isin(labels, list(keep_clusters)))[0]

    n_removed = len(labels) - len(keep_indices)
    kept_pts = len(keep_indices)
    print(
        f"  DBSCAN (eps={eps:.4f}, min_pts={min_pts}): tim thay {n_clusters} cum. "
        f"Giu {len(keep_clusters)} cum chu yeu ({kept_pts:,}/{len(labels):,} diem), "
        f"loai {n_removed:,} diem rac/troi noi."
    )
    return pc.select_by_index(keep_indices)


def _clean(pc: o3d.geometry.PointCloud, e: dict) -> tuple[o3d.geometry.PointCloud, float]:
    """Loc thong ke + ban kinh + cluster thich ung + (tuy chon) voxel downsample."""
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        return pc, 0.0
    p5 = np.percentile(pts, 5, axis=0)
    p95 = np.percentile(pts, 95, axis=0)
    diag = float(np.linalg.norm(p95 - p5))
    if diag < 1e-3:
        diag = float(np.linalg.norm(np.asarray(pc.get_max_bound()) - np.asarray(pc.get_min_bound())))
    print(f"  duong cheo (robust) = {diag:.3f}")

    # --- loc thong ke ---
    before = len(pc.points)
    pc, _  = pc.remove_statistical_outlier(
        nb_neighbors=e["stat_nb"], std_ratio=e["stat_ratio"]
    )
    print(f"  loc thong ke  nb={e['stat_nb']} ratio={e['stat_ratio']}: "
          f"{before:,} -> {len(pc.points):,}  (loai {before-len(pc.points):,})")
    if len(pc.points) == 0:
        raise RuntimeError("Loc thong ke xoa het diem. Giam stat_ratio trong config.")

    # --- loc ban kinh ---
    rad    = e["rad_radius_frac"] * diag
    before = len(pc.points)
    pc, _  = pc.remove_radius_outlier(nb_points=e["rad_nb"], radius=rad)
    print(f"  loc ban kinh  nb={e['rad_nb']} r={rad:.4f} ({e['rad_radius_frac']}*diag): "
          f"{before:,} -> {len(pc.points):,}  (loai {before-len(pc.points):,})")
    if len(pc.points) == 0:
        raise RuntimeError("Loc ban kinh xoa het diem. Tang rad_radius_frac trong config.")

    # --- giu cac cum thuc su thuoc vat the (loai cum nhieu troi noi) ---
    pc = _keep_significant_clusters(pc, diag, e)

    # --- voxel downsample ---
    if e.get("voxel_frac", 0):
        vox    = e["voxel_frac"] * diag
        before = len(pc.points)
        pc     = pc.voxel_down_sample(vox)
        print(f"  voxel downsample {e['voxel_frac']}*diag={vox:.4f}: "
              f"{before:,} -> {len(pc.points):,}")

    return pc, diag


# ---------------------------------------------------------------------------
# LOC MAT PHANG SAN / BAN XOAY (RANSAC PLANE SEGMENTATION)
# ---------------------------------------------------------------------------

def _remove_plane_ransac(
    pc: o3d.geometry.PointCloud,
    diag: float,
    e: dict,
) -> o3d.geometry.PointCloud:
    """Tu dong phat hien va loai bo mat phang san / de ban xoay bang RANSAC.

    Giup nguoi dung thoai mai dat de hoa van tren ban xoay de dinh vi camera 360 do,
    ma van tu dong tach sach san de khoi vat the o Buoc 4 ma khong chem cheo than vat.
    """
    if not e.get("remove_table_plane", True) or len(pc.points) < 100:
        return pc

    dist_thresh = float(e.get("plane_dist_threshold_frac", 0.012)) * diag
    dist_thresh = max(dist_thresh, 0.002)

    try:
        plane_model, inliers = pc.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=1000,
        )
    except Exception:
        return pc

    n_plane = len(inliers)
    n_total = len(pc.points)
    ratio = n_plane / max(n_total, 1)

    # Chi xoa neu mat phang chiem tu 5% den 65% so diem va nam o phan day/bien
    if 0.05 <= ratio <= 0.65:
        pts = np.asarray(pc.points)
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        n_len = np.linalg.norm(normal)
        if n_len > 1e-8:
            normal = normal / n_len
            dists = np.dot(pts, normal) + (d / n_len)
            pos_ratio = np.mean(dists > dist_thresh)
            neg_ratio = np.mean(dists < -dist_thresh)

            if max(pos_ratio, neg_ratio) > 0.60:
                pc_clean = pc.select_by_index(inliers, invert=True)
                print(f"  RANSAC Plane: da loai bo {n_plane:,}/{n_total:,} diem ({ratio:.1%}) thuoc mat san/ban xoay")
                return pc_clean

    return pc


# ---------------------------------------------------------------------------
# LAM MUOT POINT CLOUD (MLS - MOVING LEAST SQUARES)
# ---------------------------------------------------------------------------

def _smooth_point_cloud_mls(
    pc: o3d.geometry.PointCloud,
    e: dict,
) -> o3d.geometry.PointCloud:
    """Moving Least Squares (MLS) point cloud pre-smoothing.

    Chieu diem ve tiep dien cuc bo dua tren k-lang gieng gan nhat.
    Giup triet tieu cac lop gon song (concentric ripples / scanline steps)
    sinh ra tu sai so do khop stereo tu nhieu goc quay camera (turntable/orbit).
    """
    mls_iter = int(e.get("mls_iter", 1))
    if mls_iter <= 0 or len(pc.points) < 30:
        return pc

    k = int(e.get("mls_neighbors", 30))
    alpha = float(e.get("mls_alpha", 0.75))
    pts = np.asarray(pc.points)

    print(f"  MLS point cloud pre-smoothing: iter={mls_iter}, k={k}, alpha={alpha}...")
    for _ in range(mls_iter):
        kdtree = o3d.geometry.KDTreeFlann(pc)
        new_pts = pts.copy()
        for i in range(len(pts)):
            _, idxs, _ = kdtree.search_knn_vector_3d(pts[i], k)
            cluster = pts[idxs]
            cent = cluster.mean(axis=0)
            centered = cluster - cent
            cov = centered.T @ centered
            vals, vecs = np.linalg.eigh(cov)
            normal = vecs[:, 0]
            proj = pts[i] - np.dot(pts[i] - cent, normal) * normal
            new_pts[i] = (1.0 - alpha) * pts[i] + alpha * proj
        pts = new_pts
        pc.points = o3d.utility.Vector3dVector(pts)

    print(f"  MLS hoan tat: da lam muot {len(pts):,} diem tren be mat point cloud.")
    return pc


# ---------------------------------------------------------------------------
# PHAP TUYEN
# ---------------------------------------------------------------------------

def _compute_normals(
    pc   : o3d.geometry.PointCloud,
    e    : dict,
    diag : float,
) -> o3d.geometry.PointCloud:
    """Tinh phap tuyen 2 vong cho vat the co be mat cong.

    Vong 1 (radius nho): bat chi tiet cong cuc bo (nut chuot, cuon scroll).
    Vong 2 (radius lon): lam nhat quan huong phap tuyen tren vung phang.
    Ket qua: trung binh co trong so giua 2 vong theo do cong (curvature).

    Vat the phang (hop): 2 vong cho ket qua giong nhau -> khong anh huong.
    Vat the cong (chuot): vong 1 bat duong cong, vong 2 on dinh huong chung.
    """
    r_small = e["normal_radius_frac"] * diag          # vong 1: chi tiet
    r_large = e["normal_radius_frac"] * diag * 2.5    # vong 2: nhat quan

    # --- Vong 1: radius nho, bat duong cong cuc bo ---
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=r_small, max_nn=e["normal_nn"]
    ))
    pc.orient_normals_consistent_tangent_plane(e["orient_k"])
    n1 = np.asarray(pc.normals).copy()

    # --- Vong 2: radius lon, on dinh huong tren vung phang ---
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=r_large, max_nn=min(e["normal_nn"] * 2, 60)
    ))
    pc.orient_normals_consistent_tangent_plane(e["orient_k"])
    n2 = np.asarray(pc.normals).copy()

    # --- Đóng nhất hướng n2 theo n1 trước khi so sánh ---
    # orient_normals_consistent_tangent_plane chọn hướng ngẫu nhiên (in/out)
    # nên n1 và n2 có thể ngược hướng nhau hoàn toàn
    dot_initial = np.einsum("ij,ij->i", n1, n2)
    n2[dot_initial < 0] *= -1

    # --- Phat hien do cong: neu n1 va n2 lech nhau nhieu -> vung cong ---
    dot      = np.einsum("ij,ij->i", n1, n2).clip(-1.0, 1.0)   # [0, 1] vi da align
    w_curve  = np.clip((1.0 - dot) / 2.0, 0.0, 1.0)[:, None]   # [0, 0.5], cao = cong
    n_blend  = w_curve * n1 + (1.0 - w_curve) * n2

    # Chuan hoa ve unit vector
    lengths  = np.linalg.norm(n_blend, axis=1, keepdims=True)
    lengths  = np.where(lengths < 1e-8, 1.0, lengths)
    n_final  = n_blend / lengths

    # Lam muot truong phap tuyen bang k-NN de loai bo gon song (ripples) tren be mat
    normal_smooth_iter = int(e.get("normal_smooth_iter", 2))
    if normal_smooth_iter > 0 and len(pc.points) > 20:
        kdtree = o3d.geometry.KDTreeFlann(pc)
        k_neighbors = min(int(e.get("normal_smooth_k", 20)), len(pc.points))
        pts_arr = np.asarray(pc.points)
        n_smooth = n_final.copy()
        for _ in range(normal_smooth_iter):
            n_next = np.zeros_like(n_smooth)
            for idx in range(len(pts_arr)):
                _, idxs, _ = kdtree.search_knn_vector_3d(pts_arr[idx], k_neighbors)
                avg_n = n_smooth[idxs].mean(axis=0)
                norm = np.linalg.norm(avg_n)
                n_next[idx] = avg_n / (norm if norm > 1e-8 else 1.0)
            n_smooth = n_next
        n_final = n_smooth
        print(f"  lam muot phap tuyen k-NN: iter={normal_smooth_iter} k={k_neighbors}")

    pc.normals = o3d.utility.Vector3dVector(n_final)

    # Bao cao chat luong
    final_len = np.linalg.norm(n_final, axis=1)
    valid     = np.mean(final_len > 0.5) * 100
    n_curved  = np.mean(w_curve > 0.3) * 100
    print(f"  phap tuyen 2 vong: r_nho={r_small:.4f} r_lon={r_large:.4f}")
    print(f"  hop le={valid:.1f}%  vung cong={n_curved:.1f}%  vung phang={100-n_curved:.1f}%")
    if valid < 80:
        print("  [!!] Nhieu phap tuyen sai. Tang orient_k hoac normal_radius_frac.")
    return pc


# ---------------------------------------------------------------------------
# MODE OBB
# ---------------------------------------------------------------------------

def _obb_mesh(pc: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    """5 mat quan sat duoc -> hop kin 6 mat."""
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
# MODE POISSON
# ---------------------------------------------------------------------------

def _poisson_mesh(pc: o3d.geometry.PointCloud, e: dict) -> o3d.geometry.TriangleMesh:
    """Poisson surface reconstruction + Laplacian smoothing cho be mat cong.

    Pipeline:
      1. Poisson reconstruction tu point cloud co phap tuyen 2 vong
      2. Cat vung mat do thap (phan ngoai suy phong ra ngoai vat the)
      3. Cat theo bbox mo rong de bo phan phong con sot
      4. Laplacian smoothing: lam min duong cong, bo rang cua do tho
      5. Tinh lai phap tuyen tren mesh (chinh xac hon phap tuyen diem)
      6. Don sach topology

    Laplacian smoothing khac Poisson:
      - Poisson: dung phap tuyen diem de dung be mat
      - Laplacian: sau khi co mesh, dich chuyen moi dinh ve trung binh
        lang gieng de lam min duong cong -> quan trong cho vat the cong
    """
    depth = e["poisson_depth"]
    print(f"  Poisson depth={depth}...")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=depth
    )
    n_before = len(mesh.vertices)

    # --- Cat mat do thap ---
    dens = np.asarray(densities)
    q    = e["poisson_density_quantile"]
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, q))
    print(f"  cat {q:.0%} mat do thap: {n_before:,} -> {len(mesh.vertices):,} dinh")

    # --- Cat theo bbox point cloud mo rong (cat phan vay Poisson ngoai vat the) ---
    pad_frac = float(e.get("poisson_hull_pad_frac", 0.01))
    bbox = pc.get_axis_aligned_bounding_box()
    pad  = pad_frac * float(np.linalg.norm(np.asarray(bbox.get_extent())))
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        bbox.min_bound - pad, bbox.max_bound + pad
    )
    mesh = mesh.crop(bbox)

    # --- Don sach topology ---
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Loai bo cac diem pinch / non-manifold vertices de bao toan 2-manifold sach
    for _ in range(5):
        nmv = mesh.get_non_manifold_vertices()
        if len(nmv) == 0:
            break
        v_mask = np.zeros(len(mesh.vertices), dtype=bool)
        v_mask[np.asarray(nmv)] = True
        mesh.remove_vertices_by_mask(v_mask)
        mesh.remove_non_manifold_edges()

    # --- Giu cac thanh phan lien thong hop le (bao toan ca nap, quai xach va than hop) ---
    tri_clusters, n_tris_per_cluster, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    n_tris_per_cluster = np.asarray(n_tris_per_cluster)
    if len(n_tris_per_cluster) > 0:
        largest = n_tris_per_cluster.argmax()
        largest_n = n_tris_per_cluster[largest]
        threshold = max(30, int(largest_n * 0.03))
        keep_clusters = set(np.where(n_tris_per_cluster >= threshold)[0])
        keep_mask = np.isin(tri_clusters, list(keep_clusters))
        before_t = len(mesh.triangles)
        mesh.remove_triangles_by_mask(~keep_mask)
        mesh.remove_unreferenced_vertices()
        print(f"  giu {len(keep_clusters)}/{len(n_tris_per_cluster)} manh hop le (>=3% cum chinh): "
              f"{before_t:,} -> {len(mesh.triangles):,} tam giac")

    # --- Mesh smoothing: lam min duong cong be mat, triet tieu gon song (ripples) ---
    smooth_type = e.get("smooth_type", "taubin").lower()
    n_iter = int(e.get("smooth_iter", 30))
    lam    = float(e.get("smooth_lambda", 0.5))
    mu     = float(e.get("smooth_mu", -0.53))

    if n_iter > 0:
        if smooth_type == "taubin":
            # Taubin smoothing khong lam teo nho (shrinkage) the tich vat the
            mesh = mesh.filter_smooth_taubin(
                number_of_iterations=n_iter,
                lambda_filter=lam,
                mu=mu,
            )
            print(f"  Taubin smooth: iter={n_iter} lambda={lam} mu={mu}")
        else:
            mesh = mesh.filter_smooth_laplacian(
                number_of_iterations=n_iter,
                lambda_filter=lam,
            )
            print(f"  Laplacian smooth: iter={n_iter} lambda={lam}")

    # Giai doan 2: Gentle surface polish (danh bong be mat)
    polish_iter = int(e.get("polish_iter", 3))
    if polish_iter > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=polish_iter)
        print(f"  Simple polish: iter={polish_iter}")

    # Chuyen mau tu point cloud goc sang mesh vertices neu co mau
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
            print("  da gan mau RGB tu point cloud vao dinh mesh")
        except Exception:
            pass

    # --- Tinh lai phap tuyen tren mesh ---
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)
    print(f"  mesh cuoi: {n_v:,} dinh, {n_t:,} tam giac  watertight={mesh.is_watertight()}")
    if n_v < 100:
        print("  [!!] Mesh qua it dinh. Giam poisson_density_quantile hoac tang poisson_depth.")
    return mesh


# ---------------------------------------------------------------------------
# DONG KIN MESH (WATERTIGHT)
# ---------------------------------------------------------------------------

def _make_watertight(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Dong kin open mesh thanh watertight bang cach bit kin truc tiep cac chu trinh bien (boundary loops).

    Khong keo tuong gia xuong truc Y (y_floor) vi truc toa do 3D co the bi nghieng va
    vat the tu do (chuot, coc...) se bi bien dang thanh buc tuong vach da dung dung.
    """
    if mesh.is_watertight():
        print("  mesh da watertight")
        return mesh

    print("  dong kin mesh tai cac duong bien...")

    # Don sach topology truoc khi xac dinh bien
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    for _ in range(5):
        nmv = mesh.get_non_manifold_vertices()
        if len(nmv) == 0:
            break
        v_mask = np.zeros(len(mesh.vertices), dtype=bool)
        v_mask[np.asarray(nmv)] = True
        mesh.remove_vertices_by_mask(v_mask)
        mesh.remove_non_manifold_edges()

    verts = np.asarray(mesh.vertices).copy()
    tris  = np.asarray(mesh.triangles).copy()
    if len(tris) == 0:
        return mesh

    # Xac dinh directed boundary edge: (u, v) xuat hien trong tam giac nhung (v, u) thi khong
    directed: dict[tuple[int, int], int] = {}
    for tri in tris:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in [(a, b), (b, c), (c, a)]:
            directed[(u, v)] = directed.get((u, v), 0) + 1

    next_map: dict[int, int] = {
        u: v for (u, v) in directed if (v, u) not in directed
    }

    if not next_map:
        print("  khong co boundary directed edge, giu nguyen mesh")
        return mesh

    # Tuyen tinh cac chu trinh bien
    remaining = dict(next_map)
    loops: list[list[int]] = []
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
        print("  khong co chu trinh bien khep kin hop le, giu nguyen mesh")
        return mesh

    print(f"  tim thay {len(loops)} chu trinh bien: {[len(l) for l in loops]} dinh")
    new_v = [verts]
    new_t = [tris]
    offset = len(verts)

    # Bit kin truc tiep tung chu trinh bien bang centroid fan cua chinh no (khong keo tuong gia)
    for loop in loops:
        k = len(loop)
        cent = verts[loop].mean(axis=0)
        c_idx = offset
        new_v.append(cent[None])
        offset += 1
        # Tam giac bit kin co huong nguoc voi huong canh bien de khau kin
        fill_tris = [[c_idx, loop[(i + 1) % k], loop[i]] for i in range(k)]
        new_t.append(np.array(fill_tris, dtype=np.int32))

    # Gop va don sach
    result = o3d.geometry.TriangleMesh()
    result.vertices  = o3d.utility.Vector3dVector(np.vstack(new_v))
    result.triangles = o3d.utility.Vector3iVector(np.vstack(new_t))
    result.remove_degenerate_triangles()
    result.remove_duplicated_vertices()
    result.compute_vertex_normals()

    wt = result.is_watertight()
    print(f"  ket qua: watertight={wt}  {len(result.vertices):,} dinh, {len(result.triangles):,} tam giac")
    return result



# ---------------------------------------------------------------------------
# NOI SUY HINH HOC 3D CHO VAT THE KHUYET MAT (BOX 6 MAT & HINH TRU 360 DO)
# ---------------------------------------------------------------------------

def _build_colored_box_mesh(
    W: float,
    D: float,
    H: float,
    pc_interp: o3d.geometry.PointCloud,
    subdiv_step: float,
    pc_raw_ortho: o3d.geometry.PointCloud | None = None,
) -> o3d.geometry.TriangleMesh:
    """Tao mesh hop 6 mat kin (watertight) mat do cao, giu nguyen mau sac & noi khoi 3D (quai xach/nap hop)."""
    # Dung mesh hop chuan cua Open3D roi subdivide de giu topology watertight 100%
    box = o3d.geometry.TriangleMesh.create_box(width=W, height=D, depth=H)
    box.translate([-W / 2.0, -D / 2.0, 0.0])
    # Subdivide midpoint 5 lan (12 -> 48 -> 192 -> 768 -> 3072 -> 12288 tam giac)
    box = box.subdivide_midpoint(number_of_iterations=5)
    box.remove_duplicated_vertices()
    box.remove_degenerate_triangles()

    verts = np.asarray(box.vertices).copy()

    # Bien dang cac dinh (mat tren, mat truoc) de giu tron ven chi tiet noi 3D (quai xach, khoa cai, ranh khia)
    if pc_raw_ortho is not None and len(pc_raw_ortho.points) > 0:
        kd_raw = o3d.geometry.KDTreeFlann(pc_raw_ortho)
        raw_pts = np.asarray(pc_raw_ortho.points)
        search_rad = max(0.18 * H, subdiv_step * 3.5)
        for i, v in enumerate(verts):
            # Mat tren (+Z): bien dang theo quai xach va mat nap that
            if v[2] > 0.85 * H:
                _, idxs, _ = kd_raw.search_hybrid_vector_3d(v, radius=search_rad, max_nn=25)
                if len(idxs) > 0:
                    cand_z = raw_pts[idxs, 2]
                    z_top = float(np.percentile(cand_z, 85))
                    if z_top > 0.75 * H:
                        verts[i, 2] = z_top
            # Mat truoc (-Y): bien dang theo khoa cai hop va gân than hop that (latch protrusion)
            elif v[1] < -0.35 * D:
                _, idxs, _ = kd_raw.search_hybrid_vector_3d(v, radius=search_rad, max_nn=25)
                if len(idxs) > 0:
                    cand_y = raw_pts[idxs, 1]
                    y_front = float(np.percentile(cand_y, 20))
                    if y_front < -0.25 * D:
                        verts[i, 1] = y_front

    box.vertices = o3d.utility.Vector3dVector(verts)

    # Gan mau tu point cloud da noi suy sang tung dinh mesh
    if pc_interp.has_colors() and len(pc_interp.points) > 0:
        kdtree = o3d.geometry.KDTreeFlann(pc_interp)
        pc_cols = np.asarray(pc_interp.colors)
        verts = np.asarray(box.vertices)
        v_cols = np.zeros_like(verts)
        for i, v in enumerate(verts):
            _, idxs, dists = kdtree.search_knn_vector_3d(v, 3)
            w = 1.0 / (np.sqrt(dists) + 1e-5)
            w /= w.sum()
            v_cols[i] = (pc_cols[idxs] * w[:, None]).sum(axis=0)
        box.vertex_colors = o3d.utility.Vector3dVector(np.clip(v_cols, 0.0, 1.0))

    # Lam mem nhe canh sac (bevel tu nhien) ma van giu phang 6 mat va khoi quai xach
    box = box.filter_smooth_taubin(number_of_iterations=3, lambda_filter=0.45, mu=-0.48)
    verts = np.asarray(box.vertices)
    verts[:, 2] -= verts[:, 2].min()
    box.vertices = o3d.utility.Vector3dVector(verts)
    box.compute_triangle_normals()
    box.compute_vertex_normals()
    return box


def _complete_box_geometry(
    pc: o3d.geometry.PointCloud,
    e: dict,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    """Noi suy 6 mat hinh hop chu nhat tu point cloud bi khuyet mat (chi thay 2-3 mat).

    Thuat toan:
      1. Loai bo diem mau la (nhu vien sach xanh ben duoi neu sot lai).
      2. Sequential Orthogonal RANSAC: tim 2 mat phang vuong goc chu dao (Mat tren + Mat truoc)
         de khoi phuc he truc toa do truc giao (X, Y, Z) chinh xac tuyet doi cua chiec hop.
      3. Xac dinh kich thuoc that (W, D, H) tu phan vi inliers cua cac mat phang.
      4. Noi suy luoi diem 3D day du cho ca 6 mat (Mat tren, Mat day, Mat truoc, Mat sau doi xung,
         Mat trai, Mat phai) kem noi suy mau RGB k-NN giu nguyen chu/hoa van tren hop.
      5. Tao dong thoi point_cloud_clean.ply (du 6 mat) va mesh hop 6 mat watertight.
    """
    pts = np.asarray(pc.points)
    cols = np.asarray(pc.colors) if pc.has_colors() else np.full_like(pts, 0.25)

    # Loc bo diem vien sach mau xanh cyan (ke ca vung bong ram) neu bi lan o mep duoi
    is_cyan = (
        (cols[:, 2] > cols[:, 0] * 1.12)
        & (cols[:, 1] > cols[:, 0] * 1.12)
        & ((cols[:, 2] - cols[:, 0]) > 0.05)
    )
    if np.mean(~is_cyan) > 0.35 and np.any(is_cyan):
        n_cyan = int(np.sum(is_cyan))
        pc = pc.select_by_index(np.nonzero(~is_cyan)[0])
        print(f"  [Noi suy Hop] Loc bo {n_cyan:,} diem mau nen/sach lan o bien")

    pc, _ = pc.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
    pts = np.asarray(pc.points)
    cols = np.asarray(pc.colors) if pc.has_colors() else np.full_like(pts, 0.25)
    diag = float(np.linalg.norm(np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)))

    # Buoc 1: RANSAC tim mat phang lon nhat (Mat nap tren)
    p1_model, inliers1 = pc.segment_plane(distance_threshold=0.015 * diag, ransac_n=3, num_iterations=2000)
    n1 = np.array(p1_model[:3], dtype=np.float64)
    n1 /= np.linalg.norm(n1)

    # Buoc 2: RANSAC tim mat phang vuong goc thu hai (Mat dung truoc)
    rem_pc = pc.select_by_index(inliers1, invert=True)
    if len(rem_pc.points) >= 50:
        p2_model, inliers2 = rem_pc.segment_plane(distance_threshold=0.015 * diag, ransac_n=3, num_iterations=2000)
        n2_raw = np.array(p2_model[:3], dtype=np.float64)
    else:
        # Fallback PCA neu chi co 1 mat
        _, eigvecs = np.linalg.eigh(np.cov(pts - pts.mean(axis=0), rowvar=False))
        n2_raw = eigvecs[:, 1]
        inliers2 = list(range(len(rem_pc.points)))

    # Truc giao hoa n2 vuong goc 90 do voi n1
    n2 = n2_raw - np.dot(n2_raw, n1) * n1
    n2_norm = np.linalg.norm(n2)
    if n2_norm < 1e-6:
        _, eigvecs = np.linalg.eigh(np.cov(pts - pts.mean(axis=0), rowvar=False))
        n2 = eigvecs[:, 1] - np.dot(eigvecs[:, 1], n1) * n1
        n2_norm = np.linalg.norm(n2)
    n2 /= n2_norm
    n3 = np.cross(n1, n2)
    n3 /= np.linalg.norm(n3)

    cand_axes = [n3, n2, n1]
    pts_cent = pts - np.median(pts, axis=0)

    # 1. Tinh do mo rong (span) cua point cloud doc theo tung truc truc giao
    spans = [float(np.percentile(pts_cent @ ax, 97) - np.percentile(pts_cent @ ax, 3)) for ax in cand_axes]

    # Phan loai mau co ban: nap cam va than den/sam, tem trang
    r_cols = cols[:, 0]
    g_cols = cols[:, 1]
    b_cols = cols[:, 2]
    is_orange = (r_cols > 0.50) & (g_cols < 0.55) & (b_cols < 0.40) & (r_cols > b_cols * 1.4)
    is_dark = (r_cols < 0.35) & (g_cols < 0.35) & (b_cols < 0.35)
    is_white = (r_cols > 0.55) & (g_cols > 0.55) & (b_cols > 0.55)

    # 2. Xac dinh truc chieu cao Z (Vertical):
    # Neu co nap cam va than den: truc Z la truc song song nhat voi vector tu than den len nap cam!
    if np.any(is_orange) and np.any(is_dark):
        lid_vec = pts[is_orange].mean(axis=0) - pts[is_dark].mean(axis=0)
        lid_vec /= (np.linalg.norm(lid_vec) + 1e-8)
        dots = [abs(np.dot(ax, lid_vec)) for ax in cand_axes]
        z_idx = int(np.argmax(dots))
    else:
        # Fallback: chieu cao cua hop nam tren ban thuong la chieu ngan nhat
        z_idx = int(np.argmin(spans))

    # 3. Hai truc con lai: truc dai hon la Chieu dai/Chieu rong W (truc X), truc con lai la Chieu sau D (truc Y)
    other_idxs = [i for i in range(3) if i != z_idx]
    if spans[other_idxs[0]] >= spans[other_idxs[1]]:
        x_idx, y_idx = other_idxs[0], other_idxs[1]
    else:
        x_idx, y_idx = other_idxs[1], other_idxs[0]

    ax_x = cand_axes[x_idx].copy()
    ax_y = cand_axes[y_idx].copy()
    ax_z = cand_axes[z_idx].copy()

    # 4. Dinh huong +Z huong len phia nap cam
    if np.any(is_orange) and np.any(is_dark) and np.dot(ax_z, lid_vec) < 0:
        ax_z = -ax_z

    # 5. Dinh huong -Y huong ve mat truoc (mat co tem/khoa cai quay ve phia camera)
    p_y = pts_cent @ ax_y
    if np.any(is_white) and np.any(is_dark):
        if np.median(p_y[is_white]) > np.median(p_y[is_dark]):
            ax_y = -ax_y

    # 6. Dam bao he truc toa do ban tay phai (det(R) = +1)
    if np.dot(np.cross(ax_x, ax_y), ax_z) < 0:
        ax_x = -ax_x

    R = np.vstack([ax_x, ax_y, ax_z])
    p_ortho = (R @ pts_cent.T).T

    # Xac dinh kich thuoc hop (W, D, H) trong he truc truc giao
    x_min, x_max = [float(v) for v in np.percentile(p_ortho[:, 0], [2, 98])]
    z_min, z_max = [float(v) for v in np.percentile(p_ortho[:, 2], [2, 98])]
    y_min = float(np.percentile(p_ortho[:, 1], 2))
    y_max = float(np.percentile(p_ortho[:, 1], 98))

    W = max(0.1, x_max - x_min)
    H = max(0.1, z_max - z_min)
    measured_D = max(0.1, y_max - y_min)
    # Bao ve chieu sau: hop do nghe / hop carton thuc te luon co chieu sau toi thieu >= 55% chieu dai W va >= 120% chieu cao H
    min_realistic_D = max(0.55 * W, 1.20 * H)
    if measured_D < min_realistic_D:
        y_max = y_min + min_realistic_D
        D = min_realistic_D
    else:
        D = measured_D

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    p_ortho[:, 0] -= cx
    p_ortho[:, 1] -= cy
    p_ortho[:, 2] -= z_min

    orange_color = np.median(cols[is_orange], axis=0) if np.any(is_orange) else np.median(cols[inliers1], axis=0)
    dark_color = np.median(cols[is_dark], axis=0) if np.any(is_dark) else np.array([0.15, 0.15, 0.15])
    lid_z = 0.72 * H

    in_bounds = (
        (p_ortho[:, 0] >= -W / 2 - 0.08 * W) & (p_ortho[:, 0] <= W / 2 + 0.08 * W) &
        (p_ortho[:, 1] >= -D / 2 - 0.08 * D) & (p_ortho[:, 1] <= D / 2 + 0.08 * D) &
        (p_ortho[:, 2] >= -0.05 * H) & (p_ortho[:, 2] <= H + 0.20 * H)
    )
    p_real = np.clip(p_ortho[in_bounds], [-W / 2, -D / 2, 0.0], [W / 2, D / 2, H * 1.25])
    c_real = cols[in_bounds]

    step = W / 65.0
    xs = np.linspace(-W / 2, W / 2, max(12, int(round(W / step))))
    ys = np.linspace(-D / 2, D / 2, max(12, int(round(D / step))))
    zs = np.linspace(0.0, H, max(10, int(round(H / step))))

    pc_real = o3d.geometry.PointCloud()
    pc_real.points = o3d.utility.Vector3dVector(p_real)
    pc_real.colors = o3d.utility.Vector3dVector(c_real)
    kdtree = o3d.geometry.KDTreeFlann(pc_real)

    pc_raw_ortho = o3d.geometry.PointCloud()
    pc_raw_ortho.points = o3d.utility.Vector3dVector(p_ortho)

    interp_pts, interp_cols, interp_norms = [], [], []

    def _add_face(u_vals, v_vals, make_xyz, normal_vec, mirror_front=False, is_top=False):
        uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")
        f_pts = make_xyz(uu.ravel(), vv.ravel())
        fallback_col = orange_color if is_top else dark_color
        f_cols = np.tile(fallback_col, (len(f_pts), 1))
        for idx_p, pt in enumerate(f_pts):
            q_pt = np.array([pt[0], -D / 2.0, pt[2]]) if mirror_front else pt
            _, idxs, dists = kdtree.search_knn_vector_3d(q_pt, 3)
            if len(dists) > 0 and dists[0] < (step * 2.5) ** 2:
                w_knn = 1.0 / (np.sqrt(dists) + 1e-5)
                w_knn /= w_knn.sum()
                f_cols[idx_p] = (c_real[idxs] * w_knn[:, None]).sum(axis=0)
            elif pt[2] >= lid_z:
                f_cols[idx_p] = orange_color
        interp_pts.append(f_pts)
        interp_cols.append(f_cols)
        interp_norms.append(np.tile(normal_vec, (len(f_pts), 1)))

    _add_face(xs, ys, lambda u, v: np.column_stack([u, v, np.full_like(u, H)]), [0, 0, 1], is_top=True)
    _add_face(xs, ys, lambda u, v: np.column_stack([u, v, np.zeros_like(u)]), [0, 0, -1])
    _add_face(xs, zs, lambda u, v: np.column_stack([u, np.full_like(u, -D / 2), v]), [0, -1, 0])
    _add_face(xs, zs, lambda u, v: np.column_stack([u, np.full_like(u, D / 2), v]), [0, 1, 0], mirror_front=True)
    _add_face(ys, zs, lambda u, v: np.column_stack([np.full_like(u, -W / 2), u, v]), [-1, 0, 0])
    _add_face(ys, zs, lambda u, v: np.column_stack([np.full_like(u, W / 2), u, v]), [1, 0, 0])

    pc_out = o3d.geometry.PointCloud()
    pc_out.points = o3d.utility.Vector3dVector(np.vstack(interp_pts))
    pc_out.colors = o3d.utility.Vector3dVector(np.vstack(interp_cols))
    pc_out.normals = o3d.utility.Vector3dVector(np.vstack(interp_norms))

    mesh_out = _build_colored_box_mesh(W, D, H, pc_out, step, pc_raw_ortho=pc_raw_ortho)
    print(
        f"  [Noi suy Hop 6 mat] Kich thuoc: W={W:.2f} x D={D:.2f} x H={H:.2f} | "
        f"PointCloud={len(pc_out.points):,} diem | Mesh={len(mesh_out.vertices):,} dinh (watertight={mesh_out.is_watertight()})"
    )
    return pc_out, mesh_out


def _complete_cylinder_geometry(
    pc: o3d.geometry.PointCloud,
    e: dict,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    """Noi suy doi xung tron xoay 360 do cho vat the hinh tru (lo, chai, coc) khi chi quet duoc 1 goc cung."""
    pts_init = np.asarray(pc.points)
    diag = float(np.linalg.norm(np.percentile(pts_init, 95, axis=0) - np.percentile(pts_init, 5, axis=0)))
    labels = np.array(pc.cluster_dbscan(eps=max(0.025 * diag, 0.01), min_points=40, print_progress=False))
    if len(labels) > 0 and np.any(labels >= 0):
        counts = np.bincount(labels[labels >= 0])
        largest = int(np.argmax(counts))
        pc = pc.select_by_index(np.nonzero(labels == largest)[0])

    pts = np.asarray(pc.points)
    cols = np.asarray(pc.colors) if pc.has_colors() else np.full_like(pts, 0.5)

    cent = pts.mean(axis=0)
    centered = pts - cent
    _, v = np.linalg.eigh(np.cov(centered, rowvar=False))
    # Truc lon nhat v[:, 2] la chieu cao Z; v[:, 1] la chieu ngang X; v[:, 0] la chieu sau Y
    R = np.vstack([v[:, 1], v[:, 0], v[:, 2]])
    if np.linalg.det(R) < 0:
        R[0] *= -1
    p_loc = (R @ centered.T).T

    z_min, z_max = np.percentile(p_loc[:, 2], [2, 98])
    H = max(0.1, float(z_max - z_min))
    p_loc[:, 2] = np.clip(p_loc[:, 2] - z_min, 0.0, H)

    # Dinh huong dau hep (nap lo/co chai) len phia +Z
    bot_pts = p_loc[p_loc[:, 2] < 0.30 * H, 0]
    top_pts = p_loc[p_loc[:, 2] > 0.70 * H, 0]
    if len(bot_pts) > 10 and len(top_pts) > 10 and np.ptp(top_pts) > np.ptp(bot_pts):
        p_loc[:, 2] = H - p_loc[:, 2]
        p_loc[:, 0] *= -1

    # Can tam X theo tung lat cat Z va tinh ban kinh r(z)
    n_z = 60
    z_bins = np.linspace(0.0, H, n_z + 1)
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    radii = np.zeros(n_z, dtype=np.float64)
    z_colors = np.zeros((n_z, 3), dtype=np.float64)
    base_col = np.median(cols, axis=0)

    for i in range(n_z):
        mask = (p_loc[:, 2] >= z_bins[i]) & (p_loc[:, 2] < z_bins[i + 1])
        if np.sum(mask) >= 5:
            x_slice = p_loc[mask, 0]
            radii[i] = 0.5 * float(np.percentile(x_slice, 95) - np.percentile(x_slice, 5))
            z_colors[i] = np.median(cols[mask], axis=0)
        else:
            radii[i] = np.nan
            z_colors[i] = base_col

    valid = ~np.isnan(radii)
    if np.any(valid):
        radii = np.interp(z_centers, z_centers[valid], radii[valid])
    else:
        radii[:] = 1.0

    # Chuan hoa hinh dang than tru + co + nap lo (tranh bi thắt eo gia do thieu diem o day)
    r_body = float(np.percentile(radii[: int(0.75 * n_z)], 75))
    r_cap = float(np.median(radii[int(0.80 * n_z) :]))
    r_cap = np.clip(r_cap, 0.55 * r_body, 0.95 * r_body)
    for i, zc in enumerate(z_centers):
        zf = zc / H
        if zf < 0.72:
            radii[i] = 0.75 * r_body + 0.25 * max(radii[i], 0.85 * r_body)
        elif zf < 0.82:
            t_sh = (zf - 0.72) / 0.10
            target_r = (1.0 - t_sh) * r_body + t_sh * r_cap
            radii[i] = 0.7 * target_r + 0.3 * radii[i]
        else:
            radii[i] = 0.8 * r_cap + 0.2 * max(radii[i], 0.7 * r_cap)

    kernel = np.ones(5) / 5.0
    radii = np.convolve(np.pad(radii, (2, 2), mode="edge"), kernel, mode="valid")
    radii = np.maximum(radii, 0.05 * H)

    # Quay 360 do quanh truc Z tao luoi dinh va tam giac watertight
    n_theta = 72
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    verts, v_cols, norms = [], [], []
    for iz, (zc, rc) in enumerate(zip(z_centers, radii)):
        for th in thetas:
            cs, sn = np.cos(th), np.sin(th)
            verts.append([rc * cs, rc * sn, zc])
            v_cols.append(z_colors[iz])
            norms.append([cs, sn, 0.0])

    # Them tam day (z=0) va tam dinh (z=H) de dong kin watertight
    idx_bot_center = len(verts)
    verts.append([0.0, 0.0, float(z_centers[0])])
    v_cols.append(z_colors[0])
    norms.append([0.0, 0.0, -1.0])

    idx_top_center = len(verts)
    verts.append([0.0, 0.0, float(z_centers[-1])])
    v_cols.append(z_colors[-1])
    norms.append([0.0, 0.0, 1.0])

    tris = []
    for iz in range(n_z - 1):
        row0 = iz * n_theta
        row1 = (iz + 1) * n_theta
        for it in range(n_theta):
            it_next = (it + 1) % n_theta
            v00 = row0 + it
            v01 = row0 + it_next
            v10 = row1 + it
            v11 = row1 + it_next
            tris.append([v00, v01, v11])
            tris.append([v00, v11, v10])

    # Dong nap day va nap dinh
    for it in range(n_theta):
        it_next = (it + 1) % n_theta
        tris.append([idx_bot_center, it_next, it])
        top_row = (n_z - 1) * n_theta
        tris.append([idx_top_center, top_row + it, top_row + it_next])

    verts_arr = np.array(verts, dtype=np.float64)
    verts_arr[:, 2] -= verts_arr[:, 2].min()
    cols_arr = np.array(v_cols, dtype=np.float64)
    norms_arr = np.array(norms, dtype=np.float64)

    pc_out = o3d.geometry.PointCloud()
    pc_out.points = o3d.utility.Vector3dVector(verts_arr)
    pc_out.colors = o3d.utility.Vector3dVector(cols_arr)
    pc_out.normals = o3d.utility.Vector3dVector(norms_arr)

    mesh_out = o3d.geometry.TriangleMesh()
    mesh_out.vertices = o3d.utility.Vector3dVector(verts_arr)
    mesh_out.triangles = o3d.utility.Vector3iVector(np.array(tris, dtype=np.int32))
    mesh_out.vertex_colors = o3d.utility.Vector3dVector(cols_arr)
    mesh_out = mesh_out.filter_smooth_taubin(number_of_iterations=10, lambda_filter=0.5, mu=-0.53)
    mesh_out.compute_triangle_normals()
    mesh_out.compute_vertex_normals()

    print(
        f"  [Noi suy Hinh tru 360°] Ban kinh tb={radii.mean():.2f}, Cao H={H:.2f} | "
        f"PointCloud={len(verts_arr):,} diem | Mesh={len(mesh_out.vertices):,} dinh (watertight={mesh_out.is_watertight()})"
    )
    return pc_out, mesh_out


# ---------------------------------------------------------------------------
# CAN THANG TRUC & DAT LEN MAT SAN (AUTO-UPRIGHT & GROUNDING)
# ---------------------------------------------------------------------------

def _align_upright_and_ground(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Can thang dung vat the (Z-up), nam ngang phang tren san (Z=0) va can tam (X=0, Y=0)."""
    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        return mesh

    # Neu mesh da duoc can thang truc (vi du tu buoc noi suy hop 6 mat hoac hinh tru 360),
    # chi can dam bao tam X=0, Y=0 va sat san Z=0 ma khong xoay lech truc!
    mesh.compute_triangle_normals()
    tri_norms = np.asarray(mesh.triangle_normals)
    already_grounded = (
        abs(float(verts[:, 2].min())) < 1e-2
        and abs(float(verts[:, 0].mean())) < 5e-2
        and abs(float(verts[:, 1].mean())) < 5e-2
    )
    if len(tri_norms) > 0:
        axis_aligned_ratio = float(np.mean(np.max(np.abs(tri_norms), axis=1) > 0.94))
        if axis_aligned_ratio > 0.70 or already_grounded:
            v_fixed = verts.copy()
            v_fixed[:, 0] -= 0.5 * (v_fixed[:, 0].min() + v_fixed[:, 0].max())
            v_fixed[:, 1] -= 0.5 * (v_fixed[:, 1].min() + v_fixed[:, 1].max())
            v_fixed[:, 2] -= v_fixed[:, 2].min()
            mesh.vertices = o3d.utility.Vector3dVector(v_fixed)
            mesh.compute_triangle_normals()
            mesh.compute_vertex_normals()
            print(
                f"  [Can truc PCA] Giu nguyen he truc da can chuan: "
                f"X={v_fixed[:, 0].ptp():.2f}, Y={v_fixed[:, 1].ptp():.2f}, Z={v_fixed[:, 2].max():.2f} (sat san Z=0)"
            )
            return mesh

    m_cent = verts.mean(axis=0)
    centered = verts - m_cent

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # eigh sap xep tri rieng tang dan:
    # Kiem tra vat the tron xoay/hinh tru dung (lo, chai, coc) vs vat the nam ngang (chuot, hop dep):
    # O vat the tron xoay dung: 2 truc tiet dien ngang xap xi nhau (eigvals[0] ≈ eigvals[1]),
    # con truc cao (Z) la truc doi xung (thuong la truc lon nhat eigvecs[:, 2]).
    ratio_01 = float(eigvals[1] / max(eigvals[0], 1e-8))
    ratio_12 = float(eigvals[2] / max(eigvals[1], 1e-8))

    is_upright_cylinder = (ratio_01 < 1.35 and ratio_12 > 1.25)

    if is_upright_cylinder:
        # Dang hinh tru dung: truc lon nhat la chieu cao dung (Z)
        u_h = eigvecs[:, 2].copy()
        u_w = eigvecs[:, 0].copy()
        u_l = eigvecs[:, 1].copy()
    else:
        # Dang nam ngang (chuot, hop): truc nho nhat la chieu cao dung (Z)
        u_h = eigvecs[:, 0].copy()
        u_w = eigvecs[:, 1].copy()
        u_l = eigvecs[:, 2].copy()

    # 1. Xac dinh huong UP (huong len vom cong/nap tren)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    h_coords = np.dot(centered, u_h)
    n_dot_h = np.dot(normals, u_h)

    # 20% dinh tren cao nhat cua vom cong phai co phap tuyen huong ra ngoai cung chieu UP
    top_mask = h_coords > np.percentile(h_coords, 80)
    if n_dot_h[top_mask].mean() < 0:
        u_h = -u_h

    # 2. Xac dinh chieu dai (Y):
    if not is_upright_cylinder:
        # Dau chuot (phim bam) thuong hep hon duoi chuot (long ban tay)
        l_coords = np.dot(centered, u_l)
        w_coords = np.dot(centered, u_w)
        l_min, l_max = l_coords.min(), l_coords.max()
        pts_neg_l = w_coords[l_coords < l_min + 0.3 * (l_max - l_min)]
        pts_pos_l = w_coords[l_coords > l_max - 0.3 * (l_max - l_min)]

        w_neg = pts_neg_l.ptp() if len(pts_neg_l) > 0 else 0
        w_pos = pts_pos_l.ptp() if len(pts_pos_l) > 0 else 0

        # Chuan hoa: dau chuot huong ve -Y (phim bam), duoi chuot huong ve +Y
        if w_pos < w_neg:
            u_l = -u_l

    # 3. Truc ngang (X) tao he truc toa do ban tay phai: X = Y x Z
    u_w = np.cross(u_l, u_h)
    u_w = u_w / np.linalg.norm(u_w)
    u_l = np.cross(u_h, u_w)
    u_l = u_l / np.linalg.norm(u_l)

    # Ma tran xoay 3x3 truc giao SO(3)
    R = np.vstack([u_w, u_l, u_h])

    v_rot = (R @ centered.T).T

    # 4. Can tam X, Y ve 0 va dat day tiep xuc sat san Z=0
    z_min = v_rot[:, 2].min()
    v_rot[:, 0] -= v_rot[:, 0].mean()
    v_rot[:, 1] -= v_rot[:, 1].mean()
    v_rot[:, 2] -= z_min

    aligned_mesh = o3d.geometry.TriangleMesh()
    aligned_mesh.vertices = o3d.utility.Vector3dVector(v_rot)
    aligned_mesh.triangles = mesh.triangles
    if mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices):
        aligned_mesh.vertex_colors = mesh.vertex_colors
    aligned_mesh.compute_triangle_normals()
    aligned_mesh.compute_vertex_normals()

    print(f"  [Can truc PCA] Hoan tat: X={v_rot[:, 0].ptp():.2f}, Y={v_rot[:, 1].ptp():.2f}, Z={v_rot[:, 2].max():.2f} (sat san Z=0)")
    return aligned_mesh


# ---------------------------------------------------------------------------
# XUAT .OBJ
# ---------------------------------------------------------------------------

def _export_obj(mesh: o3d.geometry.TriangleMesh, obj_path) -> None:
    has_colors = mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices)
    o3d.io.write_triangle_mesh(
        str(obj_path),
        mesh,
        write_ascii=True,
        write_vertex_normals=True,
        write_vertex_colors=has_colors,
    )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run(sample_name: str, cfg: dict) -> None:
    e    = cfg["mesh_export"]
    mode = e.get("mesh_mode", "obb")
    if mode not in ("obb", "poisson", "both"):
        raise ValueError(f'mesh_mode phai la "obb", "poisson", hoac "both", dang la "{mode}"')

    out_dir  = paths.output_dir(sample_name)
    ply_path = out_dir / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Khong thay {ply_path}, chay Buoc 3 truoc.")

    # --- Doc ---
    print("[Buoc 4] Doc point cloud...")
    pc = _load_gaussian_ply(ply_path)
    n_raw = len(pc.points)
    print(f"  {n_raw:,} diem ban dau")
    if n_raw == 0:
        raise RuntimeError("Khong co diem nao. Kiem tra file .ply tu Buoc 3.")

    # Xac dinh loai vat the de kich hoat Noi suy hinh hoc 3D phu hop
    prompt = str(cfg.get("masking", {}).get("prompt", "object")).strip().lower()
    prefix = sample_name.split("_")[0].lower()
    cyl_keywords = ("jar", "bottle", "cup", "can", "vase", "mug")
    box_keywords = ("box", "carton", "package", "cube")

    is_cyl_target = prefix in cyl_keywords or any(k in prompt for k in cyl_keywords)
    is_box_target = (
        not is_cyl_target
        and prefix not in ("mouse",)
        and (
            mode == "obb"
            or prefix in box_keywords
            or any(k in prompt for k in box_keywords)
        )
    )
    interp_enabled = bool(e.get("interpolate_surface", True))

    obj_path = out_dir / f"{sample_name}.obj"
    clean_ply = out_dir / "point_cloud_clean.ply"

    if interp_enabled and is_box_target:
        # Voi hinh hop: goi truc tiep _complete_box_geometry de giu tron ven cum chu mat truoc & vien canh
        print("[Buoc 4] Loc nhieu & Noi suy 6 mat hinh hop (Orthogonal RANSAC + Surface Interpolation)...")
        pc, mesh = _complete_box_geometry(pc, e)
        o3d.io.write_point_cloud(str(clean_ply), pc)
        print(f"  da luu point cloud da noi suy 6 mat -> {clean_ply.name}")

    elif interp_enabled and is_cyl_target:
        print("[Buoc 4] Loc nhieu & Noi suy doi xung tron xoay 360° (Cylindrical Interpolation)...")
        pc, diag = _clean(pc, e)
        if e.get("remove_table_plane", True):
            pc = _remove_plane_ransac(pc, diag, e)
        pc, mesh = _complete_cylinder_geometry(pc, e)
        o3d.io.write_point_cloud(str(clean_ply), pc)
        print(f"  da luu point cloud da noi suy 360° -> {clean_ply.name}")

    else:
        # --- Loc nhieu + giu cluster lon nhat ---
        print("[Buoc 4] Loc nhieu...")
        pc, diag = _clean(pc, e)

        # --- Tu dong loai bo mat phang san / ban xoay (RANSAC) ---
        if e.get("remove_table_plane", True):
            pc = _remove_plane_ransac(pc, diag, e)

        if e.get("floor_cut_frac", 0) > 0:
            pts = np.asarray(pc.points)
            y_min = pts[:, 1].min()
            y_max = pts[:, 1].max()
            threshold = y_min + e["floor_cut_frac"] * (y_max - y_min)
            keep = np.where(pts[:, 1] > threshold)[0]
            pc = pc.select_by_index(keep)
            print(f"  cat nen: loai {len(pts)-len(keep):,} diem duoi y={threshold:.4f}")

        # --- MLS Point Cloud Pre-smoothing (triet tieu gon song toa do tu camera) ---
        if e.get("mls_iter", 1) > 0 and len(pc.points) > 30:
            print("[Buoc 4] Lam muot toa do diem (MLS)...")
            pc = _smooth_point_cloud_mls(pc, e)

        # Luu point cloud sach de debug trong Blender
        o3d.io.write_point_cloud(str(clean_ply), pc)
        print(f"  da luu point cloud sach -> {clean_ply.name}")

        # --- Phap tuyen (dung cho Poisson va Both) ---
        need_normals = mode in ("poisson", "both")
        if need_normals:
            print("[Buoc 4] Tinh vector phap tuyen...")
            pc = _compute_normals(pc, e, diag)

        if mode == "obb":
            print("[Buoc 4] OBB fitting...")
            mesh = _obb_mesh(pc)

        elif mode == "poisson":
            print("[Buoc 4] Poisson surface reconstruction...")
            mesh = _poisson_mesh(pc, e)
            if e.get("make_watertight", False):
                print("[Buoc 4] Dong kin mesh (watertight)...")
                mesh = _make_watertight(mesh)

        else:  # mode == "both": Poisson (be mat thuc) + OBB (hop tham chieu)
            print("[Buoc 4] OBB fitting (hop tham chieu)...")
            obb = _obb_mesh(pc)
            print("[Buoc 4] Poisson surface reconstruction (be mat thuc te)...")
            poi = _poisson_mesh(pc, e)
            if e.get("make_watertight", False):
                print("[Buoc 4] Dong kin Poisson mesh...")
                poi = _make_watertight(poi)
            mesh = obb + poi   # gop OBB + Poisson
            mesh.compute_vertex_normals()
            print(f"  gop mesh: {len(mesh.vertices):,} dinh, {len(mesh.triangles):,} tam giac")

        # --- Can thang dung (Z-up), can tam (X=0, Y=0) va dat day mesh len san (Z=0) ---
        if e.get("align_upright", True):
            print("[Buoc 4] Can thang truc (Z-up) & dat len mat san (Z=0)...")
            mesh = _align_upright_and_ground(mesh)

    # --- Xuat ban dung khoi tao ---
    print(f"[Buoc 4] Xuat {obj_path.name}...")
    _export_obj(mesh, obj_path)
    print(f"[Buoc 4] Xong -> {obj_path}  ({obj_path.stat().st_size/1024:.1f} KB)")

    # Luu ban goc raw truoc khi tinh chinh de de dang so sanh A/B trong Blender
    raw_path = out_dir / f"{sample_name}_raw.obj"
    import shutil
    shutil.copyfile(str(obj_path), str(raw_path))
    print(f"  da luu ban goc -> {raw_path.name}")

    # Tu dong goi Buoc 5 (Mesh Refine Pipeline) neu duoc bat
    if cfg.get("mesh_refine", {}).get("auto_run", True):
        from . import step5_mesh_refine
        step5_mesh_refine.run(sample_name, cfg)
    else:
        print(f"  Blender: File > Import > Wavefront (.obj) > {obj_path.name}")