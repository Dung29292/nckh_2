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

    # --- Giu thanh phan lien thong lon nhat (loai manh mesh roi rac do nhieu) ---
    tri_clusters, n_tris_per_cluster, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    n_tris_per_cluster = np.asarray(n_tris_per_cluster)
    if len(n_tris_per_cluster) > 0:
        largest = n_tris_per_cluster.argmax()
        before_t = len(mesh.triangles)
        mesh.remove_triangles_by_mask(tri_clusters != largest)
        mesh.remove_unreferenced_vertices()
        print(f"  giu thanh phan lon nhat: {len(n_tris_per_cluster)} manh, "
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

    # --- Loc nhieu + giu cluster lon nhat ---
    print("[Buoc 4] Loc nhieu...")
    pc, diag = _clean(pc, e)
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
    clean_ply = out_dir / "point_cloud_clean.ply"
    o3d.io.write_point_cloud(str(clean_ply), pc)
    print(f"  da luu point cloud sach -> {clean_ply.name}")

    # --- Phap tuyen (dung cho Poisson va Both) ---
    need_normals = mode in ("poisson", "both")
    if need_normals:
        print("[Buoc 4] Tinh vector phap tuyen...")
        pc = _compute_normals(pc, e, diag)

    # --- Dung mesh + xuat 1 file duy nhat ten theo ten mau ---
    # Vi du: mau "mouse_01" -> mouse_01.obj | mau "box_01" -> box_01.obj
    obj_path = out_dir / f"{sample_name}.obj"

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