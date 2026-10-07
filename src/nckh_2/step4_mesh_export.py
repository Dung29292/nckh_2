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

def _keep_largest_cluster(pc: o3d.geometry.PointCloud, diag: float) -> o3d.geometry.PointCloud:
    """Giu lai cluster lon nhat sau DBSCAN.

    Xu ly truc tiep cum nhieu troi noi (nhu cum den o phia tren trong Blender):
    chung la cluster nho tach biet khoi than chinh cua vat the.

    eps = 1.5% duong cheo: 2 diem trong vong nay duoc xem la cung cluster.
    min_points = 50: cluster < 50 diem bi bo.
    """
    eps        = 0.015 * diag
    labels     = np.array(pc.cluster_dbscan(eps=eps, min_points=50, print_progress=False))
    n_clusters = labels.max() + 1

    if n_clusters <= 1:
        return pc   # chi co 1 cluster hoac khong cluster nao -> giu nguyen

    # Tim cluster co nhieu diem nhat (khong tinh nhieu = label -1)
    counts     = np.bincount(labels[labels >= 0])
    largest    = int(np.argmax(counts))
    keep       = np.nonzero(labels == largest)[0]
    n_removed  = len(labels) - len(keep)
    print(f"  DBSCAN: tim thay {n_clusters} cluster, giu cluster lon nhat "
          f"({counts[largest]:,} diem), loai {n_removed:,} diem nhieu/cum nho")
    return pc.select_by_index(keep)


def _clean(pc: o3d.geometry.PointCloud, e: dict) -> tuple[o3d.geometry.PointCloud, float]:
    """Loc thong ke + ban kinh + cluster + (tuy chon) voxel downsample."""
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

    # --- giu cluster lon nhat (loai cum nhieu troi noi) ---
    pc = _keep_largest_cluster(pc, diag)

    # --- voxel downsample ---
    if e.get("voxel_frac", 0):
        vox    = e["voxel_frac"] * diag
        before = len(pc.points)
        pc     = pc.voxel_down_sample(vox)
        print(f"  voxel downsample {e['voxel_frac']}*diag={vox:.4f}: "
              f"{before:,} -> {len(pc.points):,}")

    return pc, diag


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

    # --- Cat theo bbox point cloud mo rong ---
    bbox = pc.get_axis_aligned_bounding_box()
    pad  = 0.02 * float(np.linalg.norm(np.asarray(bbox.get_extent())))
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        bbox.min_bound - pad, bbox.max_bound + pad
    )
    mesh = mesh.crop(bbox)

    # --- Don sach topology ---
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
        # --- Giu thanh phan lien thong lon nhat (loai manh mesh roi rac do nhieu) ---
    tri_clusters, n_tris_per_cluster, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    n_tris_per_cluster = np.asarray(n_tris_per_cluster)
    largest = n_tris_per_cluster.argmax()
    before_t = len(mesh.triangles)
    mesh.remove_triangles_by_mask(tri_clusters != largest)
    mesh.remove_unreferenced_vertices()
    print(f"  giu thanh phan lon nhat: {len(n_tris_per_cluster)} manh, "
          f"{before_t:,} -> {len(mesh.triangles):,} tam giac")

    # --- Laplacian smoothing: lam min duong cong be mat ---
    # iter cao hon = min hon, nhung co the mat chi tiet (nut chuot, goc)
    # lambda_ = cuong do moi buoc dich chuyen (0.5 = trung binh voi lang gieng)
    n_iter   = e.get("smooth_iter",   5)
    lam      = e.get("smooth_lambda", 0.5)
    if n_iter > 0:
        mesh = mesh.filter_smooth_laplacian(
            number_of_iterations=n_iter,
            lambda_filter=lam,
        )
        print(f"  Laplacian smooth: iter={n_iter} lambda={lam}")

    # --- Tinh lai phap tuyen TREN MESH (chinh xac hon phap tuyen diem) ---
    # compute_vertex_normals dung trung binh co trong so cua phap tuyen tam giac
    # -> bat duong cong dung cua mesh, khong phai uoc tinh tu PCA tren diem
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

def _make_watertight(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:  # noqa: C901
    """Dong kin open mesh thanh watertight de co the tai hien vat the.

    Vat the scan tu 1 phia (vi du: quan ao, chuot may tinh dat tren ban) cho
    Poisson mesh ho o mat duoi. Ham nay dong kin bang:
      - Tim boundary loop (duong vien ho cua mesh)
      - Tao tuong ben noi boundary xuong san (y_min)
      - Tao mat day phang bang fan triangulation

    Ket qua: mesh watertight co the dung cho tai hien / in 3D.
    """
    if mesh.is_watertight():
        print("  mesh da watertight")
        return mesh

    print("  them tuong ben + mat day de dong kin mesh...")
    verts = np.asarray(mesh.vertices).copy()
    tris  = np.asarray(mesh.triangles).copy()

    # --- Traverse boundary loops theo DIRECTED EDGE ---
    # Moi tam giac (a,b,c) dong gop canh huong (a->b), (b->c), (c->a).
    # Boundary edge: (u->v) co trong tam giac nhung (v->u) thi khong.
    # => moi dinh boundary chi co duy nhat 1 canh di ra -> next_map[u] = v
    directed: dict[tuple[int,int], int] = {}
    for tri in tris:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in [(a,b),(b,c),(c,a)]:
            directed[(u,v)] = directed.get((u,v),0) + 1
    next_map: dict[int,int] = {
        u: v for (u,v) in directed if (v,u) not in directed
    }

    if not next_map:
        print("  khong co boundary directed edge, giu nguyen mesh")
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
        loops.append(loop)

    loops = [lp for lp in loops if len(lp) >= 3]  # loai loop qua ngan
    loop_sizes = [len(lp) for lp in loops]
    print(f"  tim thay {len(loops)} boundary loop(s): {loop_sizes}")

    # --- Them tuong ben + mat day cho moi loop ---
    # Loop lon nhat = bien gioi chinh (mat duoi ho) -> tao tuong ben + mat day phang
    # Loop nho con lai = lo noi tai tren be mat -> fan fill tai chỗ (khong can tuong ben)
    y_floor = float(verts[:, 1].min())
    new_v: list[np.ndarray] = [verts]
    new_t: list[np.ndarray] = [tris]
    offset = len(verts)

    if not loops:
        return mesh

    # Sap xep loop theo kich thuoc giam dan
    loops_sorted = sorted(loops, key=len, reverse=True)
    main_loop  = loops_sorted[0]    # loop lon nhat: mat day chinh
    inner_loops = loops_sorted[1:]  # loop nho: lo noi tai

    # --- Xu ly main loop: tuong ben + mat day phang ---
    def _add_floor_loop(loop: list[int]) -> None:
        n = len(loop)
        loop_xyz = verts[loop]
        cap_xyz  = loop_xyz.copy()
        cap_xyz[:, 1] = y_floor

        centroid  = cap_xyz.mean(axis=0)
        cap_start = offset
        cent_idx  = offset + n

        new_v.append(cap_xyz)
        new_v.append(centroid[None])

        # Tuong ben
        side: list[list[int]] = []
        for i in range(n):
            a  = loop[i];      b  = loop[(i + 1) % n]
            ca = cap_start + i; cb = cap_start + (i + 1) % n
            side.append([a,  cb, ca])
            side.append([a,  b,  cb])
        new_t.append(np.array(side, dtype=np.int32))

        # Mat day: fan tu centroid sap xep theo goc x-z
        xz  = cap_xyz[:, [0, 2]]; cxz = centroid[[0, 2]]
        ang = np.arctan2(xz[:, 1] - cxz[1], xz[:, 0] - cxz[0])
        ord_ = np.argsort(ang)
        ft: list[list[int]] = []
        for i in range(n):
            ci = cap_start + ord_[i]; cj = cap_start + ord_[(i + 1) % n]
            ft.append([cent_idx, cj, ci])
        new_t.append(np.array(ft, dtype=np.int32))

    def _add_floor_loop(loop: list[int]) -> None:
        """Tao tuong ben + mat day phang cho loop chinh (bien gioi ngoai)."""
        nonlocal offset
        n = len(loop)
        loop_xyz = verts[loop]
        cap_xyz  = loop_xyz.copy()
        cap_xyz[:, 1] = y_floor
        centroid  = cap_xyz.mean(axis=0)
        cap_start = offset
        cent_idx  = offset + n
        new_v.append(cap_xyz)
        new_v.append(centroid[None])
        offset += n + 1
        # Tuong ben (phap tuyen huong ra ngoai)
        side: list[list[int]] = []
        for i in range(n):
            a=loop[i]; b=loop[(i+1)%n]; ca=cap_start+i; cb=cap_start+(i+1)%n
            side.append([a, ca, cb])
            side.append([a, cb, b])
        new_t.append(np.array(side, dtype=np.int32))
        # Mat day (phap tuyen huong xuong)
        xz=cap_xyz[:,[0,2]]; cxz=centroid[[0,2]]
        ang=np.arctan2(xz[:,1]-cxz[1], xz[:,0]-cxz[0])
        ord_=np.argsort(ang)
        ft=[[cent_idx, cap_start+ord_[(i+1)%n], cap_start+ord_[i]] for i in range(n)]
        new_t.append(np.array(ft, dtype=np.int32))

    def _add_inner_fill(loop: list[int]) -> None:
        """Fan fill lo noi tai tren be mat (boundary directed -> winding thuan)."""
        nonlocal offset
        n = len(loop)
        loop_xyz = verts[loop]
        centroid = loop_xyz.mean(axis=0)
        cent_idx = offset
        new_v.append(centroid[None])
        offset += 1
        xz=loop_xyz[:,[0,2]]; cxz=centroid[[0,2]]
        ang=np.arctan2(xz[:,1]-cxz[1], xz[:,0]-cxz[0])
        ord_=np.argsort(ang)
        # Winding thuan (directed boundary: u->v => tam giac [cent, u, v])
        ft=[[cent_idx, loop[ord_[i]], loop[ord_[(i+1)%n]]] for i in range(n)]
        new_t.append(np.array(ft, dtype=np.int32))

    print(f"  main loop: {len(main_loop)} dinh (tuong ben + mat day)")
    _add_floor_loop(main_loop)
    for il in inner_loops:
        print(f"  inner loop: {len(il)} dinh (fan fill)")
        _add_inner_fill(il)



    # --- Gop va don sach ---
    result = o3d.geometry.TriangleMesh()
    result.vertices  = o3d.utility.Vector3dVector(np.vstack(new_v))
    result.triangles = o3d.utility.Vector3iVector(np.vstack(new_t))
    result.remove_degenerate_triangles()
    result.remove_duplicated_vertices()
    # NOTE: khong goi remove_non_manifold_edges vi se pha cac cap moi them vao
    result.compute_vertex_normals()

    wt = result.is_watertight()
    print(f"  ket qua: watertight={wt}  "
          f"{len(result.vertices):,} dinh, {len(result.triangles):,} tam giac")
    if not wt:
        print("  [!!] Van chua watertight (co the co loop tu giao).")
        print("  [!!] Thu giam poisson_density_quantile hoac tang poisson_depth.")
    return result


# ---------------------------------------------------------------------------
# XUAT .OBJ
# ---------------------------------------------------------------------------

def _export_obj(mesh: o3d.geometry.TriangleMesh, obj_path) -> None:
    o3d.io.write_triangle_mesh(
        str(obj_path),
        mesh,
        write_ascii=True,
        write_vertex_normals=True,
        write_vertex_colors=False,
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
        print("[Buoc 4] Dong kin mesh (watertight)...")
        mesh = _make_watertight(mesh)

    else:  # mode == "both": Poisson (be mat thuc) + dong kin = 1 file kín
        print("[Buoc 4] OBB fitting (hop tham chieu)...")
        obb = _obb_mesh(pc)
        print("[Buoc 4] Poisson surface reconstruction (be mat thuc te)...")
        poi = _poisson_mesh(pc, e)
        print("[Buoc 4] Dong kin Poisson mesh...")
        poi = _make_watertight(poi)
        mesh = obb + poi   # gop OBB + Poisson watertight
        mesh.compute_vertex_normals()
        print(f"  gop mesh: {len(mesh.vertices):,} dinh, {len(mesh.triangles):,} tam giac")

    # --- Xuat ---
    print(f"[Buoc 4] Xuat {obj_path.name}...")
    _export_obj(mesh, obj_path)
    print(f"[Buoc 4] Xong -> {obj_path}  ({obj_path.stat().st_size/1024:.1f} KB)")
    print(f"  Blender: File > Import > Wavefront (.obj) > {obj_path.name}")