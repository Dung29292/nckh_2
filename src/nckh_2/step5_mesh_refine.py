"""Buoc 5: Pipeline tinh chinh & lam min mesh chuyen sau (.obj).

Tach rieng thanh mot pipeline doc lap giup:
  1. Tinh chinh do min, triet tieu cac nep gon song (contour ripples / terraces) cuc nhanh (1-2 giay).
  2. Nguoi dung co the thu nghiem va tinh chinh cac thong so lam min (Taubin, Polish, Subdivide)
     ma KHONG can chay lai COLMAP hay Poisson ton thoi gian.
  3. Xuat dong thoi file <sample_name>_refined.obj va cap nhat <sample_name>.obj
     (dong thoi luu ban goc thanh <sample_name>_raw.obj de de dang so sanh A/B trong Blender).

Vao : outputs/<mau>/<mau>.obj  (hoac <mau>_raw.obj neu da tao truoc do)
Ra  : outputs/<mau>/<mau>_refined.obj
      outputs/<mau>/<mau>.obj      (ban tinh chinh muot nhat)
      outputs/<mau>/<mau>_raw.obj  (ban goc truoc khi tinh chinh)
"""
import shutil
import time
import open3d as o3d
import numpy as np

from . import paths
from .completion.detail_smooth import (
    bilateral_normal_smooth,
    interpolate_point_confidence_to_mesh,
    pymeshlab_two_step_smooth,
    transfer_observed_detail_residuals,
)


def refine_mesh(
    mesh: o3d.geometry.TriangleMesh,
    r_cfg: dict,
    ref_points: np.ndarray | None = None,
    ref_confidence: np.ndarray | None = None,
) -> o3d.geometry.TriangleMesh:
    """Pipeline lam min da tang bao toan chi tiet:

    Giai doan 1 (Tuy chon): Subdivide tang do phan giai luoi tam giac
    Giai doan 2: Lam min giu canh sac net (``bilateral_normal`` | ``pymeshlab_twostep`` | ``taubin``)
    Giai doan 3 (Tuy chon): Re-inject micro-details tu diem quan sat thuc te (w >= 0.85)
    Giai doan 4: Can thang truc & bao toan mau sac / phap tuyen
    """
    has_colors = mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices)
    v_colors = np.asarray(mesh.vertex_colors).copy() if has_colors else None

    # --- Giai doan 1: Subdivide (tuy chon) ---
    if r_cfg.get("subdivide", False):
        s_method = r_cfg.get("subdivide_method", "midpoint").lower()
        before_v = len(mesh.vertices)
        if s_method == "loop":
            mesh = mesh.subdivide_loop(number_of_iterations=1)
        else:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        print(f"  Subdivide ({s_method}): {before_v:,} -> {len(mesh.vertices):,} dinh")
        if has_colors:
            v_colors = np.asarray(mesh.vertex_colors).copy() if len(mesh.vertex_colors) == len(mesh.vertices) else None

    v_conf = None
    if ref_points is not None and ref_confidence is not None and len(ref_points) == len(ref_confidence):
        v_conf = interpolate_point_confidence_to_mesh(mesh, ref_points, ref_confidence)

    # --- Giai doan 2: Lam min be mat ---
    smooth_mode = str(r_cfg.get("smooth_mode", "bilateral_normal")).strip().lower()

    if smooth_mode == "bilateral_normal":
        n_iter = int(r_cfg.get("bilateral_normal_iter", 6))
        v_iter = int(r_cfg.get("bilateral_vertex_iter", 20))
        sig_s = float(r_cfg.get("bilateral_sigma_s_scale", 1.5))
        sig_r = float(r_cfg.get("bilateral_sigma_r", 0.38))
        anchor_w = float(r_cfg.get("bilateral_anchor_weight", 0.06))
        mesh = bilateral_normal_smooth(
            mesh,
            normal_iter=n_iter,
            vertex_iter=v_iter,
            sigma_s_scale=sig_s,
            sigma_r=sig_r,
            vertex_confidence=v_conf,
            anchor_weight=anchor_w,
        )
        print(
            f"  Bilateral Normal Smoothing: normal_iter={n_iter}, vertex_iter={v_iter}, "
            f"sigma_r={sig_r}, anchor={anchor_w}"
        )

    elif smooth_mode == "pymeshlab_twostep":
        steps = int(r_cfg.get("twostep_steps", 3))
        norm_thr = float(r_cfg.get("twostep_normal_thr_deg", 45.0))
        mesh = pymeshlab_two_step_smooth(mesh, steps=steps, normal_thr_deg=norm_thr)
        print(f"  PyMeshLab Two-Step Feature Smoothing: steps={steps}, normal_thr={norm_thr} deg")

    else:
        # Legacy Taubin + Simple Polish
        mesh.compute_triangle_normals()
        tri_norms = np.asarray(mesh.triangle_normals)
        is_box_mesh = len(tri_norms) > 0 and float(np.mean(np.max(np.abs(tri_norms), axis=1) > 0.94)) > 0.70

        taubin_iter = int(r_cfg.get("taubin_iter", 45))
        t_lam = float(r_cfg.get("taubin_lambda", 0.5))
        t_mu = float(r_cfg.get("taubin_mu", -0.53))
        polish_iter = int(r_cfg.get("polish_iter", 4))

        if is_box_mesh:
            taubin_iter = min(taubin_iter, 3)
            polish_iter = 0

        if taubin_iter > 0:
            mesh = mesh.filter_smooth_taubin(
                number_of_iterations=taubin_iter,
                lambda_filter=t_lam,
                mu=t_mu,
            )
            print(f"  Taubin smoothing: {taubin_iter} vong (lambda={t_lam}, mu={t_mu})")

        if polish_iter > 0:
            mesh = mesh.filter_smooth_simple(number_of_iterations=polish_iter)
            print(f"  Surface polish: {polish_iter} vong")

    # --- Giai doan 3: Micro-detail baking (tuy chon cho chi tiet khac noi/gan sieu nho) ---
    if r_cfg.get("detail_residual_transfer", False) and ref_points is not None and ref_confidence is not None:
        obs_pts = ref_points[ref_confidence >= 0.85]
        if len(obs_pts) >= 50:
            mesh = transfer_observed_detail_residuals(
                mesh,
                observed_points=obs_pts,
                vertex_confidence=v_conf,
                strength=float(r_cfg.get("detail_residual_strength", 0.25)),
            )
            print(f"  Micro-detail residual transfer: da bao toan chi tiet tu {len(obs_pts):,} diem thuc")

    # --- Giai doan 4: Can truc va tiep xuc mat san Z=0 ---
    if r_cfg.get("align_upright", True):
        verts = np.asarray(mesh.vertices)
        z_min = verts[:, 2].min()
        if abs(z_min) > 1e-4:
            verts[:, 2] -= z_min
            mesh.vertices = o3d.utility.Vector3dVector(verts)

    # --- Giai doan 5: Phap tuyen & Mau sac ---
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    if v_colors is not None and len(v_colors) == len(mesh.vertices):
        mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors)

    return mesh


def run(sample_name: str, cfg: dict) -> None:
    r_cfg = cfg.get("mesh_refine", {})
    out_dir = paths.output_dir(sample_name)
    main_obj = out_dir / f"{sample_name}.obj"
    raw_obj = out_dir / f"{sample_name}_raw.obj"
    refined_obj = out_dir / f"{sample_name}_refined.obj"

    if not main_obj.exists() and not raw_obj.exists():
        raise FileNotFoundError(f"Khong tim thay {main_obj}, vui long chay Buoc 4 truoc.")

    # Luon lay ban goc raw_obj de khong bi chong cheo nhieu lan lam min
    if not raw_obj.exists():
        shutil.copyfile(str(main_obj), str(raw_obj))
        print(f"  da sao luu ban goc -> {raw_obj.name}")

    print(f"[Buoc 5] Doc mesh goc: {raw_obj.name}...")
    mesh = o3d.io.read_triangle_mesh(str(raw_obj))
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)
    print(f"  ban dau: {n_v:,} dinh, {n_t:,} tam giac")

    # Doc confidence map tu Buoc 4 (neu co) de bao ve 100% chi tiet thuc te (w=1.0)
    ref_points, ref_confidence = None, None
    clean_ply = out_dir / "point_cloud_clean.ply"
    conf_npy = out_dir / "point_confidence.npy"
    if clean_ply.exists() and conf_npy.exists():
        try:
            pc_ref = o3d.io.read_point_cloud(str(clean_ply))
            c_arr = np.load(str(conf_npy))
            if len(pc_ref.points) == len(c_arr) and len(c_arr) > 0:
                ref_points = np.asarray(pc_ref.points, dtype=np.float64)
                ref_confidence = c_arr.astype(np.float64)
        except Exception:
            pass

    # Chi chay PCA xoay lai neu Buoc 4 CHUA dong bo chung he truc voi point_cloud_clean.ply
    if r_cfg.get("align_upright", True) and len(mesh.vertices) > 0 and ref_confidence is None:
        from .step4_mesh_export import _align_upright_and_ground
        mesh = _align_upright_and_ground(mesh)
        has_colors = mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices)
        o3d.io.write_triangle_mesh(str(raw_obj), mesh, write_ascii=True, write_vertex_normals=True, write_vertex_colors=has_colors)

    t0 = time.time()
    mesh_refined = refine_mesh(mesh, r_cfg, ref_points=ref_points, ref_confidence=ref_confidence)
    dur = time.time() - t0

    # Xuat ra file refined rieng va dong thoi cap nhat file chinh
    has_colors = mesh_refined.has_vertex_colors() and len(mesh_refined.vertex_colors) == len(mesh_refined.vertices)
    o3d.io.write_triangle_mesh(
        str(refined_obj),
        mesh_refined,
        write_ascii=True,
        write_vertex_normals=True,
        write_vertex_colors=has_colors,
    )
    # Cap nhat luon vao main_obj de nguoi dung mo file mac dinh trong Blender se thay ngay
    shutil.copyfile(str(refined_obj), str(main_obj))

    print(f"[Buoc 5] Hoan tat trong {dur:.2f}s!")
    print(f"  -> {refined_obj.name} ({refined_obj.stat().st_size/1024:.1f} KB)")
    print(f"  -> {main_obj.name} (da dong bo ban tinh chinh)")
    print(f"  Blender: So sanh giua {raw_obj.name} (goc) va {sample_name}.obj (tinh chinh)")
