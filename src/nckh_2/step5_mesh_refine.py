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


def refine_mesh(mesh: o3d.geometry.TriangleMesh, r_cfg: dict) -> o3d.geometry.TriangleMesh:
    """Pipeline lam min da tang:

    Giai doan 1 (Tuy chon): Subdivide tang do phan giai luoi tam giac
    Giai doan 2: Taubin anti-shrinkage smoothing (xoa bo gon song khong lam teo the tich)
    Giai doan 3: Gentle Polish (lam min be mat sieu phang, bong dep)
    Giai doan 4: Tinh toan lai phap tuyen & bao toan mau sac
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

    # --- Giai doan 2: Taubin Smoothing (khong co cum the tich) ---
    taubin_iter = int(r_cfg.get("taubin_iter", 45))
    t_lam = float(r_cfg.get("taubin_lambda", 0.5))
    t_mu = float(r_cfg.get("taubin_mu", -0.53))

    if taubin_iter > 0:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=taubin_iter,
            lambda_filter=t_lam,
            mu=t_mu,
        )
        print(f"  Taubin smoothing: {taubin_iter} vong (lambda={t_lam}, mu={t_mu})")

    # --- Giai doan 3: Polish (lam muot be mat) ---
    polish_iter = int(r_cfg.get("polish_iter", 4))
    if polish_iter > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=polish_iter)
        print(f"  Surface polish: {polish_iter} vong")

    # --- Giai doan 4: Phap tuyen & Mau sac ---
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

    t0 = time.time()
    mesh_refined = refine_mesh(mesh, r_cfg)
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
