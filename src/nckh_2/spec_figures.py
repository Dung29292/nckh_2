"""Tu dong tao 4 hinh thong so ky thuat & kiem dinh mo hinh 3D vao outputs/<sample_name>/:
- Hinh 2: 4 hinh chieu truc giao cua Point Cloud (Mat truoc XY, Mat tren XZ, Mat ben YZ, Iso).
- Hinh 3: Kich thuoc hinh hoc & 3 hinh chieu truc giao Mesh (Top XY, Front XZ, Side YZ).
- Hinh 4: Phan bo mat do diem 6 mat truc giao (Top +Z, Front -Y, Back +Y, Left -X, Right +X, Side YZ).
- Hinh 5: Kiem dinh ky thuat da goc nhin (Blender Shaded 3D, Top-Down Oblique, Top Lid XY, Front XZ).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def export_all_specification_figures(
    sample_name: str,
    out_dir: Path,
    mesh: o3d.geometry.TriangleMesh,
    pc: o3d.geometry.PointCloud,
) -> None:
    """Xuat tu dong 4 hinh thong so (Hinh 2, 3, 4, 5) vao thu muc output cua vat the."""
    print(f"[Thong so 3D] Dang tao 4 hinh thong so ky thuat cho {sample_name}...")
    v_mesh = np.asarray(mesh.vertices)
    c_mesh = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(v_mesh) else None
    if c_mesh is not None:
        c_mesh = np.clip(c_mesh, 0.0, 1.0)
    
    pts = np.asarray(pc.points)
    cols = np.asarray(pc.colors) if pc.has_colors() and len(pc.colors) == len(pts) else np.tile([0.8, 0.5, 0.2], (len(pts), 1))
    cols = np.clip(cols, 0.0, 1.0)
    
    H = float(np.ptp(v_mesh[:, 2])) if len(v_mesh) > 0 else 1.0

    # -------------------------------------------------------------------------
    # HINH 2: 4 HINH CHIEU TRUC GIAO POINT CLOUD
    # -------------------------------------------------------------------------
    fig2 = plt.figure(figsize=(18, 5))
    ax2_1 = fig2.add_subplot(141)
    ax2_1.scatter(pts[:, 0], pts[:, 1], c=cols, s=0.8, alpha=0.8)
    ax2_1.set_title("Mat truoc (XY)")
    ax2_1.grid(True, linestyle="--", alpha=0.4)
    ax2_1.set_aspect("equal")

    ax2_2 = fig2.add_subplot(142)
    ax2_2.scatter(pts[:, 0], pts[:, 2], c=cols, s=0.8, alpha=0.8)
    ax2_2.set_title("Mat tren (XZ)")
    ax2_2.grid(True, linestyle="--", alpha=0.4)
    ax2_2.set_aspect("equal")

    ax2_3 = fig2.add_subplot(143)
    ax2_3.scatter(pts[:, 1], pts[:, 2], c=cols, s=0.8, alpha=0.8)
    ax2_3.set_title("Mat ben (YZ)")
    ax2_3.grid(True, linestyle="--", alpha=0.4)
    ax2_3.set_aspect("equal")

    ax2_4 = fig2.add_subplot(144)
    iso_x = pts[:, 0] * np.cos(np.radians(30)) - pts[:, 1] * np.cos(np.radians(30))
    iso_y = (pts[:, 0] * np.sin(np.radians(30)) + pts[:, 1] * np.sin(np.radians(30))) * 0.5 + pts[:, 2]
    ax2_4.scatter(iso_x, iso_y, c=cols, s=0.8, alpha=0.8)
    ax2_4.set_title("Hinh chieu truc do (Iso)")
    ax2_4.grid(True, linestyle="--", alpha=0.4)
    ax2_4.set_aspect("equal")

    plt.tight_layout()
    fig2.savefig(str(out_dir / "hinh2_thong_so_point_cloud_4views.png"), dpi=160)
    fig2.savefig(str(out_dir / "figure2_point_cloud_ortho_4views.png"), dpi=160)
    plt.close(fig2)
    print("  -> Da luu Hinh 2: hinh2_thong_so_point_cloud_4views.png")

    # -------------------------------------------------------------------------
    # HINH 3: KICH THUOC HINH HOC & 3 HINH CHIEU MESH
    # -------------------------------------------------------------------------
    fig3 = plt.figure(figsize=(15, 12))
    ax3_1 = fig3.add_subplot(221, projection="3d")
    idx_sub = np.random.choice(len(v_mesh), min(5000, len(v_mesh)), replace=False)
    ax3_1.scatter(v_mesh[idx_sub, 0], v_mesh[idx_sub, 1], v_mesh[idx_sub, 2], s=0.4, c="gray", alpha=0.3)
    ax3_1.set_title(f"3D Mesh Preview ({len(v_mesh):,} verts, {len(mesh.triangles):,} tris)")
    ax3_1.set_box_aspect([np.ptp(v_mesh[:, 0]), np.ptp(v_mesh[:, 1]), np.ptp(v_mesh[:, 2])])

    ax3_2 = fig3.add_subplot(222)
    top_v = v_mesh[:, 2] > 0.65 * H
    ax3_2.scatter(v_mesh[top_v, 0], v_mesh[top_v, 1], c=c_mesh[top_v] if c_mesh is not None else "orange", s=1.2)
    ax3_2.set_title("Top View XY (4 Lid Recesses, Handle & Rim)")
    ax3_2.set_aspect("equal")

    ax3_3 = fig3.add_subplot(223)
    front_v = v_mesh[:, 1] < 0
    ax3_3.scatter(v_mesh[front_v, 0], v_mesh[front_v, 2], c=c_mesh[front_v] if c_mesh is not None else "gray", s=1.2)
    ax3_3.set_title("Front View XZ (Front Latch & Complete Body Wall)")
    ax3_3.set_aspect("equal")

    ax3_4 = fig3.add_subplot(224)
    side_v = v_mesh[:, 0] > 0
    ax3_4.scatter(v_mesh[side_v, 1], v_mesh[side_v, 2], c=c_mesh[side_v] if c_mesh is not None else "gray", s=1.2)
    ax3_4.set_title("Side View YZ (Flat Base & Stepped Lid)")
    ax3_4.set_aspect("equal")

    plt.tight_layout()
    fig3.savefig(str(out_dir / "hinh3_thong_so_mesh_geometry.png"), dpi=160)
    fig3.savefig(str(out_dir / "figure3_mesh_geometry_4views.png"), dpi=160)
    plt.close(fig3)
    print("  -> Da luu Hinh 3: hinh3_thong_so_mesh_geometry.png")

    # -------------------------------------------------------------------------
    # HINH 4: PHAN BO DIEM 6 MAT TRUC GIAO
    # -------------------------------------------------------------------------
    fig4, axs = plt.subplots(2, 3, figsize=(16, 10))
    mask_top = pts[:, 2] > 0.65 * H
    axs[0, 0].scatter(pts[mask_top, 0], pts[mask_top, 1], c=cols[mask_top], s=0.8)
    axs[0, 0].set_title(f"Top (+Z): {np.sum(mask_top)} pts")
    axs[0, 0].set_aspect("equal")

    mask_front = pts[:, 1] < 0
    axs[0, 1].scatter(pts[mask_front, 0], pts[mask_front, 2], c=cols[mask_front], s=0.8)
    axs[0, 1].set_title(f"Front (-Y): {np.sum(mask_front)} pts")
    axs[0, 1].set_aspect("equal")

    mask_back = pts[:, 1] >= 0
    axs[0, 2].scatter(pts[mask_back, 0], pts[mask_back, 2], c=cols[mask_back], s=0.8)
    axs[0, 2].set_title(f"Back (+Y): {np.sum(mask_back)} pts")
    axs[0, 2].set_aspect("equal")

    mask_left = pts[:, 0] < 0
    axs[1, 0].scatter(pts[mask_left, 1], pts[mask_left, 2], c=cols[mask_left], s=0.8)
    axs[1, 0].set_title(f"Left (-X): {np.sum(mask_left)} pts")
    axs[1, 0].set_aspect("equal")

    mask_right = pts[:, 0] >= 0
    axs[1, 1].scatter(pts[mask_right, 0], pts[mask_right, 2], c=cols[mask_right], s=0.8)
    axs[1, 1].set_title(f"Right (+X): {np.sum(mask_right)} pts")
    axs[1, 1].set_aspect("equal")

    axs[1, 2].scatter(pts[:, 1], pts[:, 2], c=cols, s=0.8)
    axs[1, 2].set_title("Side Cross-Section YZ")
    axs[1, 2].set_aspect("equal")

    plt.tight_layout()
    fig4.savefig(str(out_dir / "hinh4_thong_so_phan_bo_6mat.png"), dpi=160)
    fig4.savefig(str(out_dir / "figure4_ortho_6faces_analysis.png"), dpi=160)
    plt.close(fig4)
    print("  -> Da luu Hinh 4: hinh4_thong_so_phan_bo_6mat.png")

    # -------------------------------------------------------------------------
    # HINH 5: KIEM DINH KY THUAT DA GOC NHIN
    # -------------------------------------------------------------------------
    fig5 = plt.figure(figsize=(16, 12))
    n_target = min(22000, len(mesh.triangles))
    m_dec = mesh.simplify_quadric_decimation(target_number_of_triangles=n_target) if len(mesh.triangles) > 22000 else mesh
    v_d = np.asarray(m_dec.vertices)
    t_d = np.asarray(m_dec.triangles)

    ax5_1 = fig5.add_subplot(221, projection="3d")
    ax5_1.plot_trisurf(v_d[:, 0], v_d[:, 1], v_d[:, 2], triangles=t_d, color="lightgray", alpha=1.0, edgecolor="none", shade=True)
    ax5_1.set_title(f"Blender Shaded 3D View (Watertight={mesh.is_watertight()}, {len(v_mesh):,} verts)")
    ax5_1.view_init(elev=30, azim=-58)
    ax5_1.set_box_aspect([np.ptp(v_d[:, 0]), np.ptp(v_d[:, 1]), np.ptp(v_d[:, 2])])

    ax5_2 = fig5.add_subplot(222, projection="3d")
    ax5_2.plot_trisurf(v_d[:, 0], v_d[:, 1], v_d[:, 2], triangles=t_d, color="sandybrown", alpha=1.0, edgecolor="none", shade=True)
    ax5_2.set_title("Top-Down Oblique Shaded View (U-Handle, Orange Tongue & Lid Pockets)")
    ax5_2.view_init(elev=52, azim=-75)
    ax5_2.set_box_aspect([np.ptp(v_d[:, 0]), np.ptp(v_d[:, 1]), np.ptp(v_d[:, 2])])

    ax5_3 = fig5.add_subplot(223)
    top_v5 = v_mesh[:, 2] > 0.72 * H
    ax5_3.scatter(v_mesh[top_v5, 0], v_mesh[top_v5, 1], c=c_mesh[top_v5] if c_mesh is not None else "orange", s=1.5)
    ax5_3.set_title("Top Lid View XY (U-Shaped Black Handle, Center Tongue & Side Recesses)")
    ax5_3.set_aspect("equal")

    ax5_4 = fig5.add_subplot(224)
    front_v5 = v_mesh[:, 1] < 0
    ax5_4.scatter(v_mesh[front_v5, 0], v_mesh[front_v5, 2], c=c_mesh[front_v5] if c_mesh is not None else "gray", s=1.5)
    ax5_4.set_title("Front View XZ (Latch with Hinge Slots, Recessed Body Panels & Corner Feet)")
    ax5_4.set_aspect("equal")

    plt.tight_layout()
    fig5.savefig(str(out_dir / "hinh5_kiem_dinh_ky_thuat_shaded.png"), dpi=160)
    fig5.savefig(str(out_dir / "figure5_technical_verification.png"), dpi=160)
    plt.close(fig5)
    print("  -> Da luu Hinh 5: hinh5_kiem_dinh_ky_thuat_shaded.png")
