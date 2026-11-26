"""Buoc 1-2: Phat hien lech truc (PCA/OBB skew), loc vo trung lap (ghost shells) & can chinh tong quat.

Cac ham duoc thiet ke object-agnostic (khong phu thuoc loai vat the):
  - compute_pose_skew(): do goc lech giua cac truc chinh OBB/PCA voi he truc the gioi (X, Y, Z),
    ty so the tich OBB/AABB va ty so tri rieng (de tranh bao nham tren vat doi xung cau/lap phuong).
  - detect_and_prune_ghost_shells(): phat hien thanh phan con bi nhan doi/lech pha (offset duplicate
    shells) co cung huong phap tuyen (bao ve thanh trong cua vat rong co phap tuyen nguoc chieu).
  - align_pca_canonical(): can truc PCA co bao toan det(R)=+1 va giu huong gan nhat voi tu the goc.
  - align_to_reference_multistart(): thu cac huong doi xung chinh + Point-to-Point ICP de tranh
    lat nguoc 180 do tren vat the bat doi xung.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from nckh_2.backends.geometry import (
    _spatial_connected_components,
    remove_triangles_preserving_attributes,
)


@dataclasses.dataclass
class PoseSkewReport:
    """Bao cao do lech truc & tinh doi xung hinh hoc cua mesh."""

    max_axis_skew_deg: float
    axis_skew_deg: tuple[float, float, float]
    eigen_ratios: tuple[float, float]          # (lambda0/lambda1, lambda1/lambda2)
    obb_aabb_volume_ratio: float               # nho (<0.65) thuong bao hieu vat dang bi xoay xien
    is_skewed: bool
    is_isotropic_ambiguous: bool               # True neu vat gan nhu hinh cau/lap phuong

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compute_pose_skew(
    mesh: o3d.geometry.TriangleMesh,
    skew_warn_deg: float = 5.0,
    eigen_ratio_min: float = 1.15,
) -> PoseSkewReport:
    """Do goc lech cua 3 truc chinh PCA so voi cac truc chuan (X, Y, Z) mot cach khach quan."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) < 4:
        return PoseSkewReport(0.0, (0.0, 0.0, 0.0), (1.0, 1.0), 1.0, False, True)

    centered = verts - verts.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sap xep giam dan: lambda0 >= lambda1 >= lambda2
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-12)
    eigvecs = eigvecs[:, order]

    r01 = float(eigvals[0] / eigvals[1])
    r12 = float(eigvals[1] / eigvals[2])
    ambiguous = (r01 < eigen_ratio_min) and (r12 < eigen_ratio_min)

    # Goc lech toi thieu cua moi truc chinh voi truc the gioi gan nhat
    skews: list[float] = []
    for i in range(3):
        v = eigvecs[:, i]
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        max_cos = float(np.clip(np.max(np.abs(v)), 0.0, 1.0))
        skews.append(float(np.degrees(np.arccos(max_cos))))

    # Chi xet lech tren cac truc co tri rieng phan biet ro rang
    meaningful_skews: list[float] = []
    if r01 >= eigen_ratio_min:
        meaningful_skews.append(skews[0])
    if r12 >= eigen_ratio_min:
        meaningful_skews.append(skews[2])
    if r01 >= eigen_ratio_min and r12 >= eigen_ratio_min:
        meaningful_skews.append(skews[1])
    max_skew = max(meaningful_skews) if meaningful_skews else 0.0

    # Ty so the tich OBB / AABB
    aabb_ext = np.maximum(verts.max(axis=0) - verts.min(axis=0), 1e-9)
    aabb_vol = float(np.prod(aabb_ext))
    proj = centered @ eigvecs
    obb_ext = np.maximum(proj.max(axis=0) - proj.min(axis=0), 1e-9)
    obb_vol = float(np.prod(obb_ext))
    vol_ratio = float(np.clip(obb_vol / max(aabb_vol, 1e-18), 0.0, 1.0))

    return PoseSkewReport(
        max_axis_skew_deg=round(max_skew, 2),
        axis_skew_deg=tuple(round(x, 2) for x in skews),
        eigen_ratios=(round(r01, 3), round(r12, 3)),
        obb_aabb_volume_ratio=round(vol_ratio, 4),
        is_skewed=(not ambiguous) and (max_skew > float(skew_warn_deg)),
        is_isotropic_ambiguous=ambiguous,
    )


def detect_and_prune_ghost_shells(
    mesh: o3d.geometry.TriangleMesh,
    overlap_dist_frac: float = 0.015,
    ghost_overlap_thr: float = 0.65,
    normal_align_cos: float = 0.5,
    prune: bool = True,
) -> tuple[o3d.geometry.TriangleMesh, dict[str, Any]]:
    """Phat hien va (tuy chon) loai bo cac vo trung lap / lech pha (offset ghost shells).

    An toan voi vat rong (hollow objects): chi coi mot thanh phan nho hon la ghost shell khi
    it nhat ``ghost_overlap_thr`` dinh cua no nam sat mot thanh phan lon hon (<= overlap_dist_frac * diag)
    VA phap tuyen cung chieu (dot > normal_align_cos). Thanh trong cua vat rong co phap tuyen nguoc chieu
    nen se duoc giu nguyen.
    """
    out = o3d.geometry.TriangleMesh(mesh)
    verts = np.asarray(out.vertices, dtype=np.float64)
    tris = np.asarray(out.triangles, dtype=np.int32)
    stats: dict[str, Any] = {
        "components_checked": 0,
        "ghost_shells_detected": 0,
        "ghost_shells_removed": 0,
    }
    if len(verts) == 0 or len(tris) == 0:
        return out, stats

    labels, tri_counts, comp_areas = _spatial_connected_components(verts, tris)
    n_comp = len(tri_counts)
    stats["components_checked"] = n_comp
    if n_comp <= 1:
        return out, stats

    out.compute_vertex_normals()
    v_norms = np.asarray(out.vertex_normals, dtype=np.float64)
    diag = max(float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))), 1e-6)
    dist_thr = float(overlap_dist_frac) * diag

    # Sap xep cac thanh phan tu lon nhat den nho nhat theo dien tich be mat
    comp_order = np.argsort(comp_areas)[::-1]
    accepted_comps: list[tuple[int, cKDTree, np.ndarray]] = []
    ghost_comp_ids: list[int] = []

    for cid in comp_order:
        cid_int = int(cid)
        c_tris = tris[labels == cid_int]
        c_v_idx = np.unique(c_tris.ravel())
        c_pts = verts[c_v_idx]
        c_nrm = v_norms[c_v_idx]

        # Lay mau toi da 4000 dinh de kiem tra nhanh tren luoi lon
        if len(c_pts) > 4000:
            sub = np.linspace(0, len(c_pts) - 1, 4000, dtype=np.int64)
            q_pts, q_nrm = c_pts[sub], c_nrm[sub]
        else:
            q_pts, q_nrm = c_pts, c_nrm

        is_ghost = False
        for _acc_id, acc_tree, acc_nrm in accepted_comps:
            dists, idxs = acc_tree.query(q_pts, k=1)
            close = dists <= dist_thr
            if float(np.mean(close)) < float(ghost_overlap_thr):
                continue
            # Kiem tra cung chieu phap tuyen (bao ve thanh trong cua vat the rong)
            cos_sim = np.sum(q_nrm[close] * acc_nrm[idxs[close]], axis=1)
            if float(np.mean(cos_sim > float(normal_align_cos))) >= 0.60:
                is_ghost = True
                break

        if is_ghost:
            ghost_comp_ids.append(cid_int)
        else:
            accepted_comps.append((cid_int, cKDTree(c_pts), c_nrm))

    stats["ghost_shells_detected"] = len(ghost_comp_ids)
    if prune and ghost_comp_ids:
        remove_mask = np.isin(labels, ghost_comp_ids)
        out = remove_triangles_preserving_attributes(out, remove_mask)
        out.remove_unreferenced_vertices()
        out.compute_triangle_normals()
        out.compute_vertex_normals()
        stats["ghost_shells_removed"] = len(ghost_comp_ids)

    return out, stats


def align_pca_canonical(
    mesh: o3d.geometry.TriangleMesh,
    normalize_unit_scale: bool = False,
    eigen_ratio_min: float = 1.15,
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Can truc chinh OBB/PCA ve he (X, Y, Z) voi det(R)=+1 va goc xoay toi thieu.

    Khong lam lat guong (handedness preserved) va giu huong gan nhat voi tu the ban dau
    de han che lat 180 do tren vat the bat doi xung.
    """
    out = o3d.geometry.TriangleMesh(mesh)
    verts = np.asarray(out.vertices, dtype=np.float64)
    if len(verts) < 4:
        return out, np.eye(4, dtype=np.float64)

    centroid = verts.mean(axis=0)
    centered = verts - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-12)
    eigvecs = eigvecs[:, order]

    # Neu vat the gan nhu dang huong (hinh cau/lap phuong), khong xoay ngau nhien
    if (eigvals[0] / eigvals[1] < eigen_ratio_min) and (eigvals[1] / eigvals[2] < eigen_ratio_min):
        R = np.eye(3, dtype=np.float64)
    else:
        # Gan moi truc PCA vao truc the gioi gan nhat (tranh xoay 90 do khong can thiet)
        R_cols = np.zeros((3, 3), dtype=np.float64)
        used_axes: set[int] = set()
        for i in range(3):
            v = eigvecs[:, i]
            scores = np.abs(v)
            for ax in np.argsort(scores)[::-1]:
                ax_int = int(ax)
                if ax_int not in used_axes:
                    used_axes.add(ax_int)
                    # Giu cung chieu voi truc the gioi goc
                    R_cols[:, ax_int] = v if v[ax_int] >= 0 else -v
                    break
        R = R_cols.T  # p_new = (p - c) @ R.T
        if np.linalg.det(R) < 0:
            R[2, :] *= -1.0

    scale = 1.0
    new_verts = centered @ R.T
    if normalize_unit_scale:
        diag = max(float(np.linalg.norm(new_verts.max(axis=0) - new_verts.min(axis=0))), 1e-6)
        scale = 1.0 / diag
        new_verts = new_verts * scale
    else:
        # Giu nguyen tam the gioi goc khi khong normalize
        new_verts = new_verts + centroid

    out.vertices = o3d.utility.Vector3dVector(new_verts)
    out.compute_triangle_normals()
    out.compute_vertex_normals()

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = scale * R
    T[:3, 3] = (0.0 if normalize_unit_scale else centroid) - (scale * R) @ centroid
    return out, T


def align_to_reference_multistart(
    mesh: o3d.geometry.TriangleMesh,
    ref_points: np.ndarray,
    match_scale: bool = False,
    icp_max_corr_frac: float = 0.05,
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray, float]:
    """Can chinh mesh vao tap diem tham chieu bang multi-start PCA (4 huong quay hop le) + ICP.

    Chon ung vien co Chamfer thap nhat de tranh loi lat nguoc 180 do tren vat bat doi xung.
    Tra ve (aligned_mesh, transform_4x4, scale_factor).
    """
    out = o3d.geometry.TriangleMesh(mesh)
    verts = np.asarray(out.vertices, dtype=np.float64)
    if len(verts) < 4 or len(ref_points) < 4:
        return out, np.eye(4, dtype=np.float64), 1.0

    src_cent = verts.mean(axis=0)
    ref_cent = ref_points.mean(axis=0)

    scale = 1.0
    if match_scale:
        src_diag = max(float(np.linalg.norm(np.percentile(verts, 95, axis=0) - np.percentile(verts, 5, axis=0))), 1e-6)
        ref_diag = max(float(np.linalg.norm(np.percentile(ref_points, 95, axis=0) - np.percentile(ref_points, 5, axis=0))), 1e-6)
        scale = ref_diag / src_diag

    # 4 phep xoay doi xung 180 do quanh cac truc chinh (tat ca deu co det=+1)
    sign_flips = [
        np.diag([1.0, 1.0, 1.0]),
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    ]

    # Lay mau toi da 10k diem de chon huong khoi tao nhanh
    sub_idx = np.linspace(0, len(verts) - 1, min(len(verts), 10000), dtype=np.int64)
    src_sub = (verts[sub_idx] - src_cent) * scale
    ref_tree = cKDTree(ref_points)

    best_R = np.eye(3, dtype=np.float64)
    best_err = float("inf")
    for F in sign_flips:
        cand_pts = src_sub @ F.T + ref_cent
        d, _ = ref_tree.query(cand_pts, k=1)
        err = float(np.mean(d))
        if err < best_err:
            best_err = err
            best_R = F

    # Tinh chinh bang Point-to-Point ICP
    src_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_sub @ best_R.T + ref_cent))
    ref_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref_points))
    ref_diag_full = max(float(np.linalg.norm(ref_points.max(axis=0) - ref_points.min(axis=0))), 1e-6)

    reg = o3d.pipelines.registration.registration_icp(
        src_pc,
        ref_pc,
        max_correspondence_distance=float(icp_max_corr_frac) * ref_diag_full,
        init=np.eye(4, dtype=np.float64),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    T_icp = np.asarray(reg.transformation, dtype=np.float64)

    # Tong hop phep bien doi cuoi: p' = T_icp * (best_R * scale * (p - src_cent) + ref_cent)
    T_init = np.eye(4, dtype=np.float64)
    T_init[:3, :3] = best_R * scale
    T_init[:3, 3] = ref_cent - (best_R * scale) @ src_cent
    T_total = T_icp @ T_init

    out.transform(T_total)
    out.compute_triangle_normals()
    out.compute_vertex_normals()
    return out, T_total, scale
