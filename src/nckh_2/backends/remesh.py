"""Stage 3: Controlled isotropic explicit remeshing & non-destructive reference alignment.

Equalizes triangle density across all 6 sides and eliminates sliver/jagged triangles using
PyMeshLab's isotropic explicit remeshing with crease-angle protection, while preserving
vertex colors (via cKDTree transfer) and monitoring geometric drift against the input mesh.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@dataclasses.dataclass
class RemeshStats:
    applied: bool
    reason: str
    target_edge_len: float
    verts_before: int
    verts_after: int
    tris_before: int
    tris_after: int
    p95_drift: float
    p95_drift_frac_diag: float


def isotropic_remesh(
    mesh: o3d.geometry.TriangleMesh,
    target_edge_frac: float = 0.004,
    iterations: int = 3,
    feature_angle_deg: float = 35.0,
    adaptive: bool = True,
    preserve_uvs: bool = True,
) -> tuple[o3d.geometry.TriangleMesh, RemeshStats]:
    """Apply feature-preserving isotropic explicit remeshing via PyMeshLab.

    Automatically skips remeshing if ``preserve_uvs=True`` and the mesh carries valid UVs
    (since remeshing rebuilds topology). Transfers vertex colors from the source mesh via k-NN.
    """
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)
    if n_v == 0 or n_t == 0:
        return o3d.geometry.TriangleMesh(mesh), RemeshStats(False, "empty_mesh", 0.0, n_v, n_v, n_t, n_t, 0.0, 0.0)

    has_uvs = bool(mesh.has_triangle_uvs() and len(mesh.triangle_uvs) == 3 * n_t)
    if preserve_uvs and has_uvs:
        return o3d.geometry.TriangleMesh(mesh), RemeshStats(
            False, "skipped_to_preserve_uvs", 0.0, n_v, n_v, n_t, n_t, 0.0, 0.0
        )

    try:
        import pymeshlab
    except ImportError:
        return o3d.geometry.TriangleMesh(mesh), RemeshStats(
            False, "pymeshlab_not_installed", 0.0, n_v, n_v, n_t, n_t, 0.0, 0.0
        )

    verts_in = np.asarray(mesh.vertices, dtype=np.float64)
    tris_in = np.asarray(mesh.triangles, dtype=np.int32)
    diag = max(float(np.linalg.norm(verts_in.max(axis=0) - verts_in.min(axis=0))), 1e-6)
    target_len = float(target_edge_frac) * diag

    has_colors = bool(mesh.has_vertex_colors() and len(mesh.vertex_colors) == n_v)
    cols_in = np.asarray(mesh.vertex_colors, dtype=np.float64) if has_colors else None

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=verts_in, face_matrix=tris_in))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=int(iterations),
        adaptive=bool(adaptive),
        targetlen=pymeshlab.PureValue(target_len),
        featuredeg=float(feature_angle_deg),
        checksurfdist=True,
        maxsurfdist=pymeshlab.PureValue(max(0.5 * target_len, 1e-6)),
        splitflag=True,
        collapseflag=True,
        swapflag=True,
        smoothflag=True,
        reprojectflag=True,
    )
    res = ms.current_mesh()
    verts_out = np.asarray(res.vertex_matrix(), dtype=np.float64)
    tris_out = np.asarray(res.face_matrix(), dtype=np.int32)

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(verts_out)
    out.triangles = o3d.utility.Vector3iVector(tris_out)

    # Measure P95 geometric drift against input vertices & transfer vertex colors
    tree_in = cKDTree(verts_in)
    dists, idxs = tree_in.query(verts_out, k=1)
    p95_drift = float(np.percentile(dists, 95)) if len(dists) > 0 else 0.0

    if cols_in is not None and len(verts_out) > 0:
        out.vertex_colors = o3d.utility.Vector3dVector(cols_in[idxs])

    out.compute_triangle_normals()
    out.compute_vertex_normals()

    stats = RemeshStats(
        applied=True,
        reason="ok",
        target_edge_len=round(target_len, 6),
        verts_before=n_v,
        verts_after=len(verts_out),
        tris_before=n_t,
        tris_after=len(tris_out),
        p95_drift=round(p95_drift, 6),
        p95_drift_frac_diag=round(p95_drift / diag, 6),
    )
    return out, stats


def align_mesh_to_reference_copy(
    source_mesh: o3d.geometry.TriangleMesh,
    ref_points: np.ndarray,
    n_samples: int = 20000,
    max_corr_frac: float = 0.05,
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Return an ICP-aligned COPY of ``source_mesh`` for non-destructive comparison.

    Does NOT mutate ``source_mesh``. Uses centroid translation + Point-to-Point ICP.
    """
    aligned = o3d.geometry.TriangleMesh(source_mesh)
    if len(aligned.vertices) == 0 or len(ref_points) == 0:
        return aligned, np.eye(4, dtype=np.float64)

    src_pc = aligned.sample_points_uniformly(number_of_points=min(n_samples, max(len(aligned.vertices), 1000)))
    ref_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref_points))

    src_cent = np.asarray(src_pc.points).mean(axis=0)
    ref_cent = ref_points.mean(axis=0)
    init_T = np.eye(4, dtype=np.float64)
    init_T[:3, 3] = ref_cent - src_cent

    diag = max(float(np.linalg.norm(ref_points.max(axis=0) - ref_points.min(axis=0))), 1e-6)
    reg = o3d.pipelines.registration.registration_icp(
        src_pc,
        ref_pc,
        max_correspondence_distance=max_corr_frac * diag,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    T = np.asarray(reg.transformation, dtype=np.float64)
    aligned.transform(T)
    aligned.compute_triangle_normals()
    aligned.compute_vertex_normals()
    return aligned, T
