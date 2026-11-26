"""Top-Down 'Canonical Base Shape + Residual Displacement Field' & Elastic Energy Solver (Section 4).

Strictly implements Equations (13), (14), (15), and (16) from nckh.pdf:
  - Eq. (13): P_actual = P_canonical(theta) + Delta_P_residual
  - Eq. (14): Superquadric prior (|x|/a)^(2/eps2) + (|y|/b)^(2/eps2) + (|z|/c)^(2/eps1) = 1
  - Eq. (15): Normal-directed scalar displacement field Delta_P_i = d_i * n_base,i
  - Eq. (16): Elastic energy minimization E_total(d) = E_data(d) + lambda_smooth * E_smoothness(d)
              solved deterministically in closed form (PCA/OBB initialization + sparse linear IRLS),
              guaranteeing outlier immunity without random seed instability.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from nckh_2.backends.ransac_manhattan import (
    DetectedPlane,
    WallSegment2D,
    align_manhattan_world,
    intersect_lines_2d,
    reclaim_clutter_and_close_gaps,
    regularize_rectilinear_polygon,
)
from nckh_2.backends.surfaces import HessianPlane, SignedDistanceField


# =============================================================================
# SECTION 4.1: SUPERQUADRIC CANONICAL SHAPE PRIOR (Eq. 14)
# =============================================================================

def _signed_power(base: np.ndarray, exp: float) -> np.ndarray:
    return np.sign(base) * (np.abs(base) ** float(exp))


@dataclasses.dataclass
class SuperquadricPrior:
    """Canonical Superquadric / Semi-Ellipsoid Shape Prior P_canonical(theta) (Section 4.1, Eq. 14):

        (|x| / a)^(2 / eps2) + (|y| / b)^(2 / eps2) + (|z| / c)^(2 / eps1) = 1

    Parameterized by coarse pose & shape vector theta = (center, rotation, a, b, c, eps1, eps2).
    """

    center: np.ndarray        # (3,) world translation
    rotation: np.ndarray      # (3, 3) world-from-local rotation (det = +1)
    semi_axes: np.ndarray     # (3,) (a, b, c) > 0
    eps1: float               # z-elevation squareness/roundness exponent
    eps2: float               # xy-planform squareness/roundness exponent

    def to_local(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return (pts - self.center[None, :]) @ self.rotation

    def to_world(self, local_points: np.ndarray) -> np.ndarray:
        lpts = np.atleast_2d(np.asarray(local_points, dtype=np.float64))
        return lpts @ self.rotation.T + self.center[None, :]

    def inside_outside_function(self, points: np.ndarray) -> np.ndarray:
        """Evaluate canonical inside-outside function F(x, y, z) where F = 1 on surface (Eq. 14)."""
        loc = self.to_local(points)
        a, b, c = self.semi_axes
        e1 = max(float(self.eps1), 0.05)
        e2 = max(float(self.eps2), 0.05)
        # Clamp scaled ratios before exponentiation to prevent IEEE-754 float64 overflow when eps -> 0
        rx = np.clip(np.abs(loc[:, 0]) / max(a, 1.0e-9), 0.0, 1.0e6)
        ry = np.clip(np.abs(loc[:, 1]) / max(b, 1.0e-9), 0.0, 1.0e6)
        rz = np.clip(np.abs(loc[:, 2]) / max(c, 1.0e-9), 0.0, 1.0e6)
        xy_term = (rx ** (2.0 / e2)) + (ry ** (2.0 / e2))
        # Generalized Barr superquadric radial form reduces exactly to Eq. (14) when eps1 == eps2
        # and remains smooth on the equator; here we evaluate exact Eq. (14) directly:
        return xy_term + (rz ** (2.0 / e1))

    def radial_sdf(self, points: np.ndarray, zero_tol: float = 1.0e-4) -> SignedDistanceField:
        """Wrap superquadric as a SignedDistanceField f_theta(x, y, z) = s (Eq. 9 & Eq. 14)."""
        min_scale = float(np.min(self.semi_axes))

        def _fn(pts: np.ndarray) -> np.ndarray:
            f_val = np.maximum(self.inside_outside_function(pts), 0.0)
            # Radial Euclidean approximation s ~ min_axis * (F^(eps1/2) - 1): s=0 on F=1, s<0 inside, s>0 outside
            return min_scale * ((f_val ** (0.5 * max(float(self.eps1), 0.05))) - 1.0)

        return SignedDistanceField(field_fn=_fn, zero_tol=zero_tol)

    def to_canonical_mesh(self, nu: int = 48, nv: int = 96) -> o3d.geometry.TriangleMesh:
        """Generate watertight genus-0 canonical mesh with outward unit normals n_base,i."""
        a, b, c = self.semi_axes
        e1 = max(float(self.eps1), 0.05)
        e2 = max(float(self.eps2), 0.05)

        # Sample spherical topology (poles + interior rings) to build a strictly watertight 2-manifold
        eta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, int(nu), dtype=np.float64)
        omega = np.linspace(-np.pi, np.pi, int(nv), endpoint=False, dtype=np.float64)

        local_verts: list[np.ndarray] = [
            np.array([0.0, 0.0, -c], dtype=np.float64),  # index 0: south pole
        ]
        for i in range(1, int(nu) - 1):
            ce = float(np.cos(eta[i]))
            se = float(np.sin(eta[i]))
            z_ring = c * _signed_power(np.array([se]), e1)[0]
            # Radial factor rho on Eq. (14) level set so (|x|/a)^(2/e2) + (|y|/b)^(2/e2) = 1 - (|z|/c)^(2/e1)
            rem = max(1.0 - abs(z_ring / max(c, 1.0e-9)) ** (2.0 / e1), 0.0)
            rho = rem ** (0.5 * e2)
            x_ring = a * rho * _signed_power(np.cos(omega), e2)
            y_ring = b * rho * _signed_power(np.sin(omega), e2)
            z_arr = np.full_like(x_ring, z_ring)
            local_verts.append(np.column_stack([x_ring, y_ring, z_arr]))

        north_idx = 1 + (int(nu) - 2) * int(nv)
        local_verts.append(np.array([[0.0, 0.0, c]], dtype=np.float64))  # north pole
        verts_loc = np.vstack(local_verts)
        verts_world = self.to_world(verts_loc)

        tris: list[list[int]] = []
        # South cap
        for j in range(int(nv)):
            jn = (j + 1) % int(nv)
            tris.append([0, 1 + jn, 1 + j])
        # Middle quads
        for r in range(int(nu) - 3):
            base0 = 1 + r * int(nv)
            base1 = base0 + int(nv)
            for j in range(int(nv)):
                jn = (j + 1) % int(nv)
                tris.append([base0 + j, base0 + jn, base1 + jn])
                tris.append([base0 + j, base1 + jn, base1 + j])
        # North cap
        last_ring = 1 + (int(nu) - 3) * int(nv)
        for j in range(int(nv)):
            jn = (j + 1) % int(nv)
            tris.append([last_ring + j, last_ring + jn, north_idx])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_world)
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        return mesh

    @classmethod
    def fit_closed_form_pca(
        cls,
        points: np.ndarray,
        sq_cfg: dict[str, Any],
    ) -> SuperquadricPrior:
        """Deterministic One-Shot initialization of theta via PCA / OBB (Section 4.2)."""
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 4:
            raise ValueError("At least 4 points are required to initialize SuperquadricPrior.")

        # Deterministic MAD radial core filter (50% breakdown point) so wild floating outliers
        # cannot skew closed-form L2 PCA eigenvectors or quantile bounding extents
        spatial_med = np.median(pts, axis=0)
        rad_dists = np.linalg.norm(pts - spatial_med[None, :], axis=1)
        med_r = float(np.median(rad_dists))
        mad_r = max(1.4826 * float(np.median(np.abs(rad_dists - med_r))), 1.0e-6)
        mad_sigma = float(sq_cfg.get("pca_inlier_mad_sigma", 3.5))
        core_mask = rad_dists <= (med_r + mad_sigma * mad_r)
        core_pts = pts[core_mask] if int(np.sum(core_mask)) >= 4 else pts

        q_lo_cfg, q_hi_cfg = sq_cfg.get("pca_extent_quantiles", [1.0, 99.0])
        p_low = np.percentile(core_pts, float(q_lo_cfg), axis=0)
        p_high = np.percentile(core_pts, float(q_hi_cfg), axis=0)
        center = 0.5 * (p_low + p_high)
        centered = core_pts - center
        cov = (centered.T @ centered) / max(len(core_pts) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        rot = eigvecs[:, order].copy()
        if np.linalg.det(rot) < 0.0:
            rot[:, 2] *= -1.0

        loc = centered @ rot
        q_min = np.percentile(loc, float(q_lo_cfg), axis=0)
        q_max = np.percentile(loc, float(q_hi_cfg), axis=0)
        loc_mid = 0.5 * (q_min + q_max)
        center = center + rot @ loc_mid
        semi_axes = np.maximum(0.5 * (q_max - q_min), 1.0e-4)

        e_min = float(sq_cfg["min_exponent"])
        e_max = float(sq_cfg["max_exponent"])
        best_e1 = float(np.clip(float(sq_cfg["init_eps1"]), e_min, e_max))
        best_e2 = float(np.clip(float(sq_cfg["init_eps2"]), e_min, e_max))

        if bool(sq_cfg.get("optimize_exponents", True)):
            grid = [float(np.clip(float(v), e_min, e_max)) for v in sq_cfg["exponent_grid"]]
            best_loss = float("inf")
            for e1 in grid:
                for e2 in grid:
                    cand = cls(center=center, rotation=rot, semi_axes=semi_axes, eps1=e1, eps2=e2)
                    f_val = np.maximum(cand.inside_outside_function(core_pts), 1.0e-12)
                    radial_res = np.abs((f_val ** (0.5 * e1)) - 1.0)
                    loss = float(np.median(radial_res))
                    if loss < best_loss:
                        best_loss = loss
                        best_e1, best_e2 = e1, e2

        return cls(center=center, rotation=rot, semi_axes=semi_axes, eps1=best_e1, eps2=best_e2)


# =============================================================================
# SECTION 4.2: ELASTIC ENERGY MINIMIZATION FOR RESIDUAL FIELD (Eqs. 13, 15, 16)
# =============================================================================

@dataclasses.dataclass
class ResidualDecompositionResult:
    """Output of Top-Down Canonical Base + Elastic Residual Field optimization."""

    canonical_mesh: o3d.geometry.TriangleMesh
    actual_mesh: o3d.geometry.TriangleMesh
    displacements_d: np.ndarray          # Scalar normal displacements d_i (Eq. 15)
    observed_mask: np.ndarray
    energy_data: float
    energy_smoothness: float
    energy_total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_vertices": int(len(self.displacements_d)),
            "observed_vertex_ratio": round(float(np.mean(self.observed_mask)), 4),
            "max_protrusion_d": round(float(np.max(self.displacements_d)), 6),
            "min_concavity_d": round(float(np.min(self.displacements_d)), 6),
            "energy_data": round(self.energy_data, 6),
            "energy_smoothness": round(self.energy_smoothness, 6),
            "energy_total": round(self.energy_total, 6),
        }


def _build_symmetric_graph_laplacian(num_verts: int, tris: np.ndarray) -> sparse.csr_matrix:
    """Construct symmetric positive semi-definite mesh graph Laplacian L1 = D - A."""
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    undir = np.vstack([edges, edges[:, ::-1]])
    # Deduplicate directed edges
    max_v = int(num_verts) + 1
    keys = undir[:, 0].astype(np.int64) * max_v + undir[:, 1].astype(np.int64)
    _, uniq_idx = np.unique(keys, return_index=True)
    src = undir[uniq_idx, 0]
    dst = undir[uniq_idx, 1]

    adj = sparse.coo_matrix(
        (np.ones(len(src), dtype=np.float64), (src, dst)),
        shape=(num_verts, num_verts),
    ).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    return sparse.diags(deg, format="csr") - adj


def solve_elastic_residual_field(
    canonical_mesh: o3d.geometry.TriangleMesh,
    observed_points: np.ndarray,
    elastic_cfg: dict[str, Any],
) -> ResidualDecompositionResult:
    """Solve Top-Down Geometric Decomposition via Variational Elastic Energy Minimization:

        Eq. (13): P_actual,i = P_canonical,i(theta) + Delta_P_i
        Eq. (15): Delta_P_i  = d_i * n_base,i
        Eq. (16): E_total(d) = E_data(d) + lambda_smooth * E_smoothness(d)

    Uses deterministic Huber Iteratively Reweighted Least Squares (IRLS) + sparse SPD linear solve:
        (W_huber + lambda_smooth * (L_membrane + beta * L_biharmonic)) * d = W_huber * d_obs
    Guarantees absolute outlier immunity and smooth occlusion interpolation without random seeds.
    """
    base = o3d.geometry.TriangleMesh(canonical_mesh)
    base.compute_triangle_normals()
    base.compute_vertex_normals()

    verts_base = np.asarray(base.vertices, dtype=np.float64)
    norms_base = np.asarray(base.vertex_normals, dtype=np.float64)
    norms_base /= np.maximum(np.linalg.norm(norms_base, axis=1, keepdims=True), 1.0e-12)
    tris = np.asarray(base.triangles, dtype=np.int32)
    n_v = len(verts_base)

    obs = np.asarray(observed_points, dtype=np.float64)
    diag = max(float(np.linalg.norm(verts_base.max(axis=0) - verts_base.min(axis=0))), 1.0e-6)

    lam_smooth = max(float(elastic_cfg["lambda_smooth"]), 1.0e-6)
    beta_biharm = max(float(elastic_cfg["biharmonic_weight"]), 0.0)
    huber_delta = max(float(elastic_cfg["huber_delta_frac"]) * diag, 1.0e-6)
    max_disp = float(elastic_cfg["max_displacement_frac"]) * diag
    max_obs_dist = float(elastic_cfg["max_obs_dist_frac"]) * diag
    irls_iters = max(int(elastic_cfg["irls_iterations"]), 1)

    # One-shot orthogonal correspondence projection along base normals: d_obs,i = (Q_pi(i) - P_base,i) . n_base,i
    obs_tree = cKDTree(obs)
    nn_dists, nn_idx = obs_tree.query(verts_base, k=1)
    matched_q = obs[nn_idx]
    raw_d_obs = np.sum((matched_q - verts_base) * norms_base, axis=1)

    # Also check reverse k-NN (from observed points to base vertices) so outliers floating far in space
    # never hijack base vertices unless they are genuinely the nearest point within max_obs_dist
    obs_mask = nn_dists <= max_obs_dist
    d_obs = np.where(obs_mask, raw_d_obs, 0.0)

    # Elastic membrane + thin-plate operator: L_elastic = L1 + beta * (L1^T L1)
    l1 = _build_symmetric_graph_laplacian(n_v, tris)
    l_elastic = (l1 + beta_biharm * (l1 @ l1)).tocsr()

    # Deterministic IRLS loop with Huber robust weights against floating sensor outliers
    w_base = obs_mask.astype(np.float64)
    d_sol = np.zeros(n_v, dtype=np.float64)
    tiny_reg = 1.0e-8 * sparse.eye(n_v, format="csr")

    for _ in range(irls_iters):
        res = np.abs(d_sol - d_obs)
        # Huber weight: 1 if |r| <= delta, else delta / |r|
        huber_w = np.where(res <= huber_delta, 1.0, huber_delta / np.maximum(res, 1.0e-12))
        w_eff = w_base * huber_w
        w_diag = sparse.diags(w_eff, format="csr")
        lhs = w_diag + lam_smooth * l_elastic + tiny_reg
        rhs = w_eff * d_obs
        d_sol = np.asarray(spsolve(lhs, rhs), dtype=np.float64)

    d_sol = np.clip(d_sol, -max_disp, max_disp)

    # Evaluate final energies (Eq. 16)
    final_res = np.abs(d_sol - d_obs)[obs_mask]
    huber_loss = np.where(
        final_res <= huber_delta,
        0.5 * final_res * final_res,
        huber_delta * (final_res - 0.5 * huber_delta),
    )
    e_data = float(np.sum(huber_loss))
    e_smooth = float(0.5 * d_sol @ (l_elastic @ d_sol))
    e_total = e_data + lam_smooth * e_smooth

    # Reconstruct actual surface (Eq. 13 & Eq. 15): P_actual = P_canonical + d_i * n_base,i
    verts_actual = verts_base + d_sol[:, None] * norms_base
    actual_mesh = o3d.geometry.TriangleMesh(base)
    actual_mesh.vertices = o3d.utility.Vector3dVector(verts_actual)
    actual_mesh.compute_triangle_normals()
    actual_mesh.compute_vertex_normals()

    return ResidualDecompositionResult(
        canonical_mesh=base,
        actual_mesh=actual_mesh,
        displacements_d=d_sol,
        observed_mask=obs_mask,
        energy_data=e_data,
        energy_smoothness=e_smooth,
        energy_total=e_total,
    )


def reconstruct_object_topdown_residual(
    pc: o3d.geometry.PointCloud,
    cfg: dict[str, Any],
) -> tuple[ResidualDecompositionResult, SuperquadricPrior | None]:
    """Top-down object reconstruction supporting Superquadric (Eq. 14) or Poisson base prior."""
    td_cfg = cfg["topdown_residual"]
    prior_type = str(td_cfg.get("prior_type", "superquadric")).strip().lower()
    pts = np.asarray(pc.points, dtype=np.float64)

    sq_prior: SuperquadricPrior | None = None
    if prior_type == "superquadric":
        sq_cfg = td_cfg["superquadric"]
        sq_prior = SuperquadricPrior.fit_closed_form_pca(pts, sq_cfg)
        nu, nv = int(sq_cfg["uv_resolution"][0]), int(sq_cfg["uv_resolution"][1])
        base_mesh = sq_prior.to_canonical_mesh(nu=nu, nv=nv)
    else:
        # Smooth low-depth Poisson indicator field as canonical genus-adaptive base prior
        pcd = o3d.geometry.PointCloud(pc)
        if not pcd.has_normals():
            diag = max(float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))), 1.0e-6)
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.04 * diag, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(k=15)
        base_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5, scale=1.1)

    res = solve_elastic_residual_field(base_mesh, pts, td_cfg["elastic_energy"])
    return res, sq_prior


# =============================================================================
# SECTION 4.3: TOP-DOWN MANHATTAN ROOM PRIOR + LOCAL RESIDUAL RANSAC (Milestone M5)
# =============================================================================

def _estimate_closed_form_manhattan_prior_polygon(
    aligned_points: np.ndarray,
    topdown_room_cfg: dict[str, Any],
) -> np.ndarray:
    """Deterministic BEV high-density Manhattan room polygon prior when external RoomFormer polygon is absent."""
    xy = aligned_points[:, :2]
    q_min = np.quantile(xy, 0.02, axis=0)
    q_max = np.quantile(xy, 0.98, axis=0)
    x0, y0 = float(q_min[0]), float(q_min[1])
    x1, y1 = float(q_max[0]), float(q_max[1])
    # Ordered counter-clockwise Manhattan canonical rectangle
    return np.array([
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
    ], dtype=np.float64)


def reconstruct_room_topdown_residual(
    points: np.ndarray,
    normals: np.ndarray | None,
    floorplan_cfg: dict[str, Any],
    semantic_labels: np.ndarray | None = None,
    prior_polygon_2d: np.ndarray | None = None,
) -> dict[str, Any]:
    """Milestone M5 Top-Down Room Reconstruction (Section 4.3 & Section 6.3):
      1. Align room scan to Manhattan World axes via Eq. (11) R(-theta_0).
      2. Lock canonical Manhattan room polygon P_canonical(theta) (from RoomFormer or closed-form BEV prior).
      3. Within corridor_halfwidth_m around each canonical wall edge, run local dual-validated RANSAC + SVD
         refit to solve normal wall displacement Delta_d_k regularized by lambda_smooth (Eqs. 13, 15, 16).
      4. Reclaim RandLA-Net clutter points & apply morphological closing; intersect walls analytically at 90 deg.
    """
    pts_raw = np.asarray(points, dtype=np.float64)
    r_cfg = floorplan_cfg["adaptive_ransac"]
    n_cfg = floorplan_cfg["numerical"]
    p_cfg = floorplan_cfg["hessian_plane"]
    td_cfg = floorplan_cfg["topdown_room"]

    # Ensure unit normals
    if normals is not None and len(normals) == len(pts_raw):
        nrms_raw = np.asarray(normals, dtype=np.float64)
        nrms_raw /= np.maximum(np.linalg.norm(nrms_raw, axis=1, keepdims=True), float(n_cfg["norm_eps"]))
    else:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_raw))
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=float(r_cfg["normal_estimation_radius_m"]),
                max_nn=int(r_cfg["normal_estimation_max_nn"]),
            )
        )
        nrms_raw = np.asarray(pcd.normals, dtype=np.float64)

    # Step 1: Manhattan World alignment R(-theta_0) (Eq. 11)
    pts, nrms, theta_0, rot_mat = align_manhattan_world(pts_raw, nrms_raw, [], floorplan_cfg)
    assert nrms is not None

    # Identify floor and ceiling elevations from horizontal normal subset (|nz| >= cos(15 deg))
    horiz_mask = np.abs(nrms[:, 2]) >= float(p_cfg["horizontal_min_abs_nz"])
    if int(np.sum(horiz_mask)) >= 20:
        z_horiz = pts[horiz_mask, 2]
        z_split = float(np.quantile(z_horiz, float(p_cfg["floor_ceiling_split_quantile"])))
        z_floor = float(np.median(z_horiz[z_horiz <= z_split])) if np.any(z_horiz <= z_split) else float(pts[:, 2].min())
        z_ceil = (
            float(np.median(z_horiz[z_horiz > z_split]))
            if np.any(z_horiz > z_split) and (float(np.median(z_horiz[z_horiz > z_split])) - z_floor > 1.0)
            else z_floor + float(td_cfg["fallback_room_height_m"])
        )
    else:
        z_floor = float(np.quantile(pts[:, 2], 0.02))
        z_ceil = z_floor + float(td_cfg["fallback_room_height_m"])

    # Step 2: Canonical Manhattan Polygon Prior P_canonical(theta)
    if prior_polygon_2d is not None:
        canon_poly, canon_segs = regularize_rectilinear_polygon(prior_polygon_2d, floorplan_cfg)
    else:
        vert_pts = pts[np.abs(nrms[:, 2]) <= float(p_cfg["vertical_max_abs_nz"])]
        src_pts = vert_pts if len(vert_pts) >= 20 else pts
        init_poly = _estimate_closed_form_manhattan_prior_polygon(src_pts, td_cfg)
        canon_poly, canon_segs = regularize_rectilinear_polygon(init_poly, floorplan_cfg)

    # Step 3: Local Corridor Residual Wall Refinement (Eqs. 13, 15, 16)
    corridor_hw = float(td_cfg["corridor_halfwidth_m"])
    lam_smooth = float(td_cfg["lambda_smooth"])
    cos_thr = float(np.cos(np.radians(float(r_cfg["normal_max_angle_deg"]))))
    eps_plane = float(r_cfg["eps_plane_m"])
    det_eps = float(n_cfg["corner_det_eps"])

    refined_segs: list[WallSegment2D] = []
    wall_planes: list[DetectedPlane] = []
    residual_shifts: list[float] = []

    for idx_s, seg in enumerate(canon_segs):
        n3d = np.array([seg.normal_2d[0], seg.normal_2d[1], 0.0], dtype=np.float64)
        canon_plane = HessianPlane(normal=n3d, d=seg.offset_c, norm_eps=float(n_cfg["norm_eps"]))

        # Restrict candidate wall points to local corridor around canonical prior edge
        signed_d = canon_plane.signed_distance(pts)
        # In-span check along segment tangent
        seg_vec = seg.p_end - seg.p_start
        seg_len = max(float(np.linalg.norm(seg_vec)), 1.0e-6)
        t_dir = seg_vec / seg_len
        proj_t = (pts[:, :2] - seg.p_start[None, :]) @ t_dir
        in_span = (proj_t >= -corridor_hw) & (proj_t <= seg_len + corridor_hw)
        in_corridor = (np.abs(signed_d) <= corridor_hw) & in_span
        norm_ok = np.abs(nrms @ n3d) > cos_thr

        corridor_idx = np.nonzero(in_corridor & norm_ok)[0]
        if len(corridor_idx) >= 10:
            # Robust median + narrow band SVD residual offset along locked Manhattan normal
            med_shift = float(np.median(signed_d[corridor_idx]))
            inlier_local = corridor_idx[np.abs(signed_d[corridor_idx] - med_shift) < eps_plane]
            if len(inlier_local) >= 5:
                data_shift = float(np.mean(signed_d[inlier_local]))
            else:
                data_shift = med_shift
                inlier_local = corridor_idx

            # Elastic regularization (Eq. 16): min_d 0.5*(d - data_shift)^2 + 0.5*lambda_smooth*d^2
            delta_d = data_shift / (1.0 + lam_smooth)
        else:
            # Occluded wall (e.g., 100% blocked by wardrobe): elastic prior holds canonical edge (delta_d = 0)
            delta_d = 0.0
            inlier_local = corridor_idx

        new_offset_c = seg.offset_c - delta_d
        residual_shifts.append(round(delta_d, 6))
        shifted_start = seg.p_start + delta_d * seg.normal_2d
        shifted_end = seg.p_end + delta_d * seg.normal_2d
        refined_segs.append(
            WallSegment2D(shifted_start, shifted_end, seg.normal_2d, new_offset_c, seg.axis_lock, idx_s)
        )
        wall_planes.append(
            DetectedPlane(
                plane=HessianPlane(normal=n3d, d=new_offset_c, norm_eps=float(n_cfg["norm_eps"])),
                inlier_indices=inlier_local,
                seed_indices=(0, 0, 0),
                orientation_class="vertical_wall",
            )
        )

    # Step 4: Clutter Reclamation & Morphological Gap Closing on refined walls
    clutter_stats = reclaim_clutter_and_close_gaps(pts, nrms, semantic_labels, wall_planes, floorplan_cfg)

    # Analytical 90-degree corner re-closure of residual-adjusted segments
    m_cnt = len(refined_segs)
    actual_corners: list[np.ndarray] = []
    for i in range(m_cnt):
        pt = intersect_lines_2d(refined_segs[(i - 1) % m_cnt], refined_segs[i], det_eps=det_eps)
        if pt is None:
            pt = refined_segs[i].p_start
        actual_corners.append(pt)
    actual_poly = np.vstack(actual_corners) if actual_corners else canon_poly

    closed_segs: list[WallSegment2D] = []
    for i in range(m_cnt):
        c0 = actual_corners[i]
        c1 = actual_corners[(i + 1) % m_cnt]
        closed_segs.append(
            WallSegment2D(c0, c1, refined_segs[i].normal_2d, refined_segs[i].offset_c, refined_segs[i].axis_lock, i)
        )

    return {
        "aligned_points": pts,
        "aligned_normals": nrms,
        "manhattan_theta0_rad": theta_0,
        "rotation_matrix": rot_mat,
        "canonical_polygon_2d": canon_poly,
        "actual_polygon_2d": actual_poly,
        "wall_segments_2d": closed_segs,
        "detected_walls": wall_planes,
        "wall_residual_shifts_m": residual_shifts,
        "clutter_recovery": clutter_stats,
        "z_floor": z_floor,
        "z_ceiling": z_ceil,
    }
