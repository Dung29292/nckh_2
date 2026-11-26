"""Exact Mathematical Surface Modeling & Representation Hierarchy (Section 2 of nckh.pdf).

Implements Equations (1) through (9) with zero hardcoded geometric constants:
  - Eq. (1) & Eq. (2): HessianPlane (ax + by + cz + d = 0, a^2 + b^2 + c^2 = 1)
  - Eq. (3)          : QuadricSurface (P^T M P = 0 homogeneous 4x4 symmetric form)
  - Eq. (4)          : CircularCylinder (orthogonal axis distance d(P, axis) = ||(P - C0) x a|| / ||a||)
  - Eq. (5)          : SphereSurface ((x - x0)^2 + (y - y0)^2 + (z - z0)^2 = R^2)
  - Eq. (6)          : LocalMongePatch & mls_monge_project_point_cloud (quadratic paraboloid MLS)
  - Eq. (7) & Eq. (8): ParametricSurface & mobius_strip_surface (180-deg non-orientable strip)
  - Eq. (9)          : SignedDistanceField (f_theta(x, y, z) = s implicit zero-level set)
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import trimesh


# =============================================================================
# SECTION 2.1: ARCHITECTURAL PLANES IN HESSIAN NORMAL FORM (Eq. 1 & Eq. 2)
# =============================================================================

@dataclasses.dataclass
class HessianPlane:
    """Architectural plane in Hessian Normal Form (Section 2.1, Eq. 1 & Eq. 2).

    Governing Equation (Eq. 1):
        a*x + b*y + c*z + d = 0   subject to   a^2 + b^2 + c^2 = 1
    where n = (a, b, c) is the unit normal vector and d is the signed orthogonal
    distance from the coordinate origin to the plane.
    """

    normal: np.ndarray
    d: float
    norm_eps: float = 1.0e-12
    canonical_sign: bool = False

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=np.float64).reshape(3)
        norm_val = float(np.linalg.norm(n))
        if norm_val <= self.norm_eps:
            raise ValueError(f"Degenerate plane normal ||n|| = {norm_val} <= {self.norm_eps}")
        # Enforce Eq. (1) unit-normal constraint: a^2 + b^2 + c^2 = 1
        n = n / norm_val
        d_val = float(self.d) / norm_val

        if self.canonical_sign:
            # Canonicalize sign gauge (n, d) ~ (-n, -d) deterministically
            if d_val > self.norm_eps or (abs(d_val) <= self.norm_eps and self._lex_negative(n)):
                n = -n
                d_val = -d_val

        self.normal = n
        self.d = d_val

    @staticmethod
    def _lex_negative(n: np.ndarray) -> bool:
        for comp in n:
            if abs(float(comp)) > 1.0e-9:
                return float(comp) < 0.0
        return False

    @property
    def a(self) -> float:
        return float(self.normal[0])

    @property
    def b(self) -> float:
        return float(self.normal[1])

    @property
    def c(self) -> float:
        return float(self.normal[2])

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Compute signed algebraic orthogonal distance a*x_i + b*y_i + c*z_i + d."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return pts @ self.normal + self.d

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Compute exact point-to-plane distance (Eq. 2): dist(P, plane) = |a*x_i + b*y_i + c*z_i + d|."""
        return np.abs(self.signed_distance(points))

    def project_points(self, points: np.ndarray) -> np.ndarray:
        """Orthogonally project 3D points onto the Hessian plane."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        sd = self.signed_distance(pts)
        return pts - sd[:, None] * self.normal[None, :]

    def classify_orientation(
        self,
        plane_cfg: dict[str, Any],
        inlier_z_median: float | None = None,
        scene_z_split: float | None = None,
        scene_z_bounds: tuple[float, float] | None = None,
    ) -> str:
        """Classify architectural component from unit normal n = (nx, ny, nz) (Section 1.2 & 2.1):
          - Horizontal planes: |nz| >= horizontal_min_abs_nz (distinguishing Floor, Ceiling, and intermediate Desk/Table)
          - Vertical planes  : |nz| <= vertical_max_abs_nz   (Walls: nz ~ 0, a^2 + b^2 ~ 1)
        """
        horiz_min = float(plane_cfg["horizontal_min_abs_nz"])
        vert_max = float(plane_cfg["vertical_max_abs_nz"])
        abs_nz = abs(float(self.normal[2]))

        if abs_nz >= horiz_min:
            if inlier_z_median is not None and scene_z_bounds is not None:
                z_lo, z_hi = scene_z_bounds
                z_span = max(z_hi - z_lo, 1.0e-6)
                rel_z = (inlier_z_median - z_lo) / z_span
                f_q = float(plane_cfg.get("floor_max_height_quantile", 0.15))
                c_q = float(plane_cfg.get("ceiling_min_height_quantile", 0.85))
                if rel_z <= f_q:
                    return "horizontal_floor"
                if rel_z >= c_q:
                    return "horizontal_ceiling"
                return "horizontal_furniture"
            if inlier_z_median is not None and scene_z_split is not None:
                return "horizontal_floor" if inlier_z_median <= scene_z_split else "horizontal_ceiling"
            return "horizontal_plane"
        if abs_nz <= vert_max:
            return "vertical_wall"
        return "oblique_plane"

    def rotate_z(self, angle_rad: float) -> HessianPlane:
        """Rotate plane about world Z-axis by angle_rad (origin-preserving, so d is invariant)."""
        c_ang = float(np.cos(angle_rad))
        s_ang = float(np.sin(angle_rad))
        rot_z = np.array([
            [c_ang, -s_ang, 0.0],
            [s_ang,  c_ang, 0.0],
            [0.0,    0.0,   1.0],
        ], dtype=np.float64)
        return HessianPlane(
            normal=rot_z @ self.normal,
            d=self.d,
            norm_eps=self.norm_eps,
            canonical_sign=False,
        )

    @classmethod
    def from_three_points(
        cls,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        collinear_eps: float,
        norm_eps: float = 1.0e-12,
        canonical_sign: bool = False,
    ) -> HessianPlane | None:
        """Minimal 3-point plane hypothesis (Section 3.1): n = (P2 - P1) x (P3 - P1)."""
        v1 = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
        v2 = np.asarray(p3, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
        cross_n = np.cross(v1, v2)
        cross_norm = float(np.linalg.norm(cross_n))
        if cross_norm <= collinear_eps:
            return None
        unit_n = cross_n / cross_norm
        d_val = -float(np.dot(unit_n, p1))
        return cls(normal=unit_n, d=d_val, norm_eps=norm_eps, canonical_sign=canonical_sign)

    @classmethod
    def fit_svd(
        cls,
        points: np.ndarray,
        weights: np.ndarray | None = None,
        norm_eps: float = 1.0e-12,
        canonical_sign: bool = False,
    ) -> HessianPlane:
        """Global Total Least Squares plane refit via centered SVD (Section 3.1 Global Refit)."""
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 3:
            raise ValueError("At least 3 non-collinear points are required for SVD plane fit.")

        if weights is not None:
            w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
            w_sum = max(float(np.sum(w)), norm_eps)
            centroid = np.sum(pts * w[:, None], axis=0) / w_sum
            centered = (pts - centroid) * np.sqrt(w / w_sum)[:, None]
        else:
            centroid = np.mean(pts, axis=0)
            centered = pts - centroid

        # Right singular vector corresponding to smallest singular value minimizes sum (n.x + d)^2
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        unit_n = vh[-1]
        d_val = -float(np.dot(unit_n, centroid))
        return cls(normal=unit_n, d=d_val, norm_eps=norm_eps, canonical_sign=canonical_sign)

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": round(self.a, 8),
            "b": round(self.b, 8),
            "c": round(self.c, 8),
            "d": round(self.d, 8),
            "unit_norm_sq": round(float(np.dot(self.normal, self.normal)), 8),
        }


# =============================================================================
# SECTION 2.2: ANALYTICAL QUADRIC SURFACES (Eq. 3, Eq. 4 Cylinder, Eq. 5 Sphere)
# =============================================================================

@dataclasses.dataclass
class QuadricSurface:
    """General Analytical Quadric Surface in homogeneous symmetric matrix form (Section 2.2, Eq. 3):

        P^T M P = 0  <=>  A*x^2 + B*y^2 + C*z^2 + 2*D*x*y + 2*E*y*z + 2*F*z*x
                          + 2*G*x + 2*H*y + 2*I*z + J = 0
    where P = [x, y, z, 1]^T and M = M^T in R^{4x4}.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError(f"Quadric matrix M must have shape (4, 4), got {m.shape}")
        # Enforce exact symmetry M = 0.5 * (M + M^T)
        self.matrix = 0.5 * (m + m.T)

    @classmethod
    def from_coefficients(
        cls,
        A: float, B: float, C: float,
        D: float, E: float, F: float,
        G: float, H: float, I: float, J: float,
    ) -> QuadricSurface:
        """Construct symmetric 4x4 matrix M directly from Eq. (3) scalar coefficients (A..J)."""
        m = np.array([
            [A, D, F, G],
            [D, B, E, H],
            [F, E, C, I],
            [G, H, I, J],
        ], dtype=np.float64)
        return cls(matrix=m)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate algebraic quadric form P^T M P (Eq. 3) for N points."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        ones = np.ones((len(pts), 1), dtype=np.float64)
        hom_p = np.hstack([pts, ones])  # (N, 4)
        return np.einsum("ni,ij,nj->n", hom_p, self.matrix, hom_p)

    @classmethod
    def fit_algebraic_svd(cls, points: np.ndarray) -> QuadricSurface:
        """Fit homogeneous 10-parameter quadric (Eq. 3) via SVD nullspace of design matrix."""
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 9:
            raise ValueError("At least 9 points are required to fit a general 3D quadric.")
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        design = np.column_stack([
            x * x, y * y, z * z,
            2.0 * x * y, 2.0 * y * z, 2.0 * z * x,
            2.0 * x, 2.0 * y, 2.0 * z,
            np.ones_like(x),
        ])
        _, _, vh = np.linalg.svd(design, full_matrices=False)
        coeffs = vh[-1]
        return cls.from_coefficients(*coeffs.tolist())


@dataclasses.dataclass
class CircularCylinder:
    """Right Circular Cylinder for architectural columns & pipes (Section 2.2, Eq. 4).

    Defined by axis anchor point C0(x0, y0, z0), unit axis direction a = (ax, ay, az),
    and radius r > 0.
    """

    anchor: np.ndarray
    axis: np.ndarray
    radius: float
    norm_eps: float = 1.0e-12

    def __post_init__(self) -> None:
        c0 = np.asarray(self.anchor, dtype=np.float64).reshape(3)
        ax = np.asarray(self.axis, dtype=np.float64).reshape(3)
        ax_norm = float(np.linalg.norm(ax))
        if ax_norm <= self.norm_eps:
            raise ValueError("Cylinder axis vector must be non-zero.")
        if self.radius <= 0.0:
            raise ValueError(f"Cylinder radius must be positive, got {self.radius}")
        ax = ax / ax_norm
        # Canonicalize anchor C0 orthogonal to axis a so (C0 . a) = 0
        c0 = c0 - float(np.dot(c0, ax)) * ax
        self.anchor = c0
        self.axis = ax
        self.radius = float(self.radius)

    def axis_distance(self, points: np.ndarray) -> np.ndarray:
        """Orthogonal distance from point P to cylinder axis (Eq. 4):
            d(P, axis) = ||(P - C0) x a|| / ||a||
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        diff = pts - self.anchor[None, :]
        cross_prod = np.cross(diff, self.axis[None, :])
        axis_norm = max(float(np.linalg.norm(self.axis)), self.norm_eps)
        return np.linalg.norm(cross_prod, axis=1) / axis_norm

    def residual(self, points: np.ndarray) -> np.ndarray:
        """Unsigned radial residual |d(P, axis) - r|."""
        return np.abs(self.axis_distance(points) - self.radius)

    def inlier_mask(self, points: np.ndarray, eps_cylinder: float) -> np.ndarray:
        """Cylinder membership test (Section 2.2): |d(P, axis) - r| < eps_cylinder."""
        return self.residual(points) < float(eps_cylinder)

    def to_quadric(self) -> QuadricSurface:
        """Convert cylinder ||(I - a a^T)(P - C0)||^2 - r^2 = 0 into Eq. (3) QuadricSurface."""
        proj = np.eye(3, dtype=np.float64) - np.outer(self.axis, self.axis)
        m3 = proj.T @ proj  # proj is idempotent symmetric: proj^T proj = proj
        g_vec = -m3 @ self.anchor
        j_val = float(self.anchor @ m3 @ self.anchor - self.radius * self.radius)
        mat = np.empty((4, 4), dtype=np.float64)
        mat[:3, :3] = m3
        mat[:3, 3] = g_vec
        mat[3, :3] = g_vec
        mat[3, 3] = j_val
        return QuadricSurface(matrix=mat)

    @classmethod
    def refit_least_squares(
        cls,
        points: np.ndarray,
        init_cyl: CircularCylinder,
        gn_max_iterations: int,
        gn_tol: float,
        norm_eps: float = 1.0e-12,
    ) -> CircularCylinder:
        """Global nonlinear Gauss-Newton / Trust-Region refit over all cylinder inliers (Table 1)."""
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 6:
            return init_cyl

        # Parameterize axis on S^2 via spherical angles (theta, phi), 2D offset (u, v) in tangent basis, and radius r
        a0 = init_cyl.axis
        theta0 = float(np.arccos(np.clip(a0[2], -1.0, 1.0)))
        phi0 = float(np.arctan2(a0[1], a0[0]))

        def _unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            th, ph, c_x, c_y, c_z, rad = params
            ax = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], dtype=np.float64)
            c0 = np.array([c_x, c_y, c_z], dtype=np.float64)
            c0 = c0 - float(np.dot(c0, ax)) * ax
            return c0, ax, abs(float(rad))

        def _fun(params: np.ndarray) -> np.ndarray:
            c0, ax, rad = _unpack(params)
            d_ax = np.linalg.norm(np.cross(pts - c0[None, :], ax[None, :]), axis=1)
            # Last residual weakly constrains c0 along axis so Jacobian stays full-rank
            return np.append(d_ax - rad, float(np.dot(params[2:5], ax)))

        x0 = np.array([theta0, phi0, *init_cyl.anchor.tolist(), init_cyl.radius], dtype=np.float64)
        res = least_squares(_fun, x0, method="trf", max_nfev=int(gn_max_iterations) * 6, ftol=gn_tol, xtol=gn_tol)
        c0_opt, ax_opt, r_opt = _unpack(res.x)
        return cls(anchor=c0_opt, axis=ax_opt, radius=max(r_opt, norm_eps), norm_eps=norm_eps)

    @classmethod
    def fit_ransac_with_normals(
        cls,
        points: np.ndarray,
        normals: np.ndarray,
        quad_cfg: dict[str, Any],
        num_cfg: dict[str, Any],
        constrain_vertical: bool = False,
        seed: int = 42,
    ) -> tuple[CircularCylinder | None, np.ndarray]:
        """2-point + normal RANSAC cylinder estimator + global nonlinear refit (Section 2.2 & Table 1).

        Minimal seed: 2 oriented surface points (P1, n1), (P2, n2):
          - Cylinder axis direction: a = (n1 x n2) / ||n1 x n2||
          - Project P1, n1, P2, n2 onto plane orthogonal to a to solve exact 2D circle center & radius r.
        """
        pts = np.asarray(points, dtype=np.float64)
        nrms = np.asarray(normals, dtype=np.float64)
        n_pts = len(pts)
        min_inliers = int(quad_cfg["min_inliers"])
        if n_pts < max(min_inliers, 6):
            return None, np.empty(0, dtype=np.int64)

        norm_eps = float(num_cfg["norm_eps"])
        eps_cyl = float(quad_cfg["eps_cylinder_m"])
        r_min, r_max = (float(quad_cfg["cylinder_radius_range_m"][0]), float(quad_cfg["cylinder_radius_range_m"][1]))
        max_iters = int(quad_cfg["ransac_iterations"])
        ortho_max_cos = float(quad_cfg["normal_axis_ortho_max_cos"])
        vert_min_az = float(quad_cfg["vertical_column_min_abs_az"])

        rng = np.random.default_rng(seed)
        best_cyl: CircularCylinder | None = None
        best_inliers = np.empty(0, dtype=np.int64)

        for _ in range(max_iters):
            i1, i2 = rng.choice(n_pts, size=2, replace=False)
            p1, p2 = pts[i1], pts[i2]
            n1, n2 = nrms[i1], nrms[i2]

            ax = np.cross(n1, n2)
            ax_len = float(np.linalg.norm(ax))
            if ax_len <= 1.0e-3:
                continue
            ax = ax / ax_len
            if constrain_vertical and abs(float(ax[2])) < vert_min_az:
                continue

            # Project onto 2D plane perpendicular to axis `ax`: C0 = P1 + t1 * n1_perp ~ P2 + t2 * n2_perp
            p1_p = p1 - float(np.dot(p1, ax)) * ax
            p2_p = p2 - float(np.dot(p2, ax)) * ax
            n1_p = n1 - float(np.dot(n1, ax)) * ax
            n2_p = n2 - float(np.dot(n2, ax)) * ax
            l1, l2 = float(np.linalg.norm(n1_p)), float(np.linalg.norm(n2_p))
            if l1 <= 0.2 or l2 <= 0.2:
                continue
            n1_p /= l1
            n2_p /= l2

            # Least-squares intersection of two 2D normal rays: [n1_p, -n2_p] [t1, t2]^T = p2_p - p1_p
            a_mat = np.column_stack([n1_p, -n2_p])
            rhs = p2_p - p1_p
            ata = a_mat.T @ a_mat
            if abs(float(np.linalg.det(ata))) <= norm_eps:
                continue
            t_sol = np.linalg.solve(ata, a_mat.T @ rhs)
            c0 = 0.5 * ((p1_p + t_sol[0] * n1_p) + (p2_p + t_sol[1] * n2_p))
            r_cand = 0.5 * (abs(float(t_sol[0])) + abs(float(t_sol[1])))
            if not (r_min <= r_cand <= r_max):
                continue

            cand = cls(anchor=c0, axis=ax, radius=r_cand, norm_eps=norm_eps)
            dist_ok = cand.inlier_mask(pts, eps_cyl)
            # Dual normal-orthogonality validation: surface normals on a cylinder satisfy |n_Q . a| ~ 0
            norm_ok = np.abs(nrms @ cand.axis) <= ortho_max_cos
            inliers = np.nonzero(dist_ok & norm_ok)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_cyl = cand

        if best_cyl is None or len(best_inliers) < min_inliers:
            return None, np.empty(0, dtype=np.int64)

        # Global refit over all validated non-seed inliers
        refined = cls.refit_least_squares(
            pts[best_inliers],
            best_cyl,
            gn_max_iterations=int(quad_cfg["gn_max_iterations"]),
            gn_tol=float(quad_cfg["gn_tol"]),
            norm_eps=norm_eps,
        )
        final_mask = refined.inlier_mask(pts, eps_cyl) & (np.abs(nrms @ refined.axis) <= ortho_max_cos)
        final_inliers = np.nonzero(final_mask)[0]
        return refined, final_inliers


@dataclasses.dataclass
class SphereSurface:
    """Analytical Sphere Surface for domes & spherical features (Section 2.2, Eq. 5):
        (x - x0)^2 + (y - y0)^2 + (z - z0)^2 = R^2   <=>   ||P - C|| = R
    """

    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        if self.radius <= 0.0:
            raise ValueError(f"Sphere radius R must be positive, got {self.radius}")
        self.radius = float(self.radius)

    def distance_to_center(self, points: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance ||P - C|| (Eq. 5)."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return np.linalg.norm(pts - self.center[None, :], axis=1)

    def residual(self, points: np.ndarray) -> np.ndarray:
        """Compute unsigned radial residual | ||P - C|| - R |."""
        return np.abs(self.distance_to_center(points) - self.radius)

    def inlier_mask(self, points: np.ndarray, eps_sphere: float) -> np.ndarray:
        """Sphere membership condition: | ||P - C|| - R | < eps_sphere."""
        return self.residual(points) < float(eps_sphere)

    def to_quadric(self) -> QuadricSurface:
        """Convert sphere ||P - C||^2 - R^2 = 0 to homogeneous Eq. (3) QuadricSurface."""
        x0, y0, z0 = self.center
        j_val = float(np.dot(self.center, self.center) - self.radius * self.radius)
        return QuadricSurface.from_coefficients(
            1.0, 1.0, 1.0,
            0.0, 0.0, 0.0,
            -x0, -y0, -z0, j_val,
        )

    @classmethod
    def fit_least_squares(
        cls,
        points: np.ndarray,
        gn_max_iterations: int = 25,
        gn_tol: float = 1.0e-7,
    ) -> SphereSurface:
        """Exact algebraic linear initialization (2*C^T P + (R^2 - ||C||^2) = ||P||^2) + Gauss-Newton refit."""
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 4:
            raise ValueError("At least 4 non-coplanar points are required to fit a sphere.")

        design = np.column_stack([2.0 * pts, np.ones(len(pts), dtype=np.float64)])
        rhs = np.sum(pts * pts, axis=1)
        sol, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
        c_init = sol[:3]
        r_init = float(np.sqrt(max(sol[3] + float(np.dot(c_init, c_init)), 1.0e-12)))

        def _fun(params: np.ndarray) -> np.ndarray:
            c_cur = params[:3]
            r_cur = abs(float(params[3]))
            return np.linalg.norm(pts - c_cur[None, :], axis=1) - r_cur

        res = least_squares(
            _fun,
            np.append(c_init, r_init),
            method="trf",
            max_nfev=int(gn_max_iterations) * 5,
            ftol=gn_tol,
            xtol=gn_tol,
        )
        return cls(center=res.x[:3], radius=abs(float(res.x[3])))


# =============================================================================
# SECTION 2.3: FREEFORM ERGONOMIC SURFACES — LOCAL MONGE PATCH / MLS (Eq. 6)
# =============================================================================

@dataclasses.dataclass
class LocalMongePatch:
    """Local bivariate quadratic Monge patch approximation (Section 2.3, Eq. 6):

        z = f(x, y) = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2

    defined in an orthonormal local tangent frame (u_axis, v_axis, n_axis) at origin P0.
    Preserves positive dome curvature (K > 0) and negative saddle curvature (K < 0) simultaneously.
    """

    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    n_axis: np.ndarray
    coeffs: np.ndarray  # shape (6,): [c0, c1, c2, c3, c4, c5]

    def evaluate_local(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """Evaluate z = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2 (Eq. 6)."""
        c0, c1, c2, c3, c4, c5 = self.coeffs
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        return c0 + c1 * xa + c2 * ya + c3 * xa * xa + c4 * xa * ya + c5 * ya * ya

    @property
    def projected_origin(self) -> np.ndarray:
        """Denoised surface point at local (x, y) = (0, 0): P0 + c0 * n_axis."""
        return self.origin + float(self.coeffs[0]) * self.n_axis

    @property
    def differential_normal(self) -> np.ndarray:
        """Analytical differential unit normal at (0, 0) from Monge gradient (-c1, -c2, 1)."""
        c1, c2 = float(self.coeffs[1]), float(self.coeffs[2])
        n_world = -c1 * self.u_axis - c2 * self.v_axis + self.n_axis
        return n_world / max(float(np.linalg.norm(n_world)), 1.0e-12)

    @property
    def gaussian_curvature(self) -> float:
        """Analytical Gaussian curvature K = det(II) / det(I) at (x, y) = (0, 0)."""
        _, c1, c2, c3, c4, c5 = self.coeffs
        denom = (1.0 + c1 * c1 + c2 * c2) ** 2
        return float((4.0 * c3 * c5 - c4 * c4) / max(denom, 1.0e-12))

    @property
    def mean_curvature(self) -> float:
        """Analytical Mean curvature H at (x, y) = (0, 0)."""
        _, c1, c2, c3, c4, c5 = self.coeffs
        num = (1.0 + c2 * c2) * c3 - c1 * c2 * c4 + (1.0 + c1 * c1) * c5
        denom = (1.0 + c1 * c1 + c2 * c2) ** 1.5
        return float(num / max(denom, 1.0e-12))


def fit_local_monge_patch(
    query_point: np.ndarray,
    neighbor_points: np.ndarray,
    bandwidth: float | None = None,
    reg_eps: float = 1.0e-6,
    ref_normal: np.ndarray | None = None,
) -> LocalMongePatch:
    """Fit exact quadratic Monge paraboloid z = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2 (Eq. 6)."""
    p0 = np.asarray(query_point, dtype=np.float64).reshape(3)
    nbrs = np.asarray(neighbor_points, dtype=np.float64)
    if len(nbrs) < 6:
        raise ValueError("At least 6 neighbor points are required to fit a 6-DOF quadratic Monge patch.")

    centered = nbrs - nbrs.mean(axis=0)
    cov = (centered.T @ centered) / max(len(nbrs) - 1, 1)
    _, eigvecs = np.linalg.eigh(cov)
    # Ascending eigenvalues: eigvecs[:, 0] is smallest-variance normal axis, [:, 2] & [:, 1] span tangent plane
    n_ax = eigvecs[:, 0].copy()
    if ref_normal is not None and float(np.dot(n_ax, ref_normal)) < 0.0:
        n_ax = -n_ax
    u_ax = eigvecs[:, 2].copy()
    v_ax = np.cross(n_ax, u_ax)
    v_ax /= max(float(np.linalg.norm(v_ax)), 1.0e-12)

    rel = nbrs - p0[None, :]
    x = rel @ u_ax
    y = rel @ v_ax
    z = rel @ n_ax

    if bandwidth is not None and bandwidth > 0.0:
        d2 = np.sum(rel * rel, axis=1)
        w = np.exp(-d2 / (2.0 * bandwidth * bandwidth))
    else:
        w = np.ones(len(nbrs), dtype=np.float64)

    # Vandermonde design matrix for Eq. (6): [1, x, y, x^2, x*y, y^2]
    sqrt_w = np.sqrt(np.maximum(w, 1.0e-12))
    design = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y]) * sqrt_w[:, None]
    rhs = z * sqrt_w

    ata = design.T @ design + float(reg_eps) * np.eye(6, dtype=np.float64)
    coeffs = np.linalg.solve(ata, design.T @ rhs)
    return LocalMongePatch(origin=p0, u_axis=u_ax, v_axis=v_ax, n_axis=n_ax, coeffs=coeffs)


def mls_monge_project_point_cloud(
    pc: o3d.geometry.PointCloud,
    iterations: int,
    k_neighbors: int,
    alpha: float,
    outlier_sigma: float,
    radius: float = 0.0,
    bandwidth: float = 0.0,
    reg_eps: float = 1.0e-6,
    recompute_normals: bool = True,
    chunk_size: int = 25000,
) -> o3d.geometry.PointCloud:
    """Vectorized Moving Least Squares (MLS) surface projection & differential normal solver (Eq. 6).

    For every point in batches, builds its local PCA tangent frame, solves the weighted 6x6 quadratic
    Monge system z = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2, projects P <- P + alpha * c0 * n0,
    and extracts smooth analytical normals [-c1, -c2, 1] / sqrt(1 + c1^2 + c2^2).
    """
    pts = np.asarray(pc.points, dtype=np.float64).copy()
    n_pts = len(pts)
    k_eff = min(max(int(k_neighbors), 6), n_pts - 1)
    if iterations <= 0 or n_pts < 10 or k_eff < 6:
        return pc

    has_norms = pc.has_normals() and len(pc.normals) == n_pts
    norms = np.asarray(pc.normals, dtype=np.float64).copy() if has_norms else None
    has_cols = pc.has_colors() and len(pc.colors) == n_pts
    cols = np.asarray(pc.colors, dtype=np.float64).copy() if has_cols else None

    a_step = float(np.clip(alpha, 0.0, 1.0))
    eye6 = float(reg_eps) * np.eye(6, dtype=np.float64)[None, :, :]
    last_c0 = np.zeros(n_pts, dtype=np.float64)

    for _ in range(int(iterations)):
        tree = cKDTree(pts)
        next_pts = np.empty_like(pts)
        next_norms = np.empty_like(pts)

        for start in range(0, n_pts, int(chunk_size)):
            end = min(start + int(chunk_size), n_pts)
            p_batch = pts[start:end]  # (B, 3)
            dists, idx = tree.query(p_batch, k=k_eff, workers=-1)
            nbrs = pts[idx]  # (B, k, 3)

            # Local PCA frame per point
            cent = nbrs.mean(axis=1, keepdims=True)
            diff_c = nbrs - cent
            cov = np.einsum("bki,bkj->bij", diff_c, diff_c) / max(k_eff - 1, 1)
            _, eigvecs = np.linalg.eigh(cov)  # (B, 3, 3)
            n0 = eigvecs[:, :, 0]             # (B, 3) smallest eigenvalue normal
            u0 = eigvecs[:, :, 2]             # (B, 3) major tangent
            if norms is not None:
                flip = np.sum(n0 * norms[start:end], axis=1) < 0.0
                n0 = np.where(flip[:, None], -n0, n0)
            v0 = np.cross(n0, u0)
            v0 /= np.maximum(np.linalg.norm(v0, axis=1, keepdims=True), 1.0e-12)

            rel = nbrs - p_batch[:, None, :]  # (B, k, 3)
            x = np.einsum("bki,bi->bk", rel, u0)
            y = np.einsum("bki,bi->bk", rel, v0)
            z = np.einsum("bki,bi->bk", rel, n0)

            # Spatial weights (optional radius cutoff R and Gaussian bandwidth h)
            h = float(bandwidth) if bandwidth > 0.0 else np.maximum(dists[:, -1], 1.0e-6)
            h_col = h[:, None] if isinstance(h, np.ndarray) else h
            w = np.exp(-(dists * dists) / np.maximum(2.0 * h_col * h_col, 1.0e-18))
            if radius > 0.0:
                in_rad = dists <= float(radius)
                # Ensure at least 6 neighbors remain active per point
                enough = np.sum(in_rad, axis=1) >= 6
                w = np.where(enough[:, None] & (~in_rad), 0.0, w)
            sw = np.sqrt(np.maximum(w, 1.0e-12))

            # Batched 6-parameter quadratic Monge design matrix (Eq. 6)
            X = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=2) * sw[:, :, None]
            rhs = np.einsum("bki,bk->bi", X, z * sw)
            ata = np.einsum("bki,bkj->bij", X, X) + eye6
            coeffs = np.linalg.solve(ata, rhs[..., None])[..., 0]  # (B, 6)

            c0 = coeffs[:, 0]
            c1 = coeffs[:, 1]
            c2 = coeffs[:, 2]
            last_c0[start:end] = c0
            next_pts[start:end] = p_batch + (a_step * c0)[:, None] * n0

            # Analytical Monge normal: (-c1*u0 - c2*v0 + n0) / sqrt(1 + c1^2 + c2^2)
            n_mls = -c1[:, None] * u0 - c2[:, None] * v0 + n0
            n_mls /= np.maximum(np.linalg.norm(n_mls, axis=1, keepdims=True), 1.0e-12)
            if norms is not None:
                flip2 = np.sum(n_mls * norms[start:end], axis=1) < 0.0
                n_mls = np.where(flip2[:, None], -n_mls, n_mls)
            next_norms[start:end] = n_mls

        pts = next_pts
        if recompute_normals or norms is not None:
            norms = next_norms

    # MAD outlier rejection on Monge residual |c0|
    keep = np.ones(n_pts, dtype=bool)
    if outlier_sigma > 0.0:
        abs_c0 = np.abs(last_c0)
        med = float(np.median(abs_c0))
        mad = float(np.median(np.abs(abs_c0 - med)))
        thr = med + float(outlier_sigma) * max(1.4826 * mad, 1.0e-9)
        cand_keep = abs_c0 <= thr
        if int(np.sum(cand_keep)) >= max(50, int(0.5 * n_pts)):
            keep = cand_keep

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[keep])
    if norms is not None:
        out.normals = o3d.utility.Vector3dVector(norms[keep])
    if cols is not None:
        out.colors = o3d.utility.Vector3dVector(cols[keep])
    return out


# =============================================================================
# SECTION 2.4: UNIVERSAL REPRESENTATIONS — PARAMETRIC (Eq. 7-8) & SDF (Eq. 9)
# =============================================================================

@dataclasses.dataclass
class ParametricSurface:
    """Universal Parametric Vector Surface r(u, v) = [x(u,v), y(u,v), z(u,v)]^T (Section 2.4, Eq. 7)."""

    evaluator: Callable[[np.ndarray, np.ndarray], np.ndarray]
    u_bounds: tuple[float, float]
    v_bounds: tuple[float, float]
    periodic_v: bool = False

    def evaluate(self, u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray:
        """Evaluate vector mapping r(u, v) in R^3 (Eq. 7)."""
        ua = np.asarray(u, dtype=np.float64)
        va = np.asarray(v, dtype=np.float64)
        ua, va = np.broadcast_arrays(ua, va)
        return self.evaluator(ua, va)

    def sample_mesh(self, nu: int = 32, nv: int = 128) -> o3d.geometry.TriangleMesh:
        """Tessellate parametric domain D = [u0, u1] x [v0, v1] into a TriangleMesh."""
        u_vals = np.linspace(self.u_bounds[0], self.u_bounds[1], int(nu), dtype=np.float64)
        v_vals = np.linspace(self.v_bounds[0], self.v_bounds[1], int(nv), dtype=np.float64)
        uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")
        verts = self.evaluate(uu.ravel(), vv.ravel()).reshape(-1, 3)

        tris: list[list[int]] = []
        for i in range(int(nu) - 1):
            for j in range(int(nv) - 1):
                v00 = i * int(nv) + j
                v01 = v00 + 1
                v10 = (i + 1) * int(nv) + j
                v11 = v10 + 1
                tris.append([v00, v10, v11])
                tris.append([v00, v11, v01])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        return mesh


def mobius_strip_surface(major_radius: float, half_width: float) -> ParametricSurface:
    """Exact Parametric Möbius Strip (180-deg non-orientable single-sided surface, Eq. 8):

        r(u, v) = [ (R + u*cos(v/2))*cos(v),
                    (R + u*cos(v/2))*sin(v),
                    u*sin(v/2) ]^T,     u in [-w, w], v in [0, 2*pi]
    """
    if major_radius <= abs(half_width) or half_width <= 0.0:
        raise ValueError("Require major_radius R > half_width w > 0 to avoid self-intersection.")

    def _eval(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        half_v = 0.5 * v
        radial = float(major_radius) + u * np.cos(half_v)
        x = radial * np.cos(v)
        y = radial * np.sin(v)
        z = u * np.sin(half_v)
        return np.stack([x, y, z], axis=-1)

    return ParametricSurface(
        evaluator=_eval,
        u_bounds=(-float(half_width), float(half_width)),
        v_bounds=(0.0, 2.0 * np.pi),
        periodic_v=True,
    )


@dataclasses.dataclass
class SignedDistanceField:
    """Universal Implicit Signed Distance Field f_theta(x, y, z) = s (Section 2.4, Eq. 9):
      - s = 0 : exact zero-level surface
      - s < 0 : interior of solid object
      - s > 0 : exterior free space
    """

    field_fn: Callable[[np.ndarray], np.ndarray]
    zero_tol: float = 1.0e-4
    grad_eps: float = 1.0e-5

    def __call__(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return np.asarray(self.field_fn(pts), dtype=np.float64).reshape(len(pts))

    def classify_points(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """Partition points according to Eq. (9) sign rules (surface s=0, interior s<0, exterior s>0)."""
        s = self(points)
        on_surface = np.abs(s) <= self.zero_tol
        interior = s < -self.zero_tol
        exterior = s > self.zero_tol
        return {
            "sdf": s,
            "surface_mask": on_surface,
            "interior_mask": interior,
            "exterior_mask": exterior,
        }

    def gradient(self, points: np.ndarray) -> np.ndarray:
        """Compute central-difference spatial gradient nabla f_theta(x, y, z) (Eikonal normal field)."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        h = float(self.grad_eps)
        grad = np.empty_like(pts)
        for ax in range(3):
            shift = np.zeros(3, dtype=np.float64)
            shift[ax] = h
            grad[:, ax] = (self(pts + shift) - self(pts - shift)) / (2.0 * h)
        return grad

    def extract_zero_isosurface(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        resolution: int = 48,
    ) -> o3d.geometry.TriangleMesh:
        """Extract watertight zero-level set {x : f_theta(x) = 0} via Marching Cubes."""
        bmin = np.asarray(bbox_min, dtype=np.float64).reshape(3)
        bmax = np.asarray(bbox_max, dtype=np.float64).reshape(3)
        res = max(int(resolution), 8)
        xs = np.linspace(bmin[0], bmax[0], res)
        ys = np.linspace(bmin[1], bmax[1], res)
        zs = np.linspace(bmin[2], bmax[2], res)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        grid_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        vol = self(grid_pts).reshape(res, res, res)

        # Trimesh marching cubes extracts level=0; shift & scale from voxel grid to world coordinates
        tm = trimesh.voxel.ops.matrix_to_marching_cubes(vol < 0.0, pitch=1.0)
        scale = (bmax - bmin) / float(res - 1)
        verts = np.asarray(tm.vertices, dtype=np.float64) * scale[None, :] + bmin[None, :]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32))
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        return mesh
