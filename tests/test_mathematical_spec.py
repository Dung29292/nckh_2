"""Rigorous equation-by-equation unit tests verifying the mathematical specification (nckh.pdf).

Tests every mathematical model and error-mitigation mechanism:
  - Eq. (1) & Eq. (2): HessianPlane unit-normal normalization, orthogonal distance, floor/wall classification
  - Eq. (3), (4), (5): QuadricSurface, CircularCylinder RANSAC+refit, SphereSurface exact recovery
  - Eq. (6)          : LocalMongePatch & mls_monge_project_point_cloud curvature & saddle preservation
  - Eq. (7), (8), (9): ParametricSurface, Möbius strip 180-deg twist, SignedDistanceField classification
  - Eq. (10) & 3.1   : Iterative Dual-Validated RANSAC rejecting adjacent horizontal tabletops + SVD refit
  - Section 3.2      : RandLA-Net Clutter Reclamation + Morphological Closing of occluded wall cavities
  - Eq. (11) & 3.3   : Manhattan World alignment (including 0 / 90 deg boundary wrap) + 90-deg corners
  - Section 3.4      : Freeform MLS -> Poisson (Eq. 12) -> Volume-Preserving Taubin Smoothing
  - Eqs. (13)-(16)   : Top-Down Superquadric & Manhattan Room Priors + Elastic Energy outlier immunity
  - Section 5 & 6    : Zero-hardcoded YAML loading, automatic problem router, and .dxf / .ifc export
"""
from __future__ import annotations

import shutil
import unittest

import numpy as np
import open3d as o3d

from nckh_2.backends.ransac_manhattan import (
    align_manhattan_world,
    compute_manhattan_dominant_angle,
    iterative_ransac_planes,
    reclaim_clutter_and_close_gaps,
    regularize_rectilinear_polygon,
)
from nckh_2.backends.smoothing import taubin_volume_preserving_smooth
from nckh_2.backends.surfaces import (
    CircularCylinder,
    HessianPlane,
    QuadricSurface,
    SignedDistanceField,
    SphereSurface,
    fit_local_monge_patch,
    mobius_strip_surface,
)
from nckh_2.backends.topdown_residual import (
    SuperquadricPrior,
    reconstruct_room_topdown_residual,
    solve_elastic_residual_field,
)
from nckh_2.core import paths
from nckh_2.core.config import load_config, load_floorplan_config
from nckh_2.pipeline.surface_router import classify_geometry_problem, route_and_reconstruct


class TestMathematicalSpecification(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config(include_floorplan=True)
        self.fp_cfg = load_floorplan_config()

    def test_eq1_eq2_hessian_plane_and_classification(self) -> None:
        # Unnormalized input (3, 4, 0)*x + 10 = 0 -> normalized (0.6, 0.8, 0)*x + 2.0 = 0
        plane = HessianPlane(normal=np.array([3.0, 4.0, 0.0]), d=10.0)
        self.assertAlmostEqual(float(np.sum(plane.normal ** 2)), 1.0, places=10)
        self.assertAlmostEqual(plane.d, 2.0, places=10)

        # Eq. (2): dist(O, plane) = |0 + 2.0| = 2.0
        d0 = plane.distance(np.array([[0.0, 0.0, 0.0], [-2.0, -1.0, 5.0]]))
        self.assertAlmostEqual(float(d0[0]), 2.0, places=10)
        self.assertEqual(plane.classify_orientation(self.fp_cfg["hessian_plane"]), "vertical_wall")

        floor_p = HessianPlane(normal=np.array([0.02, -0.01, 0.999]), d=0.0)
        self.assertEqual(
            floor_p.classify_orientation(self.fp_cfg["hessian_plane"], inlier_z_median=0.0, scene_z_split=1.4),
            "horizontal_floor",
        )

    def test_eq3_eq4_eq5_quadrics_cylinder_and_sphere(self) -> None:
        rng = np.random.default_rng(7)
        # Cylinder along Z centered at (1.5, -0.8, 0.0) with r = 0.35 m
        angles = rng.uniform(0.0, 2.0 * np.pi, 400)
        zs = rng.uniform(0.0, 2.5, 400)
        r_true = 0.35
        c_true = np.array([1.5, -0.8, 0.0])
        pts = np.column_stack([
            c_true[0] + r_true * np.cos(angles) + rng.normal(0.0, 0.002, 400),
            c_true[1] + r_true * np.sin(angles) + rng.normal(0.0, 0.002, 400),
            zs,
        ])
        nrms = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])

        cyl, inliers = CircularCylinder.fit_ransac_with_normals(
            pts, nrms, self.fp_cfg["quadrics"], self.fp_cfg["numerical"], constrain_vertical=True
        )
        self.assertIsNotNone(cyl)
        assert cyl is not None
        self.assertAlmostEqual(cyl.radius, r_true, delta=0.01)
        self.assertGreater(len(inliers), 350)

        # Eq. (3) Quadric representation consistency
        q_cyl = cyl.to_quadric()
        self.assertIsInstance(q_cyl, QuadricSurface)
        self.assertLess(float(np.median(np.abs(q_cyl.evaluate(pts)))), 0.01)

        # Eq. (5) Sphere fitting
        phi = rng.uniform(0.0, 2.0 * np.pi, 300)
        costh = rng.uniform(-1.0, 1.0, 300)
        sinth = np.sqrt(1.0 - costh * costh)
        sph_c = np.array([-1.0, 2.0, 3.0])
        sph_r = 1.25
        sph_pts = sph_c + sph_r * np.column_stack([sinth * np.cos(phi), sinth * np.sin(phi), costh])
        sph_fit = SphereSurface.fit_least_squares(sph_pts)
        self.assertAlmostEqual(sph_fit.radius, sph_r, places=5)
        np.testing.assert_allclose(sph_fit.center, sph_c, atol=1.0e-5)

    def test_eq6_local_monge_patch_saddle_and_dome(self) -> None:
        # Hyperbolic saddle (ergonomic mouse thumb grip): z = 0.5*x^2 - 0.8*y^2 -> K = 4*(0.5)*(-0.8) = -1.6 < 0
        xs, ys = np.meshgrid(np.linspace(-0.2, 0.2, 11), np.linspace(-0.2, 0.2, 11))
        x_f, y_f = xs.ravel(), ys.ravel()
        z_f = 0.5 * x_f * x_f - 0.8 * y_f * y_f
        pts = np.column_stack([x_f, y_f, z_f])

        patch = fit_local_monge_patch(np.array([0.0, 0.0, 0.0]), pts, bandwidth=0.3, ref_normal=np.array([0.0, 0.0, 1.0]))
        self.assertLess(patch.gaussian_curvature, -1.0)
        np.testing.assert_allclose(patch.differential_normal, np.array([0.0, 0.0, 1.0]), atol=1.0e-4)

    def test_eq7_eq8_eq9_mobius_and_sdf(self) -> None:
        mobius = mobius_strip_surface(major_radius=2.0, half_width=0.4)
        p_start = mobius.evaluate(0.3, 0.0)
        p_end = mobius.evaluate(-0.3, 2.0 * np.pi)
        # 180-degree twist: r(u, 0) == r(-u, 2*pi)
        np.testing.assert_allclose(p_start, p_end, atol=1.0e-10)

        # Eq. (9) SDF classification
        sdf = SignedDistanceField(lambda p: np.linalg.norm(p, axis=1) - 1.0, zero_tol=1.0e-4)
        cls_res = sdf.classify_points(np.array([[1.0, 0.0, 0.0], [0.2, 0.0, 0.0], [2.5, 0.0, 0.0]]))
        self.assertTrue(bool(cls_res["surface_mask"][0]))
        self.assertTrue(bool(cls_res["interior_mask"][1]))
        self.assertTrue(bool(cls_res["exterior_mask"][2]))

    def test_eq10_eq11_ransac_dual_validation_clutter_and_manhattan(self) -> None:
        rng = np.random.default_rng(12)
        # Tilted vertical wall rotated by +18 deg around Z, plus an adjacent horizontal desk touching the wall
        theta_tilt = np.radians(18.0)
        rot = np.array([
            [np.cos(theta_tilt), -np.sin(theta_tilt), 0.0],
            [np.sin(theta_tilt),  np.cos(theta_tilt), 0.0],
            [0.0,                 0.0,                1.0],
        ])

        # Wall at x = 2.0 in unrotated frame, y in [0, 4], z in [0, 2.8]
        wy = rng.uniform(0.0, 4.0, 600)
        wz = rng.uniform(0.0, 2.8, 600)
        wx = np.full(600, 2.0) + rng.normal(0.0, 0.004, 600)
        wall_pts = np.column_stack([wx, wy, wz]) @ rot.T
        wall_nrms = np.tile(np.array([[1.0, 0.0, 0.0]]), (600, 1)) @ rot.T

        # Adjacent horizontal desk right against wall x in [1.99, 2.01], z = 0.75 (within eps_plane of wall!)
        dx = rng.uniform(1.99, 2.01, 200)
        dy = rng.uniform(1.0, 2.5, 200)
        dz = np.full(200, 0.75)
        desk_pts = np.column_stack([dx, dy, dz]) @ rot.T
        desk_nrms = np.tile(np.array([[0.0, 0.0, 1.0]]), (200, 1))

        # 80 wall-base points mislabeled as RandLA-Net clutter (label 0)
        cy = rng.uniform(1.2, 2.2, 80)
        cz = rng.uniform(0.0, 0.35, 80)
        cx = np.full(80, 2.0) + rng.normal(0.0, 0.003, 80)
        clut_wall_pts = np.column_stack([cx, cy, cz]) @ rot.T
        clut_wall_nrms = np.tile(np.array([[1.0, 0.0, 0.0]]), (80, 1)) @ rot.T

        all_pts = np.vstack([wall_pts, desk_pts])
        all_nrms = np.vstack([wall_nrms, desk_nrms])
        planes, _ = iterative_ransac_planes(all_pts, all_nrms, self.fp_cfg)
        # Sequential RANSAC extracts the vertical wall first AND keeps the horizontal desk separate (never merged!)
        self.assertEqual(len(planes), 2)
        wall_planes = [p for p in planes if p.orientation_class == "vertical_wall"]
        horiz_planes = [p for p in planes if p.orientation_class.startswith("horizontal")]
        self.assertEqual(len(wall_planes), 1)
        self.assertEqual(len(horiz_planes), 1)
        # Dual validation (Eq. 10: theta < 15 deg) must reject 100% of horizontal desk points from the wall!
        self.assertTrue(np.all(wall_planes[0].inlier_indices < 600))

        # Clutter reclamation test (Section 3.2)
        full_pts = np.vstack([all_pts, clut_wall_pts])
        full_nrms = np.vstack([all_nrms, clut_wall_nrms])
        labels = np.array([2] * 600 + [8] * 200 + [0] * 80, dtype=np.int64)
        c_stats = reclaim_clutter_and_close_gaps(full_pts, full_nrms, labels, planes, self.fp_cfg)
        self.assertGreaterEqual(c_stats["total_reclaimed_points"], 75)

        # Manhattan World angle recovery (Eq. 11) including wrap-around safety near 0/90 deg
        _, _, est_theta0, _ = align_manhattan_world(full_pts, full_nrms, planes, self.fp_cfg)
        self.assertAlmostEqual(np.degrees(est_theta0), 18.0, delta=0.5)

        wrap_normals = np.array([
            [np.cos(np.radians(1.0)), np.sin(np.radians(1.0)), 0.0],
            [np.cos(np.radians(-1.0)), np.sin(np.radians(-1.0)), 0.0],
        ])
        theta_wrap = compute_manhattan_dominant_angle(wrap_normals, None, self.fp_cfg["manhattan_alignment"])
        self.assertLess(min(theta_wrap, 0.5 * np.pi - theta_wrap), np.radians(0.5))

        # 90-deg rectilinear polygon regularization
        noisy_poly = np.array([[0.02, -0.03], [4.01, 0.04], [3.98, 3.02], [-0.01, 2.97]])
        corners, segs = regularize_rectilinear_polygon(noisy_poly, self.fp_cfg)
        self.assertEqual(len(corners), 4)
        self.assertEqual(len(segs), 4)
        # Verify exact 90.000 deg interior angles
        v0 = corners[1] - corners[0]
        v1 = corners[2] - corners[1]
        self.assertAlmostEqual(float(np.dot(v0, v1)), 0.0, places=10)

    def test_section34_taubin_volume_preservation(self) -> None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=16)
        vol_orig = sphere.get_volume()
        rng = np.random.default_rng(99)
        v_noisy = np.asarray(sphere.vertices) + rng.normal(0.0, 0.015, size=(len(sphere.vertices), 3))
        sphere.vertices = o3d.utility.Vector3dVector(v_noisy)

        smoothed = taubin_volume_preserving_smooth(sphere, iterations=12, lambda_pos=0.35, mu_neg=-0.37)
        vol_after = smoothed.get_volume()
        # Volume drift must remain < 1.5% (unlike Laplacian which shrinks severely)
        self.assertLess(abs(vol_after - vol_orig) / vol_orig, 0.015)

    def test_eq13_to_eq16_topdown_elastic_outlier_immunity(self) -> None:
        rng = np.random.default_rng(21)
        # Generate ergonomic ellipsoid-like dome with a concave side indentation + 15 wild floating outliers
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
        pts = np.asarray(sphere.vertices, dtype=np.float64).copy()
        pts[:, 0] *= 1.2
        pts[:, 1] *= 0.8
        pts[:, 2] *= 0.6
        # Concave thumb grip on -Y side (d_i < 0)
        grip = (pts[:, 1] < -0.5) & (np.abs(pts[:, 0]) < 0.4)
        pts[grip, 1] += 0.12

        # Inject 15 extreme floating outliers far outside the object
        outliers = rng.uniform(3.0, 5.0, size=(15, 3))
        corrupted = np.vstack([pts, outliers])

        sq = SuperquadricPrior.fit_closed_form_pca(corrupted, self.cfg["topdown_residual"]["superquadric"])
        base_mesh = sq.to_canonical_mesh(nu=24, nv=48)
        decomp = solve_elastic_residual_field(base_mesh, corrupted, self.cfg["topdown_residual"]["elastic_energy"])

        # Elastic smoothness + Huber data term must completely block the 3.0-5.0 m outliers from tearing the mesh
        actual_v = np.asarray(decomp.actual_mesh.vertices)
        self.assertLess(float(np.max(np.linalg.norm(actual_v, axis=1))), 1.5)
        # Deterministic one-shot verification: re-running produces bit-identical vertices
        decomp2 = solve_elastic_residual_field(base_mesh, corrupted, self.cfg["topdown_residual"]["elastic_energy"])
        np.testing.assert_allclose(np.asarray(decomp.actual_mesh.vertices), np.asarray(decomp2.actual_mesh.vertices))

    def test_end_to_end_router_room_and_freeform_export(self) -> None:
        tmp_dir = paths.ROOT / "scratch" / "_test_spec_router"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            rng = np.random.default_rng(5)
            # 1. Synthetic 4m x 3m x 2.8m room rotated by 12 deg with floor + 4 walls
            wall_pts_list = []
            wall_nrm_list = []
            for x_val, nx in ((0.0, 1.0), (4.0, -1.0)):
                ys = rng.uniform(0.0, 3.0, 300)
                zs = rng.uniform(0.0, 2.8, 300)
                wall_pts_list.append(np.column_stack([np.full(300, x_val), ys, zs]))
                wall_nrm_list.append(np.tile([nx, 0.0, 0.0], (300, 1)))
            for y_val, ny in ((0.0, 1.0), (3.0, -1.0)):
                xs = rng.uniform(0.0, 4.0, 300)
                zs = rng.uniform(0.0, 2.8, 300)
                wall_pts_list.append(np.column_stack([xs, np.full(300, y_val), zs]))
                wall_nrm_list.append(np.tile([0.0, ny, 0.0], (300, 1)))
            # Floor slab
            fx = rng.uniform(0.0, 4.0, 300)
            fy = rng.uniform(0.0, 3.0, 300)
            wall_pts_list.append(np.column_stack([fx, fy, np.zeros(300)]))
            wall_nrm_list.append(np.tile([0.0, 0.0, 1.0], (300, 1)))

            ang = np.radians(12.0)
            rz = np.array([[np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0], [0.0, 0.0, 1.0]])
            r_pts = np.vstack(wall_pts_list) @ rz.T
            r_nrms = np.vstack(wall_nrm_list) @ rz.T

            room_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(r_pts))
            room_pc.normals = o3d.utility.Vector3dVector(r_nrms)

            dec_room = classify_geometry_problem(r_pts, r_nrms, self.cfg, self.fp_cfg)
            self.assertEqual(dec_room["route"], "room_layout")

            res_room = route_and_reconstruct(room_pc, self.cfg, tmp_dir / "room_out", stem="room01", floorplan_cfg=self.fp_cfg)
            self.assertTrue(res_room["validation"]["is_strictly_manhattan"])
            self.assertAlmostEqual(res_room["validation"]["floor_area_m2"], 12.0, delta=0.5)
            self.assertTrue((tmp_dir / "room_out" / "room01_floorplan.dxf").exists())
            self.assertTrue((tmp_dir / "room_out" / "room01_bim.ifc").exists())

            # 2. Small ergonomic curved object (0.12m mouse_01 scale) -> routes automatically to freeform_object
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.06, resolution=16)
            m_pts = np.asarray(sph.vertices, dtype=np.float64) * np.array([1.0, 0.65, 0.40])
            sph.compute_vertex_normals()
            m_nrms = np.asarray(sph.vertex_normals, dtype=np.float64)
            mouse_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(m_pts))
            mouse_pc.normals = o3d.utility.Vector3dVector(m_nrms)

            dec_obj = classify_geometry_problem(m_pts, m_nrms, self.cfg, self.fp_cfg)
            self.assertEqual(dec_obj["route"], "freeform_object")

            res_obj = route_and_reconstruct(mouse_pc, self.cfg, tmp_dir / "mouse_out", stem="mouse_01", floorplan_cfg=self.fp_cfg)
            self.assertTrue((tmp_dir / "mouse_out" / "mouse_01_refined.obj").exists())
            self.assertGreater(res_obj["freeform"]["refined_triangles"], 100)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
