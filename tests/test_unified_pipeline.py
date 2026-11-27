"""Comprehensive verification tests for the unified self-optimizing single-object pipeline.

Verifies:
  1. Scale invariance of data-derived adaptive parameters (10^-2x, 1x, 10^2x).
  2. Spherical trajectory diagnostics: smooth continuous spiral (0% false pruning) vs
     abrupt SfM radial outliers (>= 90% pruned).
  3. Multi-elevation dome scanning (0 -> 45 -> 90 deg): upright axis (< 2 deg skew).
  4. Branch-cut-free partial azimuth coverage (210 deg arc crossing +-180 deg).
  5. Unified region-wise BIC model selection on hollow box, tall cylinder with handle,
     and soft shape_hint rejection when data contradicts prompt.
  6. Sequential-ring pair generation, largest-gap component filtering, and VRAM budgeting.
"""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import unittest

import numpy as np
import open3d as o3d

from nckh_2.backends.adaptive import (
    compute_adaptive_point_stats,
    estimate_vram_mvs_max_image_size,
    largest_gap_component_mask,
)
from nckh_2.backends.colmap_io import (
    CameraPoseRecord,
    analyze_spherical_trajectory,
    estimate_elevation_invariant_up,
)
from nckh_2.backends.region_models import (
    fill_analytic_region_voids,
    segment_and_select_regions,
)
from nckh_2.core import paths
from nckh_2.core.config import load_config
from nckh_2.pipeline.mesh_export import _establish_canonical_frame
from nckh_2.pipeline.reconstruct import build_sequential_ring_pairs


def _make_pose(idx: int, center: np.ndarray, target: np.ndarray) -> CameraPoseRecord:
    view = target - center
    view /= max(float(np.linalg.norm(view)), 1.0e-12)
    return CameraPoseRecord(
        name=f"frame_{idx:05d}.jpg",
        camera_id=1,
        R=np.eye(3, dtype=np.float64),
        t=-center,
        center=center,
        view_dir=view,
        num_inliers_3d=200,
    )


class TestUnifiedSelfOptimizingPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()
        self.cfg_adapt = self.cfg["adaptive"]
        self.cfg_diag = self.cfg["reconstruction"]["pose_diagnostics"]
        self.num_cfg = self.cfg["numerical"]

    def test_scale_invariance_of_adaptive_stats(self):
        rng = np.random.default_rng(42)
        box = o3d.geometry.TriangleMesh.create_box(width=1.6, height=1.0, depth=0.8)
        pts_base = np.asarray(box.sample_points_uniformly(4000).points, dtype=np.float64)
        pts_base += rng.normal(0.0, 0.002, size=pts_base.shape)

        s1 = compute_adaptive_point_stats(pts_base, self.cfg_adapt, norm_eps=float(self.num_cfg["norm_eps"]))
        s_small = compute_adaptive_point_stats(pts_base * 1.0e-2, self.cfg_adapt, norm_eps=float(self.num_cfg["norm_eps"]))
        s_large = compute_adaptive_point_stats(pts_base * 1.0e2, self.cfg_adapt, norm_eps=float(self.num_cfg["norm_eps"]))

        self.assertAlmostEqual(s_small.point_spacing / s1.point_spacing, 1.0e-2, delta=1.0e-4)
        self.assertAlmostEqual(s_large.point_spacing / s1.point_spacing, 1.0e2, delta=1.0)
        self.assertAlmostEqual(s_small.inlier_tol / s1.inlier_tol, 1.0e-2, delta=5.0e-4)
        self.assertEqual(s_small.poisson_depth, s1.poisson_depth)
        self.assertEqual(s_large.poisson_depth, s1.poisson_depth)
        self.assertEqual(s_small.k_neighbors, s1.k_neighbors)
        self.assertEqual(s_large.k_neighbors, s1.k_neighbors)

    def test_smooth_continuous_spiral_vs_abrupt_radial_outliers(self):
        target = np.zeros(3, dtype=np.float64)
        n = 80
        angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        radii_smooth = np.linspace(1.0, 0.4, n)  # 2.5x intentional continuous spiral-in
        poses_smooth = []
        for i, (ang, r) in enumerate(zip(angles, radii_smooth), 1):
            c = np.array([r * math.cos(ang), r * math.sin(ang), 0.35 * r], dtype=np.float64)
            poses_smooth.append(_make_pose(i, c, target))

        diag_smooth = analyze_spherical_trajectory(poses_smooth, self.cfg_diag, self.cfg_adapt)
        self.assertEqual(diag_smooth.radius_profile, "smooth_variable")
        self.assertEqual(len(diag_smooth.pruned_cameras), 0)
        self.assertGreater(diag_smooth.theta_cov_deg, 345.0)

        # Inject 10% abrupt 2.2x radial outlier spikes into a circular orbit
        poses_noisy = []
        outlier_ids = {8, 16, 24, 32, 40, 48, 56, 64}
        for i, ang in enumerate(angles, 1):
            r = 2.2 if i in outlier_ids else 1.0
            c = np.array([r * math.cos(ang), r * math.sin(ang), 0.3], dtype=np.float64)
            poses_noisy.append(_make_pose(i, c, target))

        diag_noisy = analyze_spherical_trajectory(poses_noisy, self.cfg_diag, self.cfg_adapt)
        self.assertEqual(diag_noisy.radius_profile, "unstable_outliers")
        caught = sum(1 for i in outlier_ids if f"frame_{i:05d}.jpg" in diag_noisy.pruned_cameras)
        self.assertGreaterEqual(caught / len(outlier_ids), 0.90)
        self.assertLess(diag_noisy.d_var_after, 0.25)

    def test_multi_elevation_dome_0_45_90_degrees(self):
        target = np.zeros(3, dtype=np.float64)
        poses = []
        idx = 1
        # 3 tiers: horizontal (10 deg), oblique (45 deg), top-down (85 deg) at constant 3D distance R=1.0
        for elev_deg, count in ((10.0, 36), (45.0, 24), (85.0, 8)):
            el = math.radians(elev_deg)
            for ang in np.linspace(0.0, 2.0 * math.pi, count, endpoint=False):
                c = np.array([
                    math.cos(el) * math.cos(ang),
                    math.cos(el) * math.sin(ang),
                    math.sin(el),
                ], dtype=np.float64)
                poses.append(_make_pose(idx, c, target))
                idx += 1

        diag = analyze_spherical_trajectory(poses, self.cfg_diag, self.cfg_adapt)
        self.assertTrue(diag.is_multi_elevation)
        skew_deg = math.degrees(math.acos(min(abs(float(np.dot(diag.up_axis, [0.0, 0.0, 1.0]))), 1.0)))
        self.assertLess(skew_deg, 2.0)
        self.assertGreater(diag.elevation_span_deg, 70.0)
        self.assertEqual(len(diag.pruned_cameras), 0)
        self.assertGreater(diag.theta_cov_deg, 340.0)

    def test_partial_210_deg_arc_crossing_branch_cut(self):
        target = np.zeros(3, dtype=np.float64)
        # 210 deg arc from +75 deg to +285 deg (crosses +-180 deg branch cut!)
        angles_deg = np.linspace(75.0, 285.0, 45)
        poses = []
        for i, ad in enumerate(angles_deg, 1):
            a = math.radians(ad)
            c = np.array([math.cos(a), math.sin(a), 0.4], dtype=np.float64)
            poses.append(_make_pose(i, c, target))

        diag = analyze_spherical_trajectory(poses, self.cfg_diag, self.cfg_adapt)
        self.assertAlmostEqual(diag.theta_cov_deg, 210.0, delta=3.0)
        self.assertTrue(diag.has_significant_gap)

    def test_unified_bic_hollow_box_and_prior_rejection(self):
        rng = np.random.default_rng(7)
        box = o3d.geometry.TriangleMesh.create_box(width=1.4, height=1.0, depth=0.7)
        pcd = box.sample_points_uniformly(6000)
        pts = np.asarray(pcd.points, dtype=np.float64)
        pts += rng.normal(0.0, 0.001, size=pts.shape)

        # Carve a 60% hollow void in the center of the y=0 vertical wall (mimicking box_01 failure)
        on_front_wall = np.abs(pts[:, 1]) < 0.02
        in_core = (pts[:, 0] > 0.25) & (pts[:, 0] < 1.15) & (pts[:, 2] > 0.12) & (pts[:, 2] < 0.58)
        keep = ~(on_front_wall & in_core)
        pts_hollow = pts[keep]

        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_hollow))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
        pc.orient_normals_consistent_tangent_plane(15)
        # Ensure outward normals relative to box center
        nrms = np.asarray(pc.normals, dtype=np.float64)
        cent = np.array([0.7, 0.5, 0.35], dtype=np.float64)
        flip = np.einsum("ij,ij->i", nrms, pts_hollow - cent[None, :]) < 0.0
        nrms[flip] *= -1.0

        stats = compute_adaptive_point_stats(pts_hollow, self.cfg_adapt, norm_eps=float(self.num_cfg["norm_eps"]))
        # Deliberately pass a WRONG shape_hint="cylinder" to verify data overrides prompt!
        cfg_wrong_hint = dict(self.cfg_adapt, shape_hint="cylinder")
        regions = segment_and_select_regions(pts_hollow, nrms, stats, cfg_wrong_hint, self.num_cfg)

        planar_regs = [r for r in regions if r.model_type == "plane"]
        self.assertGreaterEqual(len(planar_regs), 4)
        self.assertTrue(any(r.prior_rejected for r in planar_regs))

        # Verify universal analytic void filler closes the hollow front wall
        aug_pts, _, _, conf, info = fill_analytic_region_voids(
            pts_hollow, nrms, None, regions, stats, visual_hull=None, cfg_adapt=self.cfg_adapt
        )
        self.assertGreater(info["added_points"], 50)
        self.assertEqual(len(conf), len(aug_pts))

    def test_tall_cylinder_with_handle_stays_upright(self):
        rng = np.random.default_rng(11)
        # Tall cylinder: radius=0.25, height=1.25 (H = 2.5 * D), tilted by 25 deg initially
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.25, height=1.25, resolution=48, split=16)
        handle = o3d.geometry.TriangleMesh.create_box(width=0.18, height=0.08, depth=0.45)
        handle.translate([0.24, -0.04, -0.22])
        mug = cyl + handle
        R_tilt = mug.get_rotation_matrix_from_xyz((math.radians(22.0), math.radians(-14.0), 0.0))
        mug.rotate(R_tilt, center=np.zeros(3))
        true_axis = R_tilt @ np.array([0.0, 0.0, 1.0])

        pts = np.asarray(mug.sample_points_uniformly(5000).points, dtype=np.float64)
        pts += rng.normal(0.0, 0.001, size=pts.shape)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
        nrms = np.asarray(pc.normals, dtype=np.float64)
        flip = np.einsum("ij,ij->i", nrms, pts - pts.mean(axis=0)) < 0.0
        nrms[flip] *= -1.0

        stats = compute_adaptive_point_stats(pts, self.cfg_adapt, norm_eps=float(self.num_cfg["norm_eps"]))
        regions = segment_and_select_regions(pts, nrms, stats, self.cfg_adapt, self.num_cfg)
        cyl_regs = [r for r in regions if r.model_type == "cylinder" and r.cylinder is not None]
        self.assertGreaterEqual(len(cyl_regs), 1)
        est_axis = cyl_regs[0].cylinder.axis
        axis_err_deg = math.degrees(math.acos(min(abs(float(np.dot(est_axis, true_axis))), 1.0)))
        self.assertLess(axis_err_deg, 3.0)

        # Canonical frame using BIC cylinder axis keeps tall mug upright (Z_span > X_span)
        cam_ring = np.array([
            R_tilt @ np.array([1.2 * math.cos(a), 1.2 * math.sin(a), 0.4])
            for a in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
        ])
        pc_can, _, _ = _establish_canonical_frame(pc, support_plane=None, cam_centers=cam_ring, cylinder_axis=est_axis)
        p_can = np.asarray(pc_can.points)
        z_span = float(np.ptp(p_can[:, 2]))
        x_span = float(np.ptp(p_can[:, 0]))
        self.assertGreater(z_span, x_span)

    def test_ring_pairs_component_gap_and_vram_budget(self):
        tmp_dir = paths.ROOT / "scratch" / "_test_ring_pairs"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            names = [f"frame_{i:05d}.jpg" for i in range(1, 41)]
            pfile = build_sequential_ring_pairs(names, overlap=6, tail_frac=0.15, out_path=tmp_dir / "pairs.txt")
            lines = pfile.read_text(encoding="utf-8").splitlines()
            self.assertIn("frame_00001.jpg frame_00002.jpg", lines)
            self.assertIn("frame_00001.jpg frame_00040.jpg", lines)
            self.assertNotIn("frame_00001.jpg frame_00020.jpg", lines)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Two comparable real objects + two tiny dust clusters
        mask = largest_gap_component_mask(np.array([5000, 4200, 120, 45]), min_abs_size=15, gap_ratio_threshold=3.0)
        self.assertEqual(mask.tolist(), [True, True, False, False])

        sz, info = estimate_vram_mvs_max_image_size(self.cfg_adapt)
        lo, hi = self.cfg_adapt["mvs_size_bounds"]
        self.assertGreaterEqual(sz, lo)
        self.assertLessEqual(sz, hi)
        self.assertIn("derived_max_image_size", info)


if __name__ == "__main__":
    unittest.main()
