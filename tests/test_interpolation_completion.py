"""Kiem thu module bu khuyet 3D bang noi suy (Single-Object & Multi-Object Per-Component)."""
from __future__ import annotations

import unittest

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from nckh_2.backends.interpolation import fill_point_cloud_holes


class TestInterpolationCompletion(unittest.TestCase):
    def test_single_object_hole_filled_by_symmetry(self):
        box = o3d.geometry.TriangleMesh.create_box(width=2.0, height=1.2, depth=0.8)
        box.translate([-1.0, -0.6, 0.0])
        pc_gt = box.sample_points_uniformly(number_of_points=4000)
        pts_gt = np.asarray(pc_gt.points, dtype=np.float64)

        # Duc 1 lo lon o nua phai mat tren (+Z)
        hole_mask = (pts_gt[:, 2] > 0.78) & (pts_gt[:, 0] > 0.25) & (pts_gt[:, 0] < 0.85) & (np.abs(pts_gt[:, 1]) < 0.35)
        pts_partial = pts_gt[~hole_mask]
        pc_partial = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_partial))
        pc_partial.paint_uniform_color([0.6, 0.5, 0.4])
        pc_partial.estimate_normals()

        # Khoang cach tu vung lo GT den point cloud truoc khi bu
        d_before, _ = cKDTree(pts_partial).query(pts_gt[hole_mask], k=1)

        pc_filled, conf, stats = fill_point_cloud_holes(
            pc_partial,
            sample_dir=None,
            interp_cfg={"enable": True, "use_visual_hull": False, "use_symmetry": True, "use_biharmonic": False},
        )
        self.assertTrue(stats["enabled"])
        self.assertGreater(stats["sym_added"], 20)
        self.assertEqual(len(conf), len(pc_filled.points))
        self.assertTrue(np.all((conf >= 0.0) & (conf <= 1.0)))

        # Khoang cach tu vung lo GT den point cloud sau khi bu phai giam manh
        d_after, _ = cKDTree(np.asarray(pc_filled.points)).query(pts_gt[hole_mask], k=1)
        self.assertLess(float(np.mean(d_after)), 0.5 * float(np.mean(d_before)))

    def test_two_different_objects_per_component(self):
        box = o3d.geometry.TriangleMesh.create_box(width=1.4, height=1.0, depth=0.7)
        box.translate([-2.5, 0.0, 0.0])
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.55, resolution=16)
        sphere.translate([2.5, 0.0, 0.4])

        p1 = np.asarray(box.sample_points_uniformly(2000).points)
        p2 = np.asarray(sphere.sample_points_uniformly(2000).points)
        # Duc lo tren ca 2 vat the
        p1 = p1[~((p1[:, 0] > -1.25) & (p1[:, 1] > 0.5))]
        p2 = p2[~((p2[:, 0] > 2.8) & (p2[:, 2] > 0.4))]

        pc_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([p1, p2])))
        pc_scene.paint_uniform_color([0.5, 0.6, 0.7])
        pc_scene.estimate_normals()

        pc_filled, conf, stats = fill_point_cloud_holes(
            pc_scene,
            sample_dir=None,
            interp_cfg={"enable": True, "use_visual_hull": False, "use_symmetry": True, "use_biharmonic": False},
        )
        self.assertEqual(stats["components"], 2)
        self.assertGreater(stats["sym_added"], 30)
        self.assertEqual(len(conf), len(pc_filled.points))


if __name__ == "__main__":
    unittest.main()
