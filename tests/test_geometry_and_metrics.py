"""Kiem tra cac thuat toan hinh hoc, lam min va metrics tren mo hinh synthetic."""
import numpy as np
import open3d as o3d
import unittest

from nckh_2.backends.geometry import clean_mesh_topology, ground_mesh_z0, robust_diag
from nckh_2.backends.smoothing import bilateral_normal_smooth, compute_discrete_curvature
from nckh_2.evaluation.metrics import (
    build_synthetic_detailed_box,
    compute_surface_metrics,
    evaluate_detail_preservation,
)


class TestGeometryAndMetrics(unittest.TestCase):
    def setUp(self):
        self.box = build_synthetic_detailed_box(subdiv_iters=2)

    def test_ground_mesh_z0(self):
        grounded = ground_mesh_z0(self.box)
        v = np.asarray(grounded.vertices)
        self.assertAlmostEqual(float(v[:, 2].min()), 0.0, places=5)

    def test_robust_diag(self):
        v = np.asarray(self.box.vertices)
        diag = robust_diag(v)
        self.assertGreater(diag, 0.5)

    def test_smoothing_preserves_geometry(self):
        smoothed = bilateral_normal_smooth(self.box, normal_iter=2, vertex_iter=5)
        self.assertEqual(len(smoothed.vertices), len(self.box.vertices))
        self.assertEqual(len(smoothed.triangles), len(self.box.triangles))

    def test_surface_metrics(self):
        pts = np.asarray(self.box.vertices)
        norms = np.asarray(self.box.vertex_normals)
        metrics = compute_surface_metrics(pts, norms, pts, norms)
        self.assertAlmostEqual(metrics.chamfer, 0.0, places=5)
        self.assertAlmostEqual(metrics.normal_consistency, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
