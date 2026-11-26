"""Kiem tra toan ven tat ca cac subpackages va modules trong nckh_2."""
import unittest


class TestImports(unittest.TestCase):
    def test_core_imports(self):
        from nckh_2.core import paths, config, types, interfaces
        self.assertIsNotNone(paths.ROOT)
        self.assertTrue(paths.CONFIG_FILE.exists())
        cfg = config.load_config()
        self.assertIn("blur_filter", cfg)
        self.assertIn("mesh_refine", cfg)

    def test_capture_imports(self):
        from nckh_2.capture import (
            sample_store,
            camera_io,
            recorder,
            mobile_camera,
            choose_sample,
            list_samples,
            record_video,
            load_mobile_html,
        )
        html = load_mobile_html()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertGreater(len(html), 5000)

    def test_backends_imports(self):
        from nckh_2.backends import (
            device,
            colmap_io,
            colmap_patch,
            geometry,
            smoothing,
            get_device,
            clean_mesh_topology,
            ground_mesh_z0,
        )
        self.assertIn(get_device(), ("cuda", "cpu"))

    def test_pipeline_imports(self):
        from nckh_2.pipeline import (
            runner,
            blur_filter,
            reconstruct,
            mesh_export,
            mesh_refine,
            completion,
        )
        self.assertEqual(set(runner.STEPS.keys()), {1, 2, 3, 4, 5})

    def test_evaluation_imports(self):
        from nckh_2.evaluation import spec_figures, metrics
        box = metrics.build_synthetic_detailed_box(subdiv_iters=1)
        self.assertGreater(len(box.vertices), 0)
        self.assertGreater(len(box.triangles), 0)


if __name__ == "__main__":
    unittest.main()
