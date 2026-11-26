"""Unit & integration tests for the unified pipeline.refine subpackage & layered profiles."""
from __future__ import annotations

import json
import shutil
import unittest

import open3d as o3d

from nckh_2.backends.remesh import isotropic_remesh
from nckh_2.backends.smoothing import drift_monitored_smooth
from nckh_2.core import paths
from nckh_2.core.config import list_profiles, load_config
from nckh_2.evaluation.mesh_audit import audit_mesh
from nckh_2.pipeline.refine import refine_obj_file


class TestRefineWorkflow(unittest.TestCase):
    def test_layered_config_profiles(self):
        profiles = list_profiles()
        self.assertIn("curved_object", profiles)
        self.assertIn("sharp_cad", profiles)
        self.assertIn("scene_multi", profiles)

        cfg_sharp = load_config(profile="sharp_cad")
        self.assertEqual(cfg_sharp["mesh_refine"]["smoothing"]["mode"], "bilateral_normal")
        self.assertIn("blur_filter", cfg_sharp)

        cfg_scene = load_config(profile="scene_multi")
        self.assertFalse(cfg_scene["mesh_refine"]["remesh"]["enable"])

    def test_6side_audit_and_remesh_and_drift_smooth(self):
        box = o3d.geometry.TriangleMesh.create_box(width=2.0, height=1.2, depth=0.8)
        box = box.subdivide_midpoint(number_of_iterations=2)
        box.compute_vertex_normals()

        rep0 = audit_mesh(box, stage_label="init")
        self.assertEqual(len(rep0.sides), 6)
        self.assertEqual(rep0.num_components, 1)
        self.assertGreater(rep0.surface_area, 0.0)

        remeshed, r_stats = isotropic_remesh(box, target_edge_frac=0.08, iterations=2)
        self.assertTrue(r_stats.applied)
        self.assertGreater(len(remeshed.triangles), 0)

        smoothed, s_stats = drift_monitored_smooth(
            remeshed, mode="hybrid", batch_iters=2, max_batches=3, max_drift_frac=0.01
        )
        self.assertGreaterEqual(s_stats["batches_run"], 1)
        self.assertEqual(len(smoothed.vertices), len(remeshed.vertices))

    def test_two_different_shape_objects_with_default_config(self):
        """Kiem thu canh gom 2 vat the khac hinh dang (1 hop canh sac + 1 hinh cau cong) voi default.yaml."""
        box = o3d.geometry.TriangleMesh.create_box(width=2.0, height=1.2, depth=0.8).subdivide_midpoint(2)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=12)
        sphere.translate([3.5, 0.5, 0.5])
        scene = box + sphere
        scene.compute_vertex_normals()

        cfg = load_config()["mesh_refine"]
        cfg["remesh"]["target_edge_frac"] = 0.06
        cfg["remesh"]["iterations"] = 2
        cfg["audit"]["enable_report"] = False

        from nckh_2.pipeline.refine import refine_mesh_with_report
        refined, rep = refine_mesh_with_report(scene, r_cfg=cfg)
        self.assertEqual(rep.before.num_components, 2)
        self.assertEqual(rep.after.num_components, 2)
        self.assertEqual(rep.smooth_mode, "hybrid")
        self.assertGreater(len(refined.triangles), 100)

    def test_end_to_end_unified_refine_obj_file(self):
        tmp_dir = paths.ROOT / "scratch" / "_test_5stage_workflow"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            in_obj = tmp_dir / "sample.obj"
            out_dir = tmp_dir / "audit_out"

            box = o3d.geometry.TriangleMesh.create_box(width=1.5, height=1.0, depth=0.7)
            box = box.subdivide_midpoint(number_of_iterations=2)
            dust = o3d.geometry.TriangleMesh.create_tetrahedron(radius=0.01)
            dust.translate([5.0, 5.0, 5.0])
            scene = box + dust
            scene.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(in_obj), scene, write_ascii=True, write_vertex_normals=True)

            cfg = load_config(profile="curved_object")["mesh_refine"]
            cfg["remesh"]["target_edge_frac"] = 0.08
            cfg["remesh"]["iterations"] = 2
            cfg["smoothing"]["max_batches"] = 2

            rep = refine_obj_file(in_obj, out_dir, r_cfg=cfg)
            ledger = out_dir / "stage_metrics.json"
            fig = out_dir / "before_after_6side_audit.png"
            final_obj = out_dir / "sample_refined.obj"
            self.assertTrue(ledger.exists())
            self.assertTrue(fig.exists())
            self.assertTrue(final_obj.exists())
            self.assertEqual(rep.before.num_components, 2)
            self.assertEqual(rep.after.num_components, 1)

            data = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(data["stages"]), 2)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
