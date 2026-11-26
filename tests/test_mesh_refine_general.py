"""Kiem thu toan dien pipeline tinh chinh .obj tong quat (multi-object, UV/MTL, toa do, determinism)."""
from __future__ import annotations

from pathlib import Path
import shutil
import unittest

import numpy as np
import open3d as o3d

from nckh_2.backends.smoothing import suppress_mesh_spikes
from nckh_2.core import paths
from nckh_2.pipeline.mesh_refine import refine_mesh_with_report, refine_obj_file


def _build_multi_object_scene_with_dust() -> o3d.geometry.TriangleMesh:
    """Tao canh gom 2 vat the hop le tai toa do the gioi khac 0 + 1 manh bui 4 tam giac."""
    obj1 = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
    obj1 = obj1.subdivide_midpoint(number_of_iterations=2)
    obj1.translate([10.0, -4.0, 3.5])

    obj2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.6, resolution=10)
    obj2.translate([14.0, -4.0, 4.1])

    # Manh bui sieu nho (4 tam giac tu dien) troi noi
    dust = o3d.geometry.TriangleMesh.create_tetrahedron(radius=0.01)
    dust.translate([18.0, 2.0, 9.0])

    scene = obj1 + obj2 + dust
    scene.paint_uniform_color([0.3, 0.6, 0.9])
    scene.compute_vertex_normals()
    return scene


class TestGeneralMeshRefine(unittest.TestCase):
    def test_multi_object_preserved_and_dust_removed(self):
        scene = _build_multi_object_scene_with_dust()
        refined, report = refine_mesh_with_report(
            scene,
            r_cfg={
                "smooth_mode": "bilateral_normal",
                "min_component_tri_ratio": 0.005,
                "min_component_area_ratio": 0.002,
                "min_component_triangles": 15,
                "align_upright": False,
            },
        )
        # 3 thanh phan ban dau (box + sphere + dust) -> giu lai 2 vat the hop le, chi loai 1 manh bui
        self.assertEqual(report.before.num_components, 3)
        self.assertEqual(report.components_removed, 1)
        self.assertEqual(report.after.num_components, 2)
        self.assertTrue(report.after.has_vertex_colors)
        self.assertTrue(report.after.has_vertex_normals)

    def test_coordinate_frame_and_scale_preserved(self):
        scene = _build_multi_object_scene_with_dust()
        refined, report = refine_mesh_with_report(
            scene,
            r_cfg={"smooth_mode": "bilateral_normal", "align_upright": False},
        )
        # Toa do goc: box [10..12, -4..-2, 3.5..5.5], sphere tam [14, -4, 4.1] ban kinh 0.6 -> y_min = -4.6
        self.assertAlmostEqual(report.after.bbox_min[0], 10.0, delta=0.05)
        self.assertAlmostEqual(report.after.bbox_min[1], -4.6, delta=0.05)
        self.assertAlmostEqual(report.after.bbox_min[2], 3.5, delta=0.05)
        # Ty le kich thuoc canh (X span ~ 4.6) khong bi normalize ve [0, 1]
        self.assertGreater(report.after.bbox_extents[0], 4.0)

    def test_spike_suppression(self):
        box = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
        box = box.subdivide_midpoint(number_of_iterations=3)
        verts = np.asarray(box.vertices, dtype=np.float64).copy()
        # Tim mot dinh nam o giua mat +Z va keo vot len thanh gai nhon
        top_idx = int(np.argmin(np.linalg.norm(verts - np.array([1.0, 1.0, 2.0]), axis=1)))
        verts[top_idx, 2] += 1.5
        box.vertices = o3d.utility.Vector3dVector(verts)

        cleaned, n_spikes = suppress_mesh_spikes(box, sigma_threshold=3.5, pull_strength=0.9)
        self.assertGreaterEqual(n_spikes, 1)
        v_after = np.asarray(cleaned.vertices)
        self.assertLess(v_after[top_idx, 2], 2.5)

    def test_determinism_and_empty_mesh(self):
        scene = _build_multi_object_scene_with_dust()
        cfg = {"smooth_mode": "bilateral_normal", "bilateral_normal_iter": 3, "bilateral_vertex_iter": 6}
        m1, _ = refine_mesh_with_report(scene, r_cfg=cfg)
        m2, _ = refine_mesh_with_report(scene, r_cfg=cfg)
        np.testing.assert_allclose(np.asarray(m1.vertices), np.asarray(m2.vertices), atol=0.0)

        # Empty mesh khong duoc crash
        empty = o3d.geometry.TriangleMesh()
        m_empty, rep_empty = refine_mesh_with_report(empty, r_cfg=cfg)
        self.assertEqual(len(m_empty.vertices), 0)
        self.assertEqual(rep_empty.after.num_triangles, 0)

    def test_obj_file_roundtrip_preserves_uvs_and_mtllib(self):
        tmp_dir = paths.ROOT / "scratch" / "_test_refine_obj_io"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            in_obj = tmp_dir / "input.obj"
            out_obj = tmp_dir / "nested" / "output.obj"

            box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
            box = box.subdivide_midpoint(number_of_iterations=2)
            n_t = len(box.triangles)
            uvs = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64), (n_t, 1))
            box.triangle_uvs = o3d.utility.Vector2dVector(uvs)
            box.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(in_obj), box, write_ascii=True, write_vertex_normals=True, write_triangle_uvs=True)

            # Them dong mtllib gia lap o dau file nguon
            in_obj.write_text("mtllib scene_materials.mtl\n" + in_obj.read_text(encoding="utf-8"), encoding="utf-8")

            report = refine_obj_file(in_obj, out_obj, r_cfg={"smooth_mode": "taubin", "taubin_iter": 3})
            self.assertTrue(out_obj.exists())
            # Phai bao toan toan bo 192 tam giac (khong bi cat nho do UV seam) va giu nguyen UVs + mtllib
            self.assertEqual(report.after.num_components, 1)
            self.assertEqual(report.after.num_triangles, n_t)
            self.assertTrue(report.after.has_triangle_uvs)
            out_text = out_obj.read_text(encoding="utf-8")
            self.assertIn("mtllib scene_materials.mtl", out_text)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
