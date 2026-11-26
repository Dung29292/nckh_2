"""Kiem thu phat hien lech truc (PCA/OBB skew), loc ghost shell, can truc va batch runner."""
from __future__ import annotations

import json
import shutil
import unittest

import numpy as np
import open3d as o3d

from nckh_2.backends.alignment import (
    align_pca_canonical,
    compute_pose_skew,
    detect_and_prune_ghost_shells,
)
from nckh_2.core import paths
from nckh_2.pipeline.refine.batch import run_batch_refine


class TestAlignmentAndBatch(unittest.TestCase):
    def test_pose_skew_detection_and_pca_canonical_alignment(self):
        box = o3d.geometry.TriangleMesh.create_box(width=3.0, height=1.5, depth=0.6)
        box = box.subdivide_midpoint(number_of_iterations=2)

        rep_straight = compute_pose_skew(box)
        self.assertFalse(rep_straight.is_skewed)
        self.assertLess(rep_straight.max_axis_skew_deg, 1.0)

        # Xoay vat the lech 22 do quanh truc Z
        R_skew = box.get_rotation_matrix_from_xyz((0.0, 0.0, np.radians(22.0)))
        box.rotate(R_skew, center=box.get_center())

        rep_skew = compute_pose_skew(box, skew_warn_deg=5.0)
        self.assertTrue(rep_skew.is_skewed)
        self.assertGreater(rep_skew.max_axis_skew_deg, 18.0)

        # Can lai bang align_pca_canonical -> het lech va bao toan det(R)=+1
        aligned, T = align_pca_canonical(box, normalize_unit_scale=False)
        self.assertGreater(float(np.linalg.det(T[:3, :3])), 0.0)
        rep_fixed = compute_pose_skew(aligned, skew_warn_deg=5.0)
        self.assertFalse(rep_fixed.is_skewed)
        self.assertLess(rep_fixed.max_axis_skew_deg, 1.5)

    def test_ghost_shell_pruned_while_hollow_inner_wall_preserved(self):
        main_box = o3d.geometry.TriangleMesh.create_box(width=2.0, height=1.5, depth=1.0)
        main_box = main_box.subdivide_midpoint(number_of_iterations=2)

        # Vo trung lap lech nhe (cung chieu phap tuyen -> ghost shell)
        ghost = o3d.geometry.TriangleMesh(main_box)
        ghost.scale(0.99, center=ghost.get_center())
        ghost.translate([0.01, 0.01, 0.0])

        # Thanh trong cua vat rong (dao nguoc phap tuyen -> phai duoc bao toan!)
        inner_wall = o3d.geometry.TriangleMesh(main_box)
        inner_wall.scale(0.99, center=inner_wall.get_center())
        inner_wall.triangles = o3d.utility.Vector3iVector(np.asarray(inner_wall.triangles)[:, ::-1])

        scene = main_box + ghost + inner_wall
        cleaned, stats = detect_and_prune_ghost_shells(scene, overlap_dist_frac=0.03, ghost_overlap_thr=0.65, prune=True)
        self.assertEqual(stats["components_checked"], 3)
        self.assertEqual(stats["ghost_shells_detected"], 1)
        self.assertEqual(stats["ghost_shells_removed"], 1)
        # Giu lai 2 thanh phan: vo ngoai + thanh trong vat rong
        self.assertEqual(len(cleaned.triangles), len(main_box.triangles) + len(inner_wall.triangles))

    def test_batch_refine_with_error_isolation(self):
        tmp_dir = paths.ROOT / "scratch" / "_test_batch_refine"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        in_dir = tmp_dir / "inputs"
        out_dir = tmp_dir / "outputs"
        in_dir.mkdir(parents=True, exist_ok=True)
        try:
            # 2 asset hop le + 1 asset rong/hong de test co lap loi
            b1 = o3d.geometry.TriangleMesh.create_box(width=1.5, height=1.0, depth=0.6).subdivide_midpoint(2)
            b2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.8, resolution=10)
            o3d.io.write_triangle_mesh(str(in_dir / "asset_a.obj"), b1, write_ascii=True)
            o3d.io.write_triangle_mesh(str(in_dir / "asset_b.obj"), b2, write_ascii=True)
            (in_dir / "asset_corrupt.obj").write_text("# empty invalid obj\n", encoding="utf-8")

            summary = run_batch_refine(
                in_dir,
                out_dir,
                pattern="*.obj",
                profile="scene_multi",
                fail_fast=False,
            )
            self.assertEqual(summary["total_assets"], 3)
            self.assertEqual(summary["succeeded"], 2)
            self.assertEqual(summary["failed"], 1)
            self.assertTrue((out_dir / "batch_summary.csv").exists())
            self.assertTrue((out_dir / "batch_summary.json").exists())
            data = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["results"]), 3)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
