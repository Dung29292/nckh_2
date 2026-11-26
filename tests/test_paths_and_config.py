"""Kiem tra duong dan va tinh doc lap voi thu muc thuc thi (CWD-independence)."""
import os
from pathlib import Path
import tempfile
import unittest


class TestPathsAndConfig(unittest.TestCase):
    def test_root_path(self):
        from nckh_2.core import paths
        self.assertTrue((paths.ROOT / "pyproject.toml").exists())
        self.assertTrue(paths.CONFIG_FILE.exists())
        self.assertTrue(paths.DATA_DIR.exists())
        self.assertTrue(paths.SAMPLES_DIR.exists())
        self.assertTrue(paths.MODELS_DIR.exists())

    def test_cwd_independence(self):
        from nckh_2.core import paths, config
        orig_cwd = os.getcwd()
        tmp_dir = paths.ROOT / "scratch" / "_test_cwd_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chdir(str(tmp_dir))
            cfg = config.load_config()
            self.assertIn("reconstruction", cfg)
            resolved_ckpt = paths.resolve_project_path(cfg["masking"]["sam2_checkpoint"])
            self.assertTrue(resolved_ckpt.is_absolute())
        finally:
            os.chdir(orig_cwd)
            if tmp_dir.exists():
                try:
                    tmp_dir.rmdir()
                except Exception:
                    pass

    def test_sample_metadata(self):
        from nckh_2.core.types import SampleMetadata
        from nckh_2.core import paths
        meta = SampleMetadata.load(paths.sample_dir("box_01"))
        self.assertIsNotNone(meta)


if __name__ == "__main__":
    unittest.main()
