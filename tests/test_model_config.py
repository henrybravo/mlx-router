import tempfile
import unittest
from pathlib import Path

from mlx_router.config.model_config import ModelConfig


class TestModelConfig(unittest.TestCase):
    def tearDown(self):
        ModelConfig.set_model_directory(None)

    def test_resolve_model_path_prefers_parent_directory_with_weights(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_root = Path(tmp_dir) / "models--mlx-community--Qwen3.5-35B-A3B-8bit"
            snapshot_dir = model_root / "snapshots" / "abc123"
            snapshot_dir.mkdir(parents=True)

            (model_root / "config.json").write_text("{}")
            (snapshot_dir / "config.json").write_text("{}")
            (model_root / "model-00001-of-00008.safetensors").write_text("weights")

            ModelConfig.set_model_directory(tmp_dir)

            resolved = ModelConfig.resolve_model_path("mlx-community/Qwen3.5-35B-A3B-8bit")

            self.assertEqual(Path(resolved).resolve(), model_root.resolve())

    def test_resolve_model_path_uses_snapshot_when_weights_live_in_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_root = Path(tmp_dir) / "models--mlx-community--Qwen3.5-35B-A3B-8bit"
            snapshot_dir = model_root / "snapshots" / "abc123"
            snapshot_dir.mkdir(parents=True)

            (snapshot_dir / "config.json").write_text("{}")
            (snapshot_dir / "model-00001-of-00008.safetensors").write_text("weights")

            ModelConfig.set_model_directory(tmp_dir)

            resolved = ModelConfig.resolve_model_path("mlx-community/Qwen3.5-35B-A3B-8bit")

            self.assertEqual(Path(resolved).resolve(), snapshot_dir.resolve())
