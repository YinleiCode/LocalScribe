from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bundle = _load("offline_asr_bundle", SCRIPT_DIR / "offline_asr_bundle.py")
sitecustomize = _load("localscribe_bundled_sitecustomize", SCRIPT_DIR / "localscribe_bundled_sitecustomize.py")


class OfflineAsrBundleTest(unittest.TestCase):
    def _resources(self, root: Path) -> Path:
        resources = root / "Resources"
        cache = resources / "huggingface/hub" / bundle.cache_dir_name()
        revision = "abc123"
        snapshot = cache / "snapshots" / revision
        blobs = cache / "blobs"
        snapshot.mkdir(parents=True)
        blobs.mkdir()
        (cache / "refs").mkdir()
        (cache / "refs/main").write_text(revision + "\n", encoding="utf-8")
        for index, name in enumerate(bundle.REQUIRED_SNAPSHOT_FILES):
            content = b"fixture"
            if name == "model.safetensors.index.json":
                content = json.dumps({"weight_map": {"layer": "model.safetensors"}}).encode()
            blob_name = f"blob-{index}"
            (blobs / blob_name).write_bytes(content)
            (snapshot / name).symlink_to(Path("../../blobs") / blob_name)
        bundle.write_manifest(resources, minimum_model_bytes=1)
        return resources

    def test_manifest_round_trip_and_hash_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources = self._resources(Path(tmp))
            result = bundle.verify_manifest(resources, check_hash=True, minimum_model_bytes=1)
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["hash_checked"])

    def test_same_size_corruption_is_rejected_by_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources = self._resources(Path(tmp))
            manifest = json.loads((resources / bundle.MANIFEST_RELATIVE).read_text())
            model_entry = next(
                item for item in manifest["required_files"] if item["relative_path"].endswith("model.safetensors")
            )
            (resources / model_entry["relative_path"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(bundle.BundleValidationError, "SHA-256 mismatch"):
                bundle.verify_manifest(resources, check_hash=True, minimum_model_bytes=1)

    def test_runtime_uses_disabled_cache_when_bundle_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources = self._resources(Path(tmp))
            manifest = json.loads((resources / bundle.MANIFEST_RELATIVE).read_text())
            missing = resources / manifest["required_files"][0]["relative_path"]
            missing.unlink()
            env = {"LOCALSCRIBE_RESOURCES": str(resources)}
            ready, reason = sitecustomize.configure_bundled_offline_asr(env)
            self.assertFalse(ready)
            self.assertIn("huggingface-disabled", env["HF_HOME"])
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["LOCALSCRIBE_OFFLINE_QWEN_AVAILABLE"], "0")
            self.assertNotEqual(reason, "ready")

    def test_runtime_uses_disabled_cache_when_mlx_audio_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            resources = self._resources(Path(tmp))
            env = {"LOCALSCRIBE_RESOURCES": str(resources)}
            with patch.object(sitecustomize.importlib.util, "find_spec", return_value=None):
                ready, reason = sitecustomize.configure_bundled_offline_asr(env)
            self.assertFalse(ready)
            self.assertEqual(reason, "mlx_audio_missing")
            self.assertIn("huggingface-disabled", env["HF_HUB_CACHE"])
            self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")

    def test_bundled_runtime_caches_are_outside_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resources = root / "LocalScribe.app/Contents/Resources"
            env = {"LOCALSCRIBE_RESOURCES": str(resources)}
            applied = sitecustomize.configure_bundled_runtime_cache(env, home=root)
            self.assertTrue(applied)
            self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertEqual(env["NUMBA_CACHE_DIR"], str(root / "Library/Caches/LocalScribe/numba"))
            self.assertEqual(
                env["LOCALSCRIBE_SENKO_CACHE_DIR"],
                str(root / "Library/Caches/LocalScribe/senko/coreml"),
            )
            self.assertFalse(env["NUMBA_CACHE_DIR"].startswith(str(resources)))


if __name__ == "__main__":
    unittest.main()
