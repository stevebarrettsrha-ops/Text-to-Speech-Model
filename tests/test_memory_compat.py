"""Qwen's module-global weights must be released even after cache hits."""
import importlib.util
import sys
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock


class Model:
    pass


class MemoryCompat(unittest.TestCase):
    def test_free_reaches_private_cache_after_lazy_node_import(self):
        mm = types.SimpleNamespace(unload_all_models=mock.Mock(),
                                   soft_empty_cache=mock.Mock())
        registry = types.SimpleNamespace(NODE_CLASS_MAPPINGS={})
        with mock.patch.dict(sys.modules, {"comfy": types.SimpleNamespace(model_management=mm),
                                          "nodes": registry}):
            path = Path(__file__).resolve().parents[1] / "compat/script_builder_memory/__init__.py"
            spec = importlib.util.spec_from_file_location("memory_shim", path)
            shim = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(shim)
            wrapped = mm.unload_all_models
            shim.install()
            self.assertIs(mm.unload_all_models, wrapped)
            upstream = types.ModuleType("fake_qwen_nodes")
            upstream._MODEL_CACHE = {"same-model": Model()}
            ref = weakref.ref(upstream._MODEL_CACHE["same-model"])
            node = type("CustomVoice", (), {"__module__": upstream.__name__})
            registry.NODE_CLASS_MAPPINGS["FB_Qwen3TTSCustomVoice"] = node
            unrelated = {"keep": Model()}
            other = types.ModuleType("unrelated_nodes")
            other._MODEL_CACHE = unrelated
            registry.NODE_CLASS_MAPPINGS["OtherNode"] = type("Other", (), {"__module__": other.__name__})
            with mock.patch.dict(sys.modules, {upstream.__name__: upstream, other.__name__: other}):
                mm.unload_all_models()
                self.assertIsNone(ref())
                self.assertEqual(upstream._MODEL_CACHE, {})
                self.assertEqual(len(unrelated), 1)
                mm.soft_empty_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()
