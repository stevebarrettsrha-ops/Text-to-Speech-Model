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
    def test_moss_offloads_both_owners_even_while_output_cache_retains_them(self):
        mm = types.SimpleNamespace(unload_all_models=mock.Mock(),
                                   soft_empty_cache=mock.Mock())
        registry = types.SimpleNamespace(NODE_CLASS_MAPPINGS={})
        upstream = types.ModuleType("fake_moss_loader")
        model, tokenizer = mock.Mock(), mock.Mock()
        processor = types.SimpleNamespace(audio_tokenizer=tokenizer)
        upstream._current_model, upstream._current_processor = model, processor
        # ComfyUI holds the loader's output independently of these globals.
        cached_pipe = (model, processor, 24000, "cuda", "moss-test")
        model_ref, tokenizer_ref = weakref.ref(model), weakref.ref(tokenizer)
        with mock.patch.dict(sys.modules, {
                "comfy": types.SimpleNamespace(model_management=mm),
                "nodes": registry, upstream.__name__: upstream}):
            path = Path(__file__).resolve().parents[1] / "compat/script_builder_memory/__init__.py"
            spec = importlib.util.spec_from_file_location("moss_memory_shim", path)
            shim = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(shim)
            registry.NODE_CLASS_MAPPINGS["MossTTSModelLoader"] = type(
                "Loader", (), {"__module__": upstream.__name__})
            mm.unload_all_models()
            model.cpu.assert_called_once_with()
            tokenizer.cpu.assert_called_once_with()
            self.assertIsNone(upstream._current_model)
            self.assertIsNone(upstream._current_processor)
            self.assertIs(cached_pipe[0], model_ref())
            self.assertIs(cached_pipe[1].audio_tokenizer, tokenizer_ref())
            # A second free has no owners to offload a second time.
            mm.unload_all_models()
            model.cpu.assert_called_once_with()
            tokenizer.cpu.assert_called_once_with()

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
