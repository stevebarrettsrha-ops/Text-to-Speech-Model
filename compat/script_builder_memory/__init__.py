"""Let ComfyUI's /free release the speech nodes' private model caches too.

Qwen stores models in nodes._MODEL_CACHE, outside ComfyUI's ModelPatcher
registry. Its cache-hit path also skips updating _unload_callback, so the
last line's unload switch can do nothing. This shim hooks the normal unload
operation, on ComfyUI's execution thread, after the line has returned.
It discovers registered classes at call time, independent of import order.
MOSS retains its model and audio tokenizer in module globals as well as
ComfyUI's output cache. Move those weights off the card before clearing the
globals, so a cached pipeline tuple cannot keep the GPU allocation alive.
"""
import gc
import sys
from functools import wraps


def release_qwen_cache():
    import nodes
    seen = set()
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        if not name.startswith("FB_Qwen3TTS"):
            continue
        module = sys.modules.get(cls.__module__)
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        cache = getattr(module, "_MODEL_CACHE", None)
        if isinstance(cache, dict):
            cache.clear()


def release_moss_cache():
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get("MossTTSModelLoader")
    module = sys.modules.get(getattr(cls, "__module__", ""))
    if module is None:
        return
    processor = getattr(module, "_current_processor", None)
    model = getattr(module, "_current_model", None)
    # These are the same owners and the same offload operation used by the
    # upstream loader when changing checkpoints. /free also resets ComfyUI's
    # output cache after unload_all_models returns, dropping the host copies.
    if processor is not None:
        tokenizer = getattr(processor, "audio_tokenizer", None)
        if tokenizer is not None:
            tokenizer.cpu()
        module._current_processor = None
    if model is not None:
        model.cpu()
        module._current_model = None


def install():
    from comfy import model_management as mm
    original = mm.unload_all_models
    if getattr(original, "_script_builder_memory", False) is True:
        return

    @wraps(original)
    def unload(*args, **kwargs):
        release_qwen_cache()
        release_moss_cache()
        result = original(*args, **kwargs)
        gc.collect()
        mm.soft_empty_cache()
        return result

    unload._script_builder_memory = True
    mm.unload_all_models = unload


class ScriptBuilderMemory:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "Script Builder"

    def noop(self):
        return ()


NODE_CLASS_MAPPINGS = {"ScriptBuilderMemoryV2": ScriptBuilderMemory}
install()
