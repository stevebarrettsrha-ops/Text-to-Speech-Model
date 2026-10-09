"""Let ComfyUI's /free release Qwen's private model cache too.

Qwen stores models in nodes._MODEL_CACHE, outside ComfyUI's ModelPatcher
registry. Its cache-hit path also skips updating _unload_callback, so the
last line's unload switch can do nothing. This shim hooks the normal unload
operation, on ComfyUI's execution thread, after the line has returned.
It discovers registered classes at call time, independent of import order.
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


def install():
    from comfy import model_management as mm
    original = mm.unload_all_models
    if getattr(original, "_script_builder_memory", False) is True:
        return

    @wraps(original)
    def unload(*args, **kwargs):
        release_qwen_cache()
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


NODE_CLASS_MAPPINGS = {"ScriptBuilderMemory": ScriptBuilderMemory}
install()
