"""A stand-in for ComfyUI that speaks enough of its HTTP API to drive Script
Builder end to end: /system_stats, /object_info, /prompt, /history, /view,
/upload/image, /interrupt.

The /object_info payload is transcribed from flybirdxx/ComfyUI-Qwen-TTS
nodes.py and richservo/comfyui-moss-tts nodes/*.py at main, so the graphs
Script Builder builds are validated against the same input names, enums and
defaults the real nodes declare.
"""
import math, os, struct, sys, threading, time, uuid, wave
from pathlib import Path
from flask import Flask, jsonify, request, send_file

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else
            os.environ.get("MOCK_COMFY_ROOT") or "./.mock-comfy")
OUT = ROOT / "output"
IN = ROOT / "input"
for d in (OUT, IN):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
HISTORY = {}
LOG = []
INTERRUPTED = threading.Event()

ATT = ["auto", "sage_attn", "flash_attn", "sdpa", "eager"]
LANG = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German",
        "Spanish", "Portuguese", "Russian", "Italian"]
SPEAKERS = ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee",
            "Uncle_fu", "Vivian"]

GEN = {
    "seed": ["INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                     "control_after_generate": True}],
    "max_new_tokens": ["INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}],
    "top_p": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}],
    "top_k": ["INT", {"default": 50, "min": 0, "max": 100, "step": 1}],
    "temperature": ["FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.1}],
    "repetition_penalty": ["FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0}],
    "attention": [ATT, {"default": "auto"}],
    "unload_model_after_generate": ["BOOLEAN", {"default": False}],
}


def _node(required, optional=None, ret="AUDIO", category="Qwen3-TTS"):
    return {"input": {"required": required, "optional": optional or {}},
            "output": [ret], "output_name": [ret.lower()],
            "category": category}


# MOSS-TTS, from richservo/comfyui-moss-tts. Its display names are what the
# loader's model_variant enum really offers — Script Builder picks the entry it
# wants out of this list by substring, because the repo id each one maps to
# lives in the node's constants and never reaches /object_info.
MOSS_VARIANTS = ["MOSS-TTS (Delay 8B)", "MOSS-TTS (Local 1.7B)",
                 "MOSS-TTSD v1.0", "MOSS-VoiceGenerator", "MOSS-SoundEffect"]
MOSS_LANG = ["auto", "zh", "en", "ja", "ko"]
MOSS_SEED = ["INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}]
MOSS_HANDLES = {
    "head_handle": ["FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}],
    "tail_handle": ["FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}],
}


def _moss_sampling(temperature, top_p, top_k, penalty):
    return {
        "temperature": ["FLOAT", {"default": temperature, "min": 0.0,
                                  "max": 5.0, "step": 0.01}],
        "top_p": ["FLOAT", {"default": top_p, "min": 0.0, "max": 1.0,
                            "step": 0.01}],
        "top_k": ["INT", {"default": top_k, "min": 1, "max": 200, "step": 1}],
        "repetition_penalty": ["FLOAT", {"default": penalty, "min": 0.5,
                                         "max": 2.0, "step": 0.01}],
        "max_new_tokens": ["INT", {"default": 4096, "min": 1, "max": 8192,
                                   "step": 1}],
    }


OBJECT_INFO = {
    "CustomVoiceNode": _node(
        {"text": ["STRING", {"multiline": True, "default": "Hello world"}],
         "speaker": [SPEAKERS, {"default": "Ryan"}],
         "model_choice": [["0.6B", "1.7B"], {"default": "1.7B"}],
         "device": [["auto", "cuda", "xpu", "mps", "cpu"], {"default": "auto"}],
         "precision": [["bf16", "fp32"], {"default": "bf16"}],
         "language": [LANG, {"default": "Auto"}]},
        dict(GEN, instruct=["STRING", {"multiline": True, "default": ""}],
             custom_model_path=["STRING", {"default": ""}],
             custom_speaker_name=["STRING", {"default": ""}])),
    "VoiceCloneNode": _node(
        {"ref_audio": ["AUDIO"], "ref_text": ["STRING", {"default": ""}],
         "target_text": ["STRING", {"multiline": True, "default": ""}],
         "model_choice": [["0.6B", "1.7B"], {"default": "0.6B"}],
         "device": [["auto", "cuda", "xpu", "mps", "cpu"], {"default": "auto"}],
         "precision": [["bf16", "fp32"], {"default": "bf16"}],
         "language": [LANG, {"default": "Auto"}]},
        dict(GEN)),
    "VoiceDesignNode": _node(
        {"text": ["STRING", {"multiline": True, "default": "Hello world"}],
         "instruct": ["STRING", {"multiline": True, "default": ""}],
         "model_choice": [["0.6B", "1.7B"], {"default": "1.7B"}],
         "device": [["auto", "cuda", "xpu", "mps", "cpu"], {"default": "auto"}],
         "precision": [["bf16", "fp32"], {"default": "bf16"}],
         "language": [LANG, {"default": "Auto"}]},
        dict(GEN)),
    "VoiceClonePromptNode": _node(
        {"ref_audio": ["AUDIO"], "ref_text": ["STRING", {"default": ""}],
         "model_choice": [["0.6B", "1.7B"], {"default": "0.6B"}],
         "device": [["auto", "cuda", "cpu"], {"default": "auto"}],
         "precision": [["bf16", "fp32"], {"default": "bf16"}],
         "attention": [ATT, {"default": "auto"}]}, {}, "QWEN_PROMPT"),
    "LoadAudio": {"input": {"required": {
        "audio": [sorted(p.name for p in IN.glob("*")) or [""],
                  {"audio_upload": True}]}},
        "output": ["AUDIO"], "output_name": ["audio"], "category": "audio"},
    "SaveAudioAdvanced": {"input": {"required": {
        "audio": ["AUDIO"],
        "filename_prefix": ["STRING", {"default": "audio/ComfyUI"}],
        "format": [["flac", "wav", "mp3", "opus"], {"default": "flac"}]}},
        "output": [], "output_name": [], "category": "audio",
        "output_node": True},

    "MossTTSModelLoader": _node(
        {"model_variant": [MOSS_VARIANTS, {"default": MOSS_VARIANTS[0]}],
         "local_model_path": ["STRING", {"default": ""}],
         "codec_local_path": ["STRING", {"default": ""}]},
        {}, "MOSS_TTS_PIPE", "audio/MOSS-TTS"),
    "MossTTSGenerate": _node(
        dict({"moss_pipe": ["MOSS_TTS_PIPE"],
              "language": [MOSS_LANG, {"default": "auto"}],
              "text": ["STRING", {"default": "", "multiline": True}],
              "seed": MOSS_SEED,
              "enable_duration_control": ["BOOLEAN", {"default": False}],
              "duration_tokens": ["INT", {"default": 325, "min": 1,
                                          "max": 4096, "step": 1}]},
             **_moss_sampling(1.7, 0.8, 25, 1.0), **MOSS_HANDLES),
        {"reference_audio": ["AUDIO"]}, "AUDIO", "audio/MOSS-TTS"),
    "MossTTSVoiceDesign": _node(
        dict({"moss_pipe": ["MOSS_TTS_PIPE"],
              "language": [MOSS_LANG, {"default": "auto"}],
              "text": ["STRING", {"default": "", "multiline": True}],
              "instruction": ["STRING", {"default": "", "multiline": True}],
              "seed": MOSS_SEED},
             **_moss_sampling(1.5, 0.6, 50, 1.1), **MOSS_HANDLES),
        {}, "AUDIO", "audio/MOSS-TTS"),
    "MossTTSSoundEffect": _node(
        dict({"moss_pipe": ["MOSS_TTS_PIPE"],
              "ambient_sound": ["STRING", {"default": "", "multiline": True}],
              "duration_seconds": ["FLOAT", {"default": 5.0, "min": 0.5,
                                             "max": 60.0, "step": 0.5}],
              "seed": MOSS_SEED},
             **_moss_sampling(1.5, 0.6, 50, 1.2), **MOSS_HANDLES),
        {}, "AUDIO", "audio/MOSS-TTS"),
    "MossTTSDialogue": _node(
        dict({"moss_pipe": ["MOSS_TTS_PIPE"],
              "language": [MOSS_LANG, {"default": "auto"}],
              "dialogue_text": ["STRING", {"default": "", "multiline": True}],
              "speaker_count": ["INT", {"default": 2, "min": 2, "max": 2,
                                        "step": 1}],
              "normalize_text": ["BOOLEAN", {"default": True}],
              "seed": MOSS_SEED},
             **_moss_sampling(1.1, 0.9, 50, 1.1), **MOSS_HANDLES),
        {"s1_reference_audio": ["AUDIO"],
         "s1_prompt_text": ["STRING", {"default": "", "multiline": False}],
         "s2_reference_audio": ["AUDIO"],
         "s2_prompt_text": ["STRING", {"default": "", "multiline": False}]},
        "AUDIO", "audio/MOSS-TTS"),
}

# Which node sets /object_info admits to having, so a test can reproduce an
# engine whose nodes ComfyUI never loaded.
HIDDEN = set()
PREFIXES = {"qwen": ("CustomVoice", "VoiceClone", "VoiceDesign"),
            "moss": ("Moss",)}


def visible_info() -> dict:
    if not HIDDEN:
        return OBJECT_INFO
    drop = tuple(pre for eid in HIDDEN for pre in PREFIXES.get(eid, ()))
    return {k: v for k, v in OBJECT_INFO.items() if not k.startswith(drop)}


def make_wav(path: Path, seconds: float, freq: float, rate=24000):
    frames = int(rate * seconds)
    data = b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / rate)))
                    for i in range(frames))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)


# 8 GB, in bytes, the way real ComfyUI reports it — so the VRAM guard has a
# card to judge against without one being present. /mock/mode can change it.
VRAM = {"total": 8588886016}


@app.get("/system_stats")
def stats():
    return jsonify({"system": {"os": "posix", "comfyui_version": "mock"},
                    "devices": [{"name": "mock", "type": "cuda",
                                 "vram_total": VRAM["total"],
                                 "vram_free": VRAM["total"] // 2}]})


@app.post("/mock/vram/<int:mb>")
def mock_vram(mb):
    VRAM["total"] = mb * 1024 * 1024
    return jsonify({"vram_total": VRAM["total"]})


@app.get("/object_info")
def object_info():
    OBJECT_INFO["LoadAudio"]["input"]["required"]["audio"][0] = \
        sorted(p.name for p in IN.glob("*")) or [""]
    LOG.append("GET /object_info")
    return jsonify(visible_info())


@app.get("/object_info/<cls>")
def object_info_one(cls):
    info = visible_info()
    return jsonify({cls: info[cls]} if cls in info else {})


@app.post("/mock/hide/<engine>")
def mock_hide(engine):
    """Pretend ComfyUI never loaded that engine's nodes — pass 'none' to
    put them all back."""
    HIDDEN.clear()
    if engine != "none":
        HIDDEN.add(engine)
    return jsonify({"hidden": sorted(HIDDEN)})


def validate(prompt: dict):
    """The parts of ComfyUI's own validation that catch a malformed graph.

    ComfyUI calls INPUT_TYPES() afresh for every validation, so LoadAudio sees
    files uploaded since the last /object_info. Mirror that, or an upload made
    after the client cached the schema looks like a value not in the list.
    """
    OBJECT_INFO["LoadAudio"]["input"]["required"]["audio"][0] = \
        sorted(p.name for p in IN.glob("*")) or [""]
    errors = {}
    for nid, node in prompt.items():
        cls = node.get("class_type")
        spec = OBJECT_INFO.get(cls)
        if not spec:
            errors[nid] = {"class_type": cls, "errors": [
                {"message": "Cannot execute because node type does not exist",
                 "details": str(cls)}]}
            continue
        req = spec["input"]["required"]
        opt = spec["input"].get("optional", {})
        node_errs = []
        for name, definition in req.items():
            if name not in node.get("inputs", {}):
                node_errs.append({"message": "Required input is missing",
                                  "details": name})
        for name, value in node.get("inputs", {}).items():
            if name not in req and name not in opt:
                node_errs.append({"message": "Unexpected input",
                                  "details": name})
                continue
            definition = req.get(name, opt.get(name))
            kind = definition[0]
            if isinstance(value, list) and len(value) == 2 \
                    and isinstance(value[0], str):
                continue  # a link
            if isinstance(kind, list) and value not in kind:
                node_errs.append({"message": "Value not in list",
                                  "details": f"{name}: '{value}' not in {kind}"})
            elif kind == "INT" and not isinstance(value, int):
                node_errs.append({"message": "Value is not an int",
                                  "details": f"{name}: {value!r}"})
            elif kind == "FLOAT" and not isinstance(value, (int, float)):
                node_errs.append({"message": "Value is not a float",
                                  "details": f"{name}: {value!r}"})
            elif kind == "BOOLEAN" and not isinstance(value, bool):
                node_errs.append({"message": "Value is not a bool",
                                  "details": f"{name}: {value!r}"})
            elif kind == "STRING" and not isinstance(value, str):
                node_errs.append({"message": "Value is not a string",
                                  "details": f"{name}: {value!r}"})
        if node_errs:
            errors[nid] = {"class_type": cls, "errors": node_errs}
    return errors


# Behaviour switches the test harness pokes at.
MODE = {"fail_on": None, "delay": 0.4, "vary_rate": False, "force_ext": ""}


@app.post("/mock/mode")
def set_mode():
    MODE.update(request.get_json(silent=True) or {})
    return jsonify(MODE)


@app.get("/mock/log")
def get_log():
    return jsonify({"log": LOG, "prompts": len(HISTORY)})


@app.post("/prompt")
def prompt():
    body = request.get_json(silent=True) or {}
    graph = body.get("prompt") or {}
    errs = validate(graph)
    if errs:
        LOG.append("REJECT " + str(errs)[:200])
        return jsonify({"error": {"type": "prompt_outputs_failed_validation",
                                  "message": "Prompt outputs failed validation",
                                  "details": ""},
                        "node_errors": errs}), 400
    pid = uuid.uuid4().hex
    HISTORY[pid] = {"status": {"status_str": "running", "completed": False,
                               "messages": []}, "outputs": {}}
    text = ""
    save_fmt = "wav"
    shape = []
    for nid in sorted(graph):
        node = graph[nid]
        ins = node.get("inputs", {})
        text = ins.get("text") or ins.get("target_text") or text
        if node["class_type"] == "SaveAudioAdvanced":
            save_fmt = ins.get("format", "wav")
        bits = node["class_type"]
        if "model_choice" in ins:
            bits += f"(model={ins['model_choice']},seed={ins.get('seed')})"
        if "speaker" in ins:
            bits += f"[{ins['speaker']}]"
        if "model_variant" in ins:
            # Logged so a test can prove which MOSS checkpoint was asked for,
            # and whether it was pointed at a local folder or left to fetch.
            local = "local" if ins.get("local_model_path") else "hub"
            bits += f"[{ins['model_variant']}|{local}]"
        shape.append(bits)
    LOG.append(f"QUEUE {' -> '.join(shape)} fmt={save_fmt} text={text[:34]!r}")
    threading.Thread(target=run_prompt, args=(pid, graph, text, save_fmt),
                     daemon=True).start()
    return jsonify({"prompt_id": pid, "number": len(HISTORY),
                    "node_errors": {}})


def run_prompt(pid, graph, text, fmt):
    INTERRUPTED.clear()
    time.sleep(MODE["delay"])
    if INTERRUPTED.is_set():
        HISTORY[pid]["status"] = {"status_str": "error", "completed": False,
                                  "messages": [["execution_interrupted",
                                                {"node_type": "CustomVoiceNode",
                                                 "exception_message":
                                                     "Processing interrupted"}]]}
        return
    if MODE.get("fail_on") and MODE["fail_on"] in text:
        HISTORY[pid]["status"] = {
            "status_str": "error", "completed": False,
            "messages": [["execution_error",
                          {"node_type": "CustomVoiceNode",
                           "exception_message":
                               "CUDA out of memory (mock failure)"}]]}
        return
    ext = MODE.get("force_ext") or fmt
    name = f"ScriptBuilder_{pid[:8]}.{ext}"
    dest = OUT / "audio" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Length follows the text so the joined take has a believable duration.
    # vary_rate makes each clip a different sample rate, which is what
    # stitch_wavs is meant to refuse.
    rate = 24000
    if MODE.get("vary_rate"):
        rate = [24000, 16000, 44100][len(HISTORY) % 3]
    make_wav(dest, max(0.4, min(len(text), 80) * 0.045), 180 + len(text) % 200,
             rate=rate)
    HISTORY[pid]["status"] = {"status_str": "success", "completed": True,
                              "messages": []}
    HISTORY[pid]["outputs"] = {"3": {"audio": [
        {"filename": name, "subfolder": "audio", "type": "output"}]}}


@app.get("/history/<pid>")
def history(pid):
    return jsonify({pid: HISTORY[pid]} if pid in HISTORY else {})


@app.get("/view")
def view():
    name = request.args.get("filename", "")
    sub = request.args.get("subfolder", "")
    kind = request.args.get("type", "output")
    base = OUT if kind == "output" else IN
    path = base / sub / name
    if not path.exists():
        return "not found", 404
    return send_file(path, mimetype="audio/wav")


@app.post("/upload/image")
def upload():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "no file"}), 400
    name = Path(f.filename).name
    (IN).mkdir(parents=True, exist_ok=True)
    f.save(IN / name)
    LOG.append(f"UPLOAD {name}")
    return jsonify({"name": name, "subfolder": "", "type": "input"})


@app.post("/interrupt")
def interrupt():
    INTERRUPTED.set()
    LOG.append("INTERRUPT")
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("MOCK_COMFY_PORT", "8188")),
            threaded=True)
