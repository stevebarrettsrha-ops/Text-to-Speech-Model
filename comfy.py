"""
comfy.py - talks to ComfyUI and builds the Qwen3-TTS and MOSS-TTS graphs.

Graphs are built from ComfyUI's own /object_info rather than a stored workflow.
The custom node renames and adds inputs between releases; reading the schema
turns a rename into a clear message instead of a silently wrong value, and any
new required input picks up its own default.

One line of dialogue is one small graph, picked from how that speaker's voice
is set up:

  preset voice   CustomVoiceNode(text, speaker, instruct) ─► Save
  cloned voice   LoadAudio ─► VoiceCloneNode(ref_audio, ref_text, target_text) ─► Save
  designed voice VoiceDesignNode(text, instruct) ─► Save

  (Python class names — ComfyUI knows them as FB_Qwen3TTSCustomVoice etc.)

Lines are generated one at a time and stitched afterwards, which is what lets
the pause between lines, per-speaker voices and per-line retries work.

MOSS-TTS is the same idea in two nodes rather than one, because its weights are
loaded by a separate node that hands a MOSS_TTS_PIPE to the generator:

  model's voice  Loader ─► MossTTSGenerate(text) ─► Save
  cloned voice   Loader + LoadAudio ─► MossTTSGenerate(text, reference) ─► Save
  designed voice Loader ─► MossTTSVoiceDesign(text, instruction) ─► Save

It has no preset speakers at all — the voice comes from a reference clip, from
a written description, or from the model itself.
"""

from __future__ import annotations

import json
import random
import threading
import time
import uuid

import requests

CUSTOM = "CustomVoiceNode"
CLONE = "VoiceCloneNode"
DESIGN = "VoiceDesignNode"
CLONE_PROMPT = "VoiceClonePromptNode"
DIALOGUE = "DialogueInferenceNode"

# The names above are roles, not what ComfyUI calls the nodes. ComfyUI
# registers a node under its NODE_CLASS_MAPPINGS key, and flybirdxx's pack
# keys them "FB_Qwen3TTSCustomVoice" and so on — the Python class names above
# never reach /object_info. Asking for them directly found no Qwen node on any
# real install, so the engine never read as ready. Each role is resolved
# against the schema through these lists, newest name first; rule 2's
# candidate lists, one level up.
QWEN_CLASS_NAMES = {
    CUSTOM: ["FB_Qwen3TTSCustomVoice", "Qwen3TTSCustomVoice", CUSTOM],
    CLONE: ["FB_Qwen3TTSVoiceClone", "Qwen3TTSVoiceClone", CLONE],
    DESIGN: ["FB_Qwen3TTSVoiceDesign", "Qwen3TTSVoiceDesign", DESIGN],
    CLONE_PROMPT: ["FB_Qwen3TTSVoiceClonePrompt", "Qwen3TTSVoiceClonePrompt",
                   CLONE_PROMPT],
    DIALOGUE: ["FB_Qwen3TTSDialogueInference", "Qwen3TTSDialogueInference",
               DIALOGUE],
}

MOSS_LOADER = "MossTTSModelLoader"
MOSS_GEN = "MossTTSGenerate"
MOSS_DESIGN = "MossTTSVoiceDesign"
MOSS_DIALOGUE = "MossTTSDialogue"

# The loader's model_variant is an enum of display names — "MOSS-TTS (Local
# 1.7B)" and friends — and the repo id each one maps to lives in the node's
# constants, not in /object_info. So the entry is picked by substring, the same
# way the Qwen design branch picks 1.7B out of its own enum, rather than typed
# in here where a rename would go unnoticed.
MOSS_VARIANT_HINTS = {
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer": ["local", "1.7"],
    "OpenMOSS-Team/MOSS-TTS": ["delay", "8b"],
    "OpenMOSS-Team/MOSS-VoiceGenerator": ["voicegenerator", "voice generator"],
}

MOSS_DEFAULT_MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
MOSS_VOICE_GENERATOR = "OpenMOSS-Team/MOSS-VoiceGenerator"
MOSS_CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

# What OpenMOSS tuned each checkpoint to sample with: the node's own
# utils/constants.py DEFAULT_PARAMS — keep it in step with that file, as
# MOSS_MODEL_REPOS is. The node publishes the table and never applies it:
# every MossTTSGenerate input defaults to the Delay 8B's numbers whatever the
# loader holds, so leaving them to the schema ran the Local 1.7B — the model
# this app loads by default — with no repetition penalty and half its top_k,
# the settings the node's own README says to change for that model. None of
# this reaches /object_info, which is why it is written down here at all.
MOSS_SAMPLING = {
    "OpenMOSS-Team/MOSS-TTS": {
        "temperature": 1.7, "top_p": 0.8, "top_k": 25,
        "repetition_penalty": 1.0},
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer": {
        "temperature": 1.0, "top_p": 0.95, "top_k": 50,
        "repetition_penalty": 1.1},
    "OpenMOSS-Team/MOSS-VoiceGenerator": {
        "temperature": 1.5, "top_p": 0.6, "top_k": 50,
        "repetition_penalty": 1.1},
}

# Where the page's Expressiveness slider rests — Qwen's own default
# temperature. On MOSS it scales the model's tuned temperature rather than
# replacing it: 0.9 means "as OpenMOSS tuned it", which for the 8B is 1.7 and
# for VoiceGenerator 1.5, so passing 0.9 through unchanged cooled every MOSS
# model below the range it was trained for.
NEUTRAL_TEMPERATURE = 0.9

FALLBACK_SPEAKERS = ["Aiden", "Eric", "Serena"]


class ComfyError(RuntimeError):
    pass


OFFLINE = ("ComfyUI stopped answering at {url}. It may have crashed or been "
           "closed — check its console, then start it again from the Engine "
           "panel.")


def root_from_argv(argv) -> str:
    """The ComfyUI folder an argv list was launched from, or "".

    /system_stats reports the process's own argv, which starts with the
    main.py it was started from. Empty when it cannot be told: older builds do
    not report argv at all, and that must never read as "someone else's".
    """
    for arg in argv or []:
        if isinstance(arg, str) and arg.lower().endswith("main.py"):
            # Both separators appear: a Windows path read on any platform.
            cut = max(arg.rfind("/"), arg.rfind("\\"))
            return arg[:cut] if cut > 0 else ""
    return ""


def _reach(fn, url: str):
    """Run a request, and turn a dead engine into a sentence.

    Everything else this app says is plain English; a socket error read as
    "ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8188): Max
    retries exceeded" in the middle of a take, which tells nobody what to do.
    """
    try:
        return fn()
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise ComfyError(OFFLINE.format(url=url)) from exc


class ComfyClient:
    def __init__(self, url: str = "http://127.0.0.1:8188") -> None:
        self.url = url.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self._schema: dict | None = None
        self._schema_at = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # schema
    # ------------------------------------------------------------------ #
    def schema(self, force: bool = False) -> dict:
        with self._lock:
            if force or self._schema is None or time.time() - self._schema_at > 120:
                r = _reach(lambda: requests.get(f"{self.url}/object_info",
                                                timeout=30), self.url)
                r.raise_for_status()
                self._schema = r.json()
                self._schema_at = time.time()
            return self._schema

    def real(self, class_type: str) -> str:
        """The name this ComfyUI registered a role under (see QWEN_CLASS_NAMES)."""
        names = QWEN_CLASS_NAMES.get(class_type)
        if not names:
            return class_type
        schema = self.schema()
        return next((n for n in names if n in schema), class_type)

    def has(self, class_type: str) -> bool:
        return self.real(class_type) in self.schema()

    def vram_mb(self) -> int:
        """What ComfyUI says the card has, as a second opinion to nvidia-smi.

        A portable ComfyUI carries its own CUDA and will happily report the
        card on a machine where nvidia-smi is not on PATH — the same gap rule
        5b is about. Bytes here, megabytes out; 0 when nothing can be read.
        """
        try:
            r = _reach(lambda: requests.get(f"{self.url}/system_stats",
                                           timeout=10), self.url)
            r.raise_for_status()
            best = 0
            for dev in (r.json().get("devices") or []):
                total = dev.get("vram_total") or 0
                if isinstance(total, (int, float)) and total > best:
                    best = int(total)
            # ComfyUI reports bytes; anything smaller is already megabytes.
            return best // (1024 * 1024) if best > 1 << 20 else int(best)
        except Exception:  # noqa: BLE001
            return 0

    def engine_root(self) -> str:
        """The folder of the ComfyUI actually answering this address.

        /system_stats reports the process's own argv, which starts with the
        main.py it was launched from. Comparing that against the install being
        managed is the only way to notice that the port is held by a
        *different* ComfyUI — which looks exactly like nodes that failed to
        load, because the install we manage has them and the engine answering
        has none of them. Two engines make it likelier still: 8188 is the port
        every ComfyUI picks by default.

        Empty when it cannot be told: older builds do not report argv.
        """
        try:
            r = _reach(lambda: requests.get(f"{self.url}/system_stats",
                                            timeout=10), self.url)
            r.raise_for_status()
            argv = ((r.json() or {}).get("system") or {}).get("argv") or []
        except Exception:  # noqa: BLE001
            return ""
        return root_from_argv(argv)

    def node_inputs(self, class_type: str) -> dict:
        info = self.schema().get(self.real(class_type))
        if not info:
            kit = "MOSS-TTS" if class_type.startswith("Moss") else "Qwen-TTS"
            raise ComfyError(
                f"This ComfyUI has no '{class_type}' node. Install the "
                f"{kit} nodes from the Engine panel, then restart ComfyUI.")
        spec = info.get("input", {})
        merged = {}
        merged.update(spec.get("required", {}) or {})
        merged.update(spec.get("optional", {}) or {})
        return merged

    def ensure_supported(self, engine: str = "qwen") -> None:
        if engine == "moss":
            if self.has(MOSS_LOADER) and (self.has(MOSS_GEN)
                                          or self.has(MOSS_DESIGN)):
                return
            raise ComfyError(
                "The MOSS-TTS nodes are not loaded in ComfyUI. Install them "
                "from the Engine panel and restart ComfyUI — or switch the "
                "engine back to Qwen3-TTS on the Create page.")
        if not self.has(CUSTOM) and not self.has(DESIGN) and not self.has(CLONE):
            raise ComfyError(
                "The Qwen-TTS nodes are not loaded in ComfyUI. Install them "
                "from the Engine panel and restart ComfyUI. If they are "
                "installed, check the ComfyUI console for an IMPORT FAILED "
                "line — that is usually a missing requirement.")

    def engine_ready(self, engine: str) -> bool:
        try:
            self.ensure_supported(engine)
            return True
        except ComfyError:
            return False

    def _enum(self, class_type: str, name: str) -> list[str]:
        try:
            spec = self.node_inputs(class_type).get(name)
        except ComfyError:
            return []
        if spec and isinstance(spec[0], list):
            return [str(v) for v in spec[0]]
        # ComfyUI's V3 nodes publish some choices as a DynamicCombo:
        # ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "flac", ...}]}], and
        # the prompt takes the key as a plain string. SaveAudioAdvanced's
        # format is one, so reading only plain lists found no formats at all.
        if spec and len(spec) > 1 and isinstance(spec[1], dict) \
                and isinstance(spec[1].get("options"), list):
            return [str(o.get("key")) for o in spec[1]["options"]
                    if isinstance(o, dict) and o.get("key")]
        return []

    def speakers(self) -> list[str]:
        for name in ("speaker", "speaker_name", "voice"):
            vals = self._enum(CUSTOM, name)
            if vals:
                return vals
        return []

    def models(self) -> list[str]:
        for cls in (CLONE, DESIGN, CUSTOM):
            vals = self._enum(cls, "model_choice")
            if vals:
                return vals
        return []

    def attentions(self) -> list[str]:
        for cls in (CUSTOM, CLONE, DESIGN):
            vals = self._enum(cls, "attention")
            if vals:
                return vals
        return []

    def capabilities(self, engine: str = "qwen") -> dict:
        if engine == "moss":
            # No preset key at all would read as "unknown" in the page; MOSS
            # genuinely has no speaker list, and saying so is the point.
            return {"preset": False, "clone": self.has(MOSS_GEN),
                    "design": self.has(MOSS_DESIGN),
                    "dialogue": self.has(MOSS_DIALOGUE),
                    "own_voice": self.has(MOSS_GEN)}
        return {"preset": self.has(CUSTOM), "clone": self.has(CLONE),
                "design": self.has(DESIGN), "dialogue": self.has(DIALOGUE),
                "own_voice": False}

    def moss_variants(self) -> list[str]:
        return self._enum(MOSS_LOADER, "model_variant")

    def model_list(self, engine: str) -> list[str]:
        """The engine's own account of which checkpoints it can load.

        MOSS names them: MossTTSModelLoader.model_variant is a list of MOSS
        checkpoint display names, so "moss" being absent from it means the
        engine answering cannot load a MOSS model at all. Qwen's enums name
        sizes ("0.6B") and preset speakers ("Ryan") and never a model, which
        is why ENGINES["qwen"] declares no model_marker and this is empty for
        it — an empty list with no marker to match is not evidence of
        anything, and `stale_engine` treats it as none.
        """
        return self.moss_variants() if engine == "moss" else []

    def moss_variant_for(self, repo: str) -> str:
        """The loader enum entry that means `repo`, read off the node.

        Empty when nothing matches, and the caller then leaves model_variant on
        its own default rather than sending a value the node would reject.
        """
        hints = MOSS_VARIANT_HINTS.get(repo) or [repo.split("/")[-1].lower()]
        variants = self.moss_variants()
        for v in variants:
            low = v.lower()
            if all(h in low for h in hints):
                return v
        for v in variants:
            low = v.lower()
            if any(h in low for h in hints):
                return v
        return ""

    # ------------------------------------------------------------------ #
    # output node
    # ------------------------------------------------------------------ #
    def save_node(self, prefer_wav: bool = True) -> tuple[str, str]:
        """(class_type, format). wav keeps the clips stitchable in one file."""
        if self.has("SaveAudioAdvanced"):
            fmts = self._enum("SaveAudioAdvanced", "format")
            if prefer_wav and "wav" in fmts:
                return "SaveAudioAdvanced", "wav"
            # Current ComfyUI offers no wav at all — flac, mp3 and opus. Flac
            # is lossless, so the server can turn it back into wav and join
            # the take (see server.flac_to_wav).
            if "flac" in fmts or not fmts:
                return "SaveAudioAdvanced", "flac"
            return "SaveAudioAdvanced", fmts[0]
        if self.has("SaveAudio"):
            return "SaveAudio", "flac"
        raise ComfyError("ComfyUI has no audio save node. Update ComfyUI.")

    # ------------------------------------------------------------------ #
    # graph building
    # ------------------------------------------------------------------ #
    @staticmethod
    def _match(available: dict, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in available:
                return c
        low = {k.lower(): k for k in available}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

    def _node(self, class_type: str, wanted: dict) -> dict:
        spec = self.node_inputs(class_type)
        inputs: dict = {}
        for key, want in wanted.items():
            name = self._match(spec, want["names"])
            if name is None:
                if want.get("required"):
                    kit = "MOSS-TTS" if class_type.startswith("Moss") \
                        else "Qwen-TTS"
                    raise ComfyError(
                        f"{class_type} has no input for '{key}'. This version "
                        f"of the {kit} nodes does not match Script Builder — "
                        "update it from the Engine panel.")
                continue
            inputs[name] = want["value"]

        for name, definition in spec.items():
            if name in inputs or name == "control_after_generate":
                continue
            if not isinstance(definition, (list, tuple)) or not definition:
                continue
            kind = definition[0]
            opts = definition[1] if len(definition) > 1 else {}
            if not isinstance(opts, dict):
                opts = {}
            if isinstance(kind, list):
                inputs[name] = opts.get("default", kind[0] if kind else "")
            elif kind in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                if "default" in opts:
                    inputs[name] = opts["default"]
                elif kind == "STRING":
                    inputs[name] = ""
        return {"class_type": self.real(class_type), "inputs": inputs}

    def _save(self, g: dict, source: str, opts: dict) -> str:
        save_class, fmt = self.save_node(prefer_wav=opts.get("prefer_wav", True))
        wanted = {
            "audio": {"names": ["audio"], "value": [source, 0], "required": True},
            "prefix": {"names": ["filename_prefix"],
                       "value": "audio/ScriptBuilder"},
        }
        if save_class == "SaveAudioAdvanced":
            wanted["format"] = {"names": ["format"], "value": fmt}
        g["3"] = self._node(save_class, wanted)
        return fmt

    @staticmethod
    def line_weights(voice: dict, opts: dict) -> tuple:
        """Which checkpoint a line makes the engine hold.

        Both node packs keep exactly one model resident: Qwen's
        `load_qwen_model` clears its cache before loading a different one, and
        MOSS's loader moves the last model off the card first. So a script
        alternating a preset speaker with a cloned one reloads a checkpoint
        from disk on every line. `run_job` groups lines by this key, and this
        mirrors the choices the two builders below make.
        """
        kind = voice.get("kind") or "preset"
        if (opts.get("engine") or "qwen") == "moss":
            if kind == "design":
                return ("moss", MOSS_VOICE_GENERATOR)
            return ("moss", opts.get("moss_model") or MOSS_DEFAULT_MODEL)
        if kind == "design":
            return ("qwen", DESIGN, "1.7B")
        node = CLONE if kind == "clone" else CUSTOM
        return ("qwen", node, opts.get("model") or "")

    def build_line(self, line: dict, voice: dict, opts: dict) -> dict:
        """One line of dialogue → one prompt graph, on whichever engine."""
        if (opts.get("engine") or "qwen") == "moss":
            return self.build_moss_line(line, voice, opts)
        return self.build_qwen_line(line, voice, opts)

    def build_moss_line(self, line: dict, voice: dict, opts: dict) -> dict:
        """One line → loader + generator + save.

        MOSS splits what Qwen does in one node across two: the loader holds the
        weights and hands a MOSS_TTS_PIPE downstream. Everything about which
        checkpoint that is comes off the node's own enum.
        """
        self.ensure_supported("moss")
        text = (line.get("text") or "").strip()
        if not text:
            raise ComfyError("There is an empty line in the script.")

        kind = voice.get("kind") or "preset"
        style = (opts.get("style") or "").strip()
        instruct = (voice.get("instruct") or "").strip() or style
        dirs = opts.get("moss_dirs") or {}
        repo = opts.get("moss_model") or MOSS_DEFAULT_MODEL

        if kind == "design":
            if not self.has(MOSS_DESIGN):
                raise ComfyError("This ComfyUI has no MossTTSVoiceDesign node, "
                                 "so a described voice cannot be used on "
                                 "MOSS-TTS.")
            # The node itself warns when it is handed anything else: only
            # MOSS-VoiceGenerator was trained to build a voice from a
            # description, so the loader is pointed at it whatever the model
            # picker says. Same reasoning as Qwen's VoiceDesign forcing 1.7B.
            repo = MOSS_VOICE_GENERATOR

        g: dict = {}
        variant = self.moss_variant_for(repo)
        loader = {}
        if variant:
            loader["variant"] = {"names": ["model_variant"], "value": variant}
        # Only ever a folder that is really there. The loader treats a path it
        # cannot stat as a HuggingFace repo id and calls snapshot_download on
        # it, which fails on an absolute path — so a missing folder has to come
        # through as "", which lets the node fetch the model itself.
        local = dirs.get(repo) or ""
        loader["local"] = {"names": ["local_model_path"], "value": local}
        codec = dirs.get(MOSS_CODEC) or ""
        loader["codec"] = {"names": ["codec_local_path"], "value": codec}
        g["1"] = self._node(MOSS_LOADER, loader)

        gen_class = MOSS_DESIGN if kind == "design" else MOSS_GEN
        wanted = {
            "pipe": {"names": ["moss_pipe"], "value": ["1", 0], "required": True},
            "text": {"names": ["text", "target_text"], "value": text,
                     "required": True},
            # Without this the schema default of 0 is filled in for every line
            # and a retry gives back exactly what it gave before.
            "seed": {"names": ["seed", "noise_seed"],
                     "value": random.randint(0, 2 ** 31 - 1)},
        }
        for name, value in moss_sampling(repo, opts).items():
            wanted[name] = {"names": [name], "value": value}
        if opts.get("language"):
            wanted["language"] = {"names": ["language"],
                                  "value": opts["language"]}

        if kind == "design":
            wanted["instruct"] = {
                "names": ["instruction", "instruct", "description"],
                "value": instruct or "A clear, natural voice", "required": True}
        elif kind == "clone":
            ref = voice.get("ref_audio")
            if not ref:
                raise ComfyError("That speaker is set to a cloned voice but has "
                                 "no reference audio loaded.")
            g["2"] = self._node("LoadAudio",
                                {"audio": {"names": ["audio"], "value": ref,
                                           "required": True}})
            wanted["reference"] = {"names": ["reference_audio", "ref_audio"],
                                   "value": ["2", 0], "required": True}
        # kind == "preset" falls through with no reference and no instruction:
        # MOSS has no speaker list, and the base model then speaks in a voice
        # of its own, which is what the page offers as "the model's own voice".

        g["4"] = self._node(gen_class, wanted)
        fmt = self._save(g, "4", opts)
        return {"prompt": g, "format": fmt}

    def build_qwen_line(self, line: dict, voice: dict, opts: dict) -> dict:
        """One line of dialogue → one prompt graph.

        line  : {"text": str}
        voice : {"kind": preset|clone|design, "speaker": str,
                 "instruct": str, "ref_audio": str, "ref_text": str}
        opts  : {"model", "attention", "unload", "style",
                 "temperature", "top_p", "format"}
        """
        self.ensure_supported()
        text = (line.get("text") or "").strip()
        if not text:
            raise ComfyError("There is an empty line in the script.")

        kind = voice.get("kind") or "preset"
        style = (opts.get("style") or "").strip()
        instruct = (voice.get("instruct") or "").strip() or style
        model = opts.get("model") or ""
        attention = opts.get("attention") or "auto"
        unload = bool(opts.get("unload"))

        common = {
            # Without this the schema default of 0 is filled in for every line,
            # so a take is byte-identical to the last one and retrying a line
            # gives back exactly what it gave before.
            "seed": {"names": ["seed", "noise_seed"],
                     "value": random.randint(0, 2 ** 31 - 1)},
            "attention": {"names": ["attention"], "value": attention},
            "unload": {"names": ["unload_model_after_generate", "unload_model"],
                       "value": unload},
        }
        if model:
            common["model"] = {"names": ["model_choice", "model"], "value": model}
        if opts.get("temperature") is not None:
            common["temperature"] = {"names": ["temperature"],
                                     "value": float(opts["temperature"])}
        if opts.get("top_p") is not None:
            common["top_p"] = {"names": ["top_p"], "value": float(opts["top_p"])}

        g: dict = {}

        if kind == "clone":
            if not self.has(CLONE):
                raise ComfyError("This ComfyUI has no VoiceCloneNode, so a "
                                 "cloned voice cannot be used.")
            ref = voice.get("ref_audio")
            if not ref:
                raise ComfyError("That speaker is set to a cloned voice but has "
                                 "no reference audio loaded.")
            g["1"] = self._node("LoadAudio",
                                {"audio": {"names": ["audio"], "value": ref,
                                           "required": True}})
            ref_text = (voice.get("ref_text") or "").strip()
            wanted = dict(common)
            wanted.update({
                "ref_audio": {"names": ["ref_audio", "reference_audio"],
                              "value": ["1", 0], "required": True},
                "ref_text": {"names": ["ref_text", "reference_text"],
                             "value": ref_text},
                # Without a transcript the node's default mode refuses the
                # line outright — "ref_text is required when
                # x_vector_only_mode=False (ICL mode)" — and the page calls the
                # transcript optional. The speaker embedding alone still copies
                # the voice, less closely, so that is what an empty box asks
                # for.
                "x_vector_only": {"names": ["x_vector_only",
                                            "x_vector_only_mode"],
                                  "value": not ref_text},
                "text": {"names": ["target_text", "text"], "value": text,
                         "required": True},
            })
            g["2"] = self._node(CLONE, wanted)
        elif kind == "design":
            if not self.has(DESIGN):
                raise ComfyError("This ComfyUI has no VoiceDesignNode, so a "
                                 "described voice cannot be used.")
            wanted = dict(common)
            wanted.update({
                "text": {"names": ["text", "target_text"], "value": text,
                         "required": True},
                "instruct": {"names": ["instruct", "description", "instruction"],
                             "value": instruct or "A clear, natural voice",
                             "required": True},
            })
            # VoiceDesign only ships as 1.7B: the node raises outright on 0.6B,
            # and 1.7B-VoiceDesign is the only folder setup fetches for it. The
            # model picker offers 0.6B because cloning has one, so a designed
            # voice would fail on the picker's own first entry. Which value
            # means 1.7B is read off the node, never typed in here.
            big = next((c for c in self._enum(DESIGN, "model_choice")
                        if "1.7" in c), "")
            if big:
                wanted["model"] = {"names": ["model_choice", "model"],
                                   "value": big}
            g["2"] = self._node(DESIGN, wanted)
        else:
            if not self.has(CUSTOM):
                raise ComfyError("This ComfyUI has no CustomVoiceNode.")
            speaker = voice.get("speaker") or ""
            available = self.speakers()
            if available and speaker not in available:
                speaker = available[0]
            wanted = dict(common)
            wanted.update({
                "text": {"names": ["text", "target_text"], "value": text,
                         "required": True},
                "speaker": {"names": ["speaker", "speaker_name", "voice"],
                            "value": speaker, "required": True},
                "instruct": {"names": ["instruct", "instruction", "style"],
                             "value": instruct},
            })
            g["2"] = self._node(CUSTOM, wanted)

        fmt = self._save(g, "2", opts)
        return {"prompt": g, "format": fmt}

    # ------------------------------------------------------------------ #
    # queue / results
    # ------------------------------------------------------------------ #
    def queue(self, prompt: dict) -> str:
        body = {"prompt": prompt, "client_id": self.client_id}
        r = _reach(lambda: requests.post(f"{self.url}/prompt", json=body,
                                        timeout=60), self.url)
        if r.status_code >= 400:
            try:
                raise ComfyError(_readable(r.json()))
            except ValueError:
                raise ComfyError(r.text[:400])
        return r.json()["prompt_id"]

    def interrupt(self, prompt_id: str = "") -> None:
        """Stop a prompt. Given its id, only that one — ComfyUI skips the
        interrupt when something else is running — and it is also taken out
        of the queue if it had not started. Without an id, whatever runs."""
        try:
            if prompt_id:
                requests.post(f"{self.url}/interrupt",
                              json={"prompt_id": prompt_id}, timeout=10)
                requests.post(f"{self.url}/queue",
                              json={"delete": [prompt_id]}, timeout=10)
            else:
                requests.post(f"{self.url}/interrupt", timeout=10)
        except Exception:
            pass

    def history(self, prompt_id: str) -> dict:
        r = _reach(lambda: requests.get(f"{self.url}/history/{prompt_id}",
                                       timeout=20), self.url)
        r.raise_for_status()
        return r.json().get(prompt_id) or {}

    @staticmethod
    def _audio(hist: dict) -> list[dict]:
        found = []
        for node_out in (hist.get("outputs") or {}).values():
            for key in ("audio", "audios", "result"):
                for item in node_out.get(key, []) or []:
                    if isinstance(item, dict) and item.get("filename"):
                        found.append(item)
        return found

    @staticmethod
    def _error(hist: dict) -> str | None:
        status = (hist.get("status") or {})
        if status.get("status_str") == "error":
            for kind, data in status.get("messages", []):
                if kind == "execution_error":
                    return (f"{data.get('node_type')}: "
                            f"{data.get('exception_message')}")
            return "ComfyUI reported an error while generating."
        return None

    def outputs(self, prompt_id: str) -> list[dict]:
        return self._audio(self.history(prompt_id))

    def failed(self, prompt_id: str) -> str | None:
        return self._error(self.history(prompt_id))

    def result(self, prompt_id: str) -> tuple[list[dict], str | None]:
        """(audio, error) from one read of the history.

        A prompt ComfyUI calls finished with no audio in it is an error now:
        it used to be waited on for the full fifteen minutes, since nothing
        was ever going to arrive.
        """
        hist = self.history(prompt_id)
        err = self._error(hist)
        if err:
            return [], err
        outs = self._audio(hist)
        if not outs and (hist.get("status") or {}).get("completed"):
            return [], ("ComfyUI finished the line but saved no audio. "
                        "Check the engine's console for a warning.")
        return outs, None

    def view(self, item: dict):
        params = {"filename": item.get("filename", ""),
                  "subfolder": item.get("subfolder", ""),
                  "type": item.get("type", "output")}
        return _reach(lambda: requests.get(f"{self.url}/view", params=params,
                                          stream=True, timeout=180), self.url)

    def upload_bytes(self, name: str, data: bytes, mimetype: str) -> str:
        files = {"image": (name, data, mimetype or "audio/wav")}
        r = _reach(lambda: requests.post(
            f"{self.url}/upload/image", files=files,
            data={"type": "input", "overwrite": "true"}, timeout=180),
            self.url)
        r.raise_for_status()
        data = r.json()
        got = data.get("name") or name
        sub = data.get("subfolder") or ""
        return f"{sub}/{got}" if sub else got

    def upload_audio(self, file_storage) -> str:
        files = {"image": (file_storage.filename, file_storage.stream,
                           file_storage.mimetype or "audio/wav")}
        r = _reach(lambda: requests.post(
            f"{self.url}/upload/image", files=files,
            data={"type": "input", "overwrite": "true"}, timeout=180),
            self.url)
        r.raise_for_status()
        data = r.json()
        name = data.get("name") or file_storage.filename
        sub = data.get("subfolder") or ""
        return f"{sub}/{name}" if sub else name


def moss_sampling(repo: str, opts: dict) -> dict:
    """temperature, top_p, top_k and repetition_penalty for one MOSS line.

    The checkpoint's own tuning, with the Expressiveness slider as a scale on
    its temperature. A repo the table does not know gets the default model's
    numbers, which are the conservative ones.
    """
    tuned = dict(MOSS_SAMPLING.get(repo) or MOSS_SAMPLING[MOSS_DEFAULT_MODEL])
    if opts.get("temperature") is not None:
        scale = float(opts["temperature"]) / NEUTRAL_TEMPERATURE
        tuned["temperature"] = round(tuned["temperature"] * scale, 3)
    if opts.get("top_p") is not None:
        tuned["top_p"] = float(opts["top_p"])
    return tuned


def _readable(err: dict) -> str:
    for node_id, info in (err.get("node_errors") or {}).items():
        for e in info.get("errors", []):
            return (f"{info.get('class_type', 'node ' + str(node_id))}: "
                    f"{e.get('message')} {e.get('details', '')}".strip())
    top = err.get("error") or {}
    if top:
        return f"{top.get('message', 'Rejected by ComfyUI')} " \
               f"{top.get('details', '')}".strip()
    return json.dumps(err)[:300]
