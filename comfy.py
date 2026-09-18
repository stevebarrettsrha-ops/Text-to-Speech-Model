"""
comfy.py - talks to ComfyUI and builds the Qwen3-TTS graphs.

Graphs are built from ComfyUI's own /object_info rather than a stored workflow.
The custom node renames and adds inputs between releases; reading the schema
turns a rename into a clear message instead of a silently wrong value, and any
new required input picks up its own default.

One line of dialogue is one small graph, picked from how that speaker's voice
is set up:

  preset voice   CustomVoiceNode(text, speaker, instruct) ─► Save
  cloned voice   LoadAudio ─► VoiceCloneNode(ref_audio, ref_text, target_text) ─► Save
  designed voice VoiceDesignNode(text, instruct) ─► Save

Lines are generated one at a time and stitched afterwards, which is what lets
the pause between lines, per-speaker voices and per-line retries work.
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

FALLBACK_SPEAKERS = ["Aiden", "Eric", "Serena"]


class ComfyError(RuntimeError):
    pass


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
                r = requests.get(f"{self.url}/object_info", timeout=30)
                r.raise_for_status()
                self._schema = r.json()
                self._schema_at = time.time()
            return self._schema

    def has(self, class_type: str) -> bool:
        return class_type in self.schema()

    def node_inputs(self, class_type: str) -> dict:
        info = self.schema().get(class_type)
        if not info:
            raise ComfyError(
                f"This ComfyUI has no '{class_type}' node. Install the "
                "Qwen-TTS nodes from the Engine panel, then restart ComfyUI.")
        spec = info.get("input", {})
        merged = {}
        merged.update(spec.get("required", {}) or {})
        merged.update(spec.get("optional", {}) or {})
        return merged

    def ensure_supported(self) -> None:
        if not self.has(CUSTOM) and not self.has(DESIGN) and not self.has(CLONE):
            raise ComfyError(
                "The Qwen-TTS nodes are not loaded in ComfyUI. Install them "
                "from the Engine panel and restart ComfyUI. If they are "
                "installed, check the ComfyUI console for an IMPORT FAILED "
                "line — that is usually a missing requirement.")

    def _enum(self, class_type: str, name: str) -> list[str]:
        try:
            spec = self.node_inputs(class_type).get(name)
        except ComfyError:
            return []
        if spec and isinstance(spec[0], list):
            return [str(v) for v in spec[0]]
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

    def capabilities(self) -> dict:
        return {"preset": self.has(CUSTOM), "clone": self.has(CLONE),
                "design": self.has(DESIGN), "dialogue": self.has(DIALOGUE)}

    # ------------------------------------------------------------------ #
    # output node
    # ------------------------------------------------------------------ #
    def save_node(self, prefer_wav: bool = True) -> tuple[str, str]:
        """(class_type, format). wav keeps the clips stitchable in one file."""
        if self.has("SaveAudioAdvanced"):
            fmts = self._enum("SaveAudioAdvanced", "format")
            if prefer_wav and "wav" in fmts:
                return "SaveAudioAdvanced", "wav"
            return "SaveAudioAdvanced", (fmts[0] if fmts else "flac")
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
                    raise ComfyError(
                        f"{class_type} has no input for '{key}'. This version "
                        "of the Qwen-TTS nodes does not match Script Builder — "
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
        return {"class_type": class_type, "inputs": inputs}

    def build_line(self, line: dict, voice: dict, opts: dict) -> dict:
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
            wanted = dict(common)
            wanted.update({
                "ref_audio": {"names": ["ref_audio", "reference_audio"],
                              "value": ["1", 0], "required": True},
                "ref_text": {"names": ["ref_text", "reference_text"],
                             "value": voice.get("ref_text", "")},
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

        save_class, fmt = self.save_node(prefer_wav=opts.get("prefer_wav", True))
        save_wanted = {
            "audio": {"names": ["audio"], "value": ["2", 0], "required": True},
            "prefix": {"names": ["filename_prefix"], "value": "audio/ScriptBuilder"},
        }
        if save_class == "SaveAudioAdvanced":
            save_wanted["format"] = {"names": ["format"], "value": fmt}
        g["3"] = self._node(save_class, save_wanted)

        return {"prompt": g, "format": fmt}

    # ------------------------------------------------------------------ #
    # queue / results
    # ------------------------------------------------------------------ #
    def queue(self, prompt: dict) -> str:
        body = {"prompt": prompt, "client_id": self.client_id}
        r = requests.post(f"{self.url}/prompt", json=body, timeout=60)
        if r.status_code >= 400:
            try:
                raise ComfyError(_readable(r.json()))
            except ValueError:
                raise ComfyError(r.text[:400])
        return r.json()["prompt_id"]

    def interrupt(self) -> None:
        try:
            requests.post(f"{self.url}/interrupt", timeout=10)
        except Exception:
            pass

    def history(self, prompt_id: str) -> dict:
        r = requests.get(f"{self.url}/history/{prompt_id}", timeout=20)
        r.raise_for_status()
        return r.json().get(prompt_id) or {}

    def outputs(self, prompt_id: str) -> list[dict]:
        hist = self.history(prompt_id)
        found = []
        for node_out in (hist.get("outputs") or {}).values():
            for key in ("audio", "audios", "result"):
                for item in node_out.get(key, []) or []:
                    if isinstance(item, dict) and item.get("filename"):
                        found.append(item)
        return found

    def failed(self, prompt_id: str) -> str | None:
        status = (self.history(prompt_id).get("status") or {})
        if status.get("status_str") == "error":
            for kind, data in status.get("messages", []):
                if kind == "execution_error":
                    return (f"{data.get('node_type')}: "
                            f"{data.get('exception_message')}")
            return "ComfyUI reported an error while generating."
        return None

    def view(self, item: dict):
        params = {"filename": item.get("filename", ""),
                  "subfolder": item.get("subfolder", ""),
                  "type": item.get("type", "output")}
        return requests.get(f"{self.url}/view", params=params, stream=True,
                            timeout=180)

    def upload_audio(self, file_storage) -> str:
        files = {"image": (file_storage.filename, file_storage.stream,
                           file_storage.mimetype or "audio/wav")}
        r = requests.post(f"{self.url}/upload/image", files=files,
                          data={"type": "input", "overwrite": "true"},
                          timeout=180)
        r.raise_for_status()
        data = r.json()
        name = data.get("name") or file_storage.filename
        sub = data.get("subfolder") or ""
        return f"{sub}/{name}" if sub else name


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
