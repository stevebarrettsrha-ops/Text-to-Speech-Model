"""
bootstrap.py - first-launch setup for Script Builder.

Steps, in order:
  1. Find a real Python 3.10+ (tested by execution, not PATH lookup).
  2. Find an existing ComfyUI, or clone a managed one into ./ComfyUI.
  3. Clone ComfyUI-Qwen-TTS into ComfyUI/custom_nodes and install its
     requirements with the interpreter that ComfyUI itself runs on — the
     portable python_embeded when that is what is there, otherwise the venv.
  4. Download the Qwen3-TTS model folders from HuggingFace into
     ComfyUI/models/qwen-tts/Qwen/.
  5. Start ComfyUI headless and wait for /system_stats.

Everything long runs on a worker thread and reports into a Progress object the
page polls.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

APP_DIR = Path(__file__).resolve().parent
# Config, the library and the finished takes. SCRIPT_BUILDER_DATA moves the
# lot, which lets a second copy run without touching the first one's takes —
# the test suite relies on it, and so does anyone keeping their library on
# another disk.
DATA_DIR = Path(os.environ.get("SCRIPT_BUILDER_DATA") or (APP_DIR / "data"))
CONFIG_PATH = DATA_DIR / "config.json"

COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
NODE_REPO = "https://github.com/flybirdxx/ComfyUI-Qwen-TTS.git"
NODE_DIR_NAME = "ComfyUI-Qwen-TTS"
MOSS_NODE_REPO = "https://github.com/richservo/comfyui-moss-tts.git"
MOSS_NODE_DIR_NAME = "comfyui-moss-tts"

HF_BASE = "https://huggingface.co"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"


def clean_url(url) -> str:
    """A base URL fit to build requests on, or "" if it cannot be made into one.

    No stray whitespace and no trailing slash — appending /system_stats to
    "http://host:8188/" asks for //system_stats, which is a 404, not a health
    check. A bare "localhost:8188", which is what people type, gains the scheme
    it is missing; anything with no host at all comes back empty so the caller
    can keep whatever address was already working.
    """
    text = url.strip().rstrip("/") if isinstance(url, str) else ""
    if not text:
        return ""
    if "://" not in text:
        text = "http://" + text
    parts = urlsplit(text)
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or any(ch.isspace() for ch in parts.netloc)):
        return ""
    return text


def comfy_port(url: str) -> int:
    """The port to start ComfyUI on, read out of its URL.

    This used to be int(url.rsplit(":")[-1]), which blew up on a trailing slash
    or a port-less address — typed once into Settings, that config stopped the
    server from booting at all.
    """
    try:
        port = urlsplit(clean_url(url) or DEFAULT_COMFY_URL).port
    except ValueError:
        port = None
    return port or 8188

# The Qwen3-TTS collection on HuggingFace. The custom node looks for these
# under ComfyUI/models/qwen-tts/Qwen/<folder>.
#
# Which checkpoint serves which node: the CustomVoice weights carry the preset
# speakers, the Base weights do zero-shot cloning, VoiceDesign builds a voice
# from a description, and the tokenizer is needed by all of them. The node's
# README only lists Base + VoiceDesign, so the CustomVoice mapping is inferred
# from the model names — if a preset voice errors, download the matching Base
# folder from the Models page and it will be found.
MODEL_REPOS = [
    {"repo": "Qwen/Qwen3-TTS-Tokenizer-12Hz", "group": "core", "params": "0.2B",
     "note": "Speech tokenizer. Nothing generates without it."},
    {"repo": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "group": "preset",
     "params": "0.9B", "note": "Preset speakers, fast. The default voice source."},
    {"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "group": "preset_hq",
     "params": "2B", "note": "Preset speakers at higher quality."},
    {"repo": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "group": "clone", "params": "0.9B",
     "note": "Zero-shot voice cloning from a reference clip, fast."},
    {"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "group": "clone_hq", "params": "2B",
     "note": "Voice cloning at higher quality."},
    {"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "group": "design",
     "params": "2B", "note": "Builds a voice from a written description."},
]

# group -> the config flag that asks for it. core and preset are always needed.
GROUP_FLAG = {"core": None, "preset": None, "clone": "want_clone",
              "clone_hq": "want_17b", "preset_hq": "want_17b",
              "design": "want_voicedesign",
              "moss_core": None, "moss_hq": "want_moss_8b",
              "moss_design": "want_moss_design"}

# MOSS-TTS. The repo ids are the ones in the node's own
# utils/constants.py MODEL_VARIANTS — keep them in step with that file, not
# with its README, exactly as MODEL_REPOS tracks the Qwen node's HF_MODEL_MAP.
#
# Sizes come from OpenMOSS's own model table, NOT from the ComfyUI node's
# README: that README lists MOSS-VoiceGenerator as "Delay 8B, ~18 GB", which
# conflates the architecture with the size. `MossTTSDelay` is the
# architecture; OpenMOSS publishes VoiceGenerator at 1.7B. Believing the node
# README put voice design behind a warning that it would not run on an 8 GB
# card, when it fits about as comfortably as the Local 1.7B does.
#
# The genuinely 8B checkpoints do want roughly 18 GB through this node, which
# loads bf16 weights with AutoModel.from_pretrained. OpenMOSS's own llama.cpp
# path fits 8B on an 8 GB card with Q4_K_M weights and staged loading, but the
# ComfyUI node implements none of that — no GGUF, no ONNX, no low-memory mode
# — so through Script Builder the 8B stays a tick rather than a default.
MOSS_MODEL_REPOS = [
    {"repo": "OpenMOSS-Team/MOSS-Audio-Tokenizer", "group": "moss_core",
     "params": "codec",
     "note": "Shared audio codec. Every MOSS model needs it."},
    {"repo": "OpenMOSS-Team/MOSS-TTS-Local-Transformer", "group": "moss_core",
     "params": "1.7B",
     "note": "Speech and zero-shot cloning. ~5 GB of VRAM, and the fast one."},
    {"repo": "OpenMOSS-Team/MOSS-VoiceGenerator", "group": "moss_design",
     "params": "1.7B",
     "note": "Builds a voice from a description. ~5 GB of VRAM."},
    {"repo": "OpenMOSS-Team/MOSS-TTS", "group": "moss_hq", "params": "8B",
     "note": "Delay 8B — better, far slower, and ~18 GB of VRAM through "
             "this node."},
]

MOSS_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

DEFAULT_CONFIG = {
    "comfy_url": DEFAULT_COMFY_URL,
    "comfy_dir": "",
    "models_dir": "",        # ComfyUI/models
    "python": "",            # interpreter that runs ComfyUI
    "managed": True,
    "auto_start_comfy": True,
    "torch_index": "",
    "hf_token": "",
    "hf_endpoint": HF_BASE,
    "hf_repo": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "want_clone": True,
    "want_17b": False,
    "want_voicedesign": False,
    "want_moss": True,
    "want_moss_8b": False,
    # On by default: MOSS has no preset speakers, so describing a voice is one
    # of only two ways to pin one down, and at 1.7B it runs on the same card
    # as the base model.
    "want_moss_design": True,
    "engine": "qwen",
    "setup_complete": False,
}

QWEN_SUBDIR = Path("qwen-tts")
MOSS_SUBDIR = Path("moss-tts")

# Everything that differs between the two engines, in one place, so adding a
# third is a table entry rather than a hunt through four files.
#
# `layout` is the part that bites: the Qwen node searches
# models/qwen-tts/<Org>/<Name>, while the MOSS loader builds its cache path as
# repo_id.replace("/", "--") under models/moss-tts. Put a MOSS folder in the
# Qwen shape and the node silently ignores it and downloads its own copy.
ENGINES = {
    "qwen": {
        "id": "qwen",
        "label": "Qwen3-TTS",
        "node_repo": NODE_REPO,
        "node_dir": NODE_DIR_NAME,
        "node_marker": "nodes.py",
        "subdir": QWEN_SUBDIR,
        "layout": "org",
        "models": MODEL_REPOS,
        "blurb": "Preset speakers, cloning and voice design. Small and fast.",
    },
    "moss": {
        "id": "moss",
        "label": "MOSS-TTS",
        "node_repo": MOSS_NODE_REPO,
        "node_dir": MOSS_NODE_DIR_NAME,
        "node_marker": "__init__.py",
        "subdir": MOSS_SUBDIR,
        "layout": "flat",
        "models": MOSS_MODEL_REPOS,
        "blurb": "Zero-shot cloning and voice design, no preset speakers.",
    },
}
DEFAULT_ENGINE = "qwen"


def engine_of(repo: str) -> str:
    """Which engine a model repo belongs to, from the tables themselves."""
    for eid, eng in ENGINES.items():
        if any(m["repo"] == repo for m in eng["models"]):
            return eid
    return DEFAULT_ENGINE


def engine_enabled(cfg: dict, engine: str) -> bool:
    """MOSS can be turned off; Qwen is the engine the app is built around."""
    if engine == "moss":
        return bool(cfg.get("want_moss", True))
    return True


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    # Heal a URL saved before it was normalised — a trailing slash in here used
    # to keep the whole app from starting.
    cfg["comfy_url"] = clean_url(cfg.get("comfy_url")) or DEFAULT_COMFY_URL
    return cfg


def save_config(cfg: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #
class Progress:
    STEPS = [
        ("python", "Check Python"),
        ("comfyui", "Install ComfyUI"),
        ("node", "Install the speech nodes"),
        ("deps", "Install dependencies"),
        ("models", "Download voices and models"),
        ("launch", "Start ComfyUI"),
    ]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.lines: list[str] = []
        self.running = False
        self.done = False
        self.error: str | None = None
        self.step = ""
        # pct is None while a step has no measurable progress — a bar drawn at
        # 0% for the whole of a fifteen-minute step reads as "stuck", so the
        # page shows none until there is a number to put in it.
        self.steps = {k: {"label": v, "state": "pending", "detail": "",
                          "pct": None}
                      for k, v in self.STEPS}

    def log(self, msg: str) -> None:
        with self._lock:
            self.lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            if len(self.lines) > 4000:
                del self.lines[:2000]
        print(f"[setup] {msg}", flush=True)

    def begin(self, key: str, detail: str = "") -> None:
        with self._lock:
            self.step = key
            self.steps[key]["state"] = "running"
            self.steps[key]["detail"] = detail
            self.steps[key]["pct"] = None

    def detail(self, key: str, detail: str, pct: float | None = None) -> None:
        """pct None puts the bar away rather than leaving the last one up.

        A download that finishes and hands over to a step with nothing to
        measure — unpacking a wheel, resolving dependencies — used to leave the
        bar frozen at 100% for the minutes that followed, which reads as a run
        that has finished and hung. An absent bar reads as "no number yet",
        which is the truth.
        """
        with self._lock:
            self.steps[key]["detail"] = detail
            self.steps[key]["pct"] = None if pct is None \
                else max(0.0, min(100.0, float(pct)))

    def finish(self, key: str, detail: str = "") -> None:
        with self._lock:
            self.steps[key]["state"] = "done"
            self.steps[key]["pct"] = None
            if detail:
                self.steps[key]["detail"] = detail

    def fail(self, key: str, detail: str) -> None:
        with self._lock:
            self.steps[key]["state"] = "error"
            self.steps[key]["detail"] = detail

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            return {"running": self.running, "done": self.done,
                    "error": self.error, "step": self.step,
                    # A list, in the order the steps actually happen. This used
                    # to be the dict itself, and Flask sorts the keys of every
                    # dict it sends — which listed Check Python last, after the
                    # step that starts the engine, on the one screen where
                    # order is the whole point.
                    "steps": [{"key": key, **self.steps[key]}
                              for key, _ in self.STEPS],
                    "cursor": len(self.lines), "lines": self.lines[since:]}


# --------------------------------------------------------------------------- #
# interpreters
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def find_python(prog: Progress | None = None) -> str:
    """A real Python 3.10+. Windows Store stubs resolve on PATH and fail to
    run, so every candidate is tested by actually executing it."""
    candidates: list[list[str]] = [[sys.executable]]
    if platform.system() == "Windows":
        candidates += [["py", "-3.12"], ["py", "-3.11"], ["py", "-3.10"],
                       ["py", "-3"], ["python"]]
    else:
        candidates += [["python3.12"], ["python3.11"], ["python3.10"],
                       ["python3"], ["python"]]
    for cand in candidates:
        try:
            out = _run(cand + ["-c", "import sys;print(sys.executable);"
                                     "print('%d.%d' % sys.version_info[:2])"],
                       timeout=25)
        except Exception:
            continue
        if out.returncode != 0:
            continue
        parts = [p.strip() for p in out.stdout.strip().splitlines() if p.strip()]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            major, minor = (int(x) for x in parts[1].split("."))
        except ValueError:
            continue
        if (major, minor) >= (3, 10):
            if prog:
                prog.log(f"Using Python {parts[1]} at {parts[0]}")
            return parts[0]
    raise RuntimeError("No Python 3.10 or newer found. Install Python from "
                       "python.org, tick 'Add to PATH', and run setup again.")


def portable_python(comfy_dir: Path) -> Path | None:
    """ComfyUI Windows portable ships python_embeded next to the ComfyUI
    folder. That is the interpreter its packages must go into."""
    for base in (comfy_dir.parent, comfy_dir):
        cand = base / "python_embeded" / "python.exe"
        if cand.exists():
            return cand
    return None


def venv_python(comfy_dir: Path) -> Path:
    venv = comfy_dir.parent / "comfy-venv"
    return venv / ("Scripts/python.exe" if platform.system() == "Windows"
                   else "bin/python")


def _interpreters(comfy_dir: Path) -> list[Path]:
    """Where a ComfyUI install keeps the interpreter it runs on, best first.

    Portable python_embeded leads, as the install instructions say. After it
    come the environments an install someone else set up keeps beside or inside
    its own folder — an existing ComfyUI already has torch in one of these, and
    the node's requirements have to land in the same place or ComfyUI will not
    import them. Our own comfy-venv is last, because it only exists when we
    built it.
    """
    win = platform.system() == "Windows"
    exe = "Scripts/python.exe" if win else "bin/python"
    cands: list[Path] = []
    if win:
        cands += [comfy_dir.parent / "python_embeded" / "python.exe",
                  comfy_dir / "python_embeded" / "python.exe"]
    cands += [comfy_dir / "venv" / exe,
              comfy_dir / ".venv" / exe,
              comfy_dir.parent / "venv" / exe,
              comfy_dir.parent / ".venv" / exe]
    if win:
        cands += [comfy_dir.parent / "python_standalone" / "python.exe"]
    else:
        cands += [comfy_dir.parent / "python_standalone" / "bin" / "python"]
    cands += [venv_python(comfy_dir)]
    return cands


def existing_python(comfy_dir: Path) -> str:
    """The interpreter an existing ComfyUI already runs on, if we can find it.

    Tested by running it, never by its path alone — the same rule as
    find_python(). An install whose environment has torch wins outright; a
    working interpreter without torch is the fallback, because it is still that
    install's own environment and ours has no business replacing it.
    """
    fallback = ""
    for cand in _interpreters(comfy_dir):
        try:
            if not cand.exists():
                continue
            out = _run([str(cand), "-c", "import importlib.util as u;"
                                         "print(bool(u.find_spec('torch')))"],
                       timeout=60)
        except Exception:
            continue
        if out.returncode != 0:
            continue
        if out.stdout.strip().splitlines()[-1:] == ["True"]:
            return str(cand)
        fallback = fallback or str(cand)
    return fallback


def comfy_python(cfg: dict) -> str:
    """Whichever interpreter ComfyUI runs on: portable first, then the
    environment the install already has, then whatever setup recorded."""
    comfy_dir = Path(cfg["comfy_dir"]) if cfg.get("comfy_dir") else None
    if comfy_dir:
        p = portable_python(comfy_dir)
        if p:
            return str(p)
        found = existing_python(comfy_dir)
        if found:
            return found
    return cfg.get("python") or ""


def have_git() -> bool:
    return shutil.which("git") is not None


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def detect_comfy_dirs() -> list[str]:
    home = Path.home()
    cands = [APP_DIR / "ComfyUI", home / "ComfyUI",
             home / "Documents" / "ComfyUI", home / "Desktop" / "ComfyUI",
             Path("C:/ComfyUI"), Path("C:/ComfyUI_windows_portable/ComfyUI"),
             Path("D:/ComfyUI"), Path("D:/ComfyUI_windows_portable/ComfyUI")]
    appdata = os.environ.get("APPDATA")
    local = os.environ.get("LOCALAPPDATA")
    if appdata:
        cands.append(Path(appdata) / "ComfyUI")
    if local:
        cands.append(Path(local) / "Programs" / "@comfyorgcomfyui-electron"
                     / "resources" / "ComfyUI")
    out, seen = [], set()
    for c in cands:
        try:
            if ((c / "main.py").exists() or (c / "models").is_dir()) \
                    and str(c) not in seen:
                seen.add(str(c))
                out.append(str(c))
        except OSError:
            continue
    return out


def model_dir(models_dir: Path, repo: str, engine: str = "") -> Path:
    """Where a model folder has to live for its own node to find it.

    Two different layouts, and neither is a preference:
      qwen  models/qwen-tts/<Org>/<Name>  — where the Qwen node searches.
      moss  models/moss-tts/<Org>--<Name> — what the MOSS loader builds from
            repo_id.replace("/", "--"). Put a MOSS folder in the Qwen shape
            and the node does not see it; it downloads its own second copy.
    """
    eng = ENGINES[engine or engine_of(repo)]
    org, name = repo.split("/", 1)
    if eng["layout"] == "flat":
        return models_dir / eng["subdir"] / f"{org}--{name}"
    return models_dir / eng["subdir"] / org / name


def qwen_model_dir(models_dir: Path, repo: str) -> Path:
    """Kept for callers that only ever meant Qwen."""
    return model_dir(models_dir, repo, "qwen")


def model_installed(models_dir: Path, repo: str, engine: str = "") -> bool:
    d = model_dir(models_dir, repo, engine)
    if not d.is_dir():
        return False
    # A .part is a download that stopped part way through. The config.json
    # beside it arrived first and is perfectly good, which is exactly why this
    # has to be checked: without it a folder whose weights are still half here
    # reports as installed, and the engine reports ready.
    if any(d.rglob("*.part")):
        return False
    weights = [f for f in d.rglob("*")
               if f.suffix in (".safetensors", ".bin", ".pt", ".pth")]
    has_config = (d / "config.json").exists()
    return bool(weights) or has_config


def wanted_models(cfg: dict, engine: str = "") -> list[dict]:
    """The folders this setup actually needs, from what the person asked for.

    Each entry carries its own engine, because the caller downloading them has
    to know which of the two folder layouts above to use.
    """
    out = []
    for eid, eng in ENGINES.items():
        if engine and eid != engine:
            continue
        if not engine_enabled(cfg, eid):
            continue
        for m in eng["models"]:
            flag = GROUP_FLAG.get(m["group"])
            if flag is not None and not cfg.get(flag):
                continue
            if m["group"] == "clone_hq" and not cfg.get("want_clone"):
                continue
            out.append({**m, "engine": eid})
    return out


def missing_models(models_dir: Path, cfg: dict, engine: str = "") -> list[dict]:
    return [m for m in wanted_models(cfg, engine)
            if not model_installed(models_dir, m["repo"], m["engine"])]


def node_installed(comfy_dir: Path, engine: str = "qwen") -> bool:
    eng = ENGINES[engine]
    return (comfy_dir / "custom_nodes" / eng["node_dir"]
            / eng["node_marker"]).exists()


# --------------------------------------------------------------------------- #
# huggingface snapshot download
# --------------------------------------------------------------------------- #
def hf_headers(cfg: dict) -> dict:
    token = (cfg.get("hf_token") or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_endpoint(cfg: dict) -> str:
    return (cfg.get("hf_endpoint") or HF_BASE).rstrip("/")


def hf_tree(cfg: dict, repo: str, revision: str = "main") -> list[dict]:
    """Every file in a repo, with sizes."""
    base = hf_endpoint(cfg)
    last = ""
    for kind in ("models", "datasets"):
        url = f"{base}/api/{kind}/{repo}/tree/{revision}?recursive=1"
        try:
            r = requests.get(url, headers=hf_headers(cfg), timeout=30)
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            continue
        if r.status_code == 401:
            raise RuntimeError("This repo needs a HuggingFace token. Add one in "
                               "Models, then try again.")
        if r.status_code == 403:
            raise RuntimeError("Your token cannot read this repo. Accept the "
                               "model licence on huggingface.co first.")
        if r.status_code == 404:
            continue
        r.raise_for_status()
        files = []
        for e in r.json():
            if e.get("type") != "file":
                continue
            size = (e.get("lfs") or {}).get("size") or e.get("size") or 0
            files.append({"path": e["path"], "size": size})
        return files
    raise RuntimeError(f"Could not find '{repo}' on {base}. "
                       + (last or "Check the spelling, or add a token."))


SKIP_SUFFIX = (".gitattributes", ".md")


def wanted_files(files: list[dict]) -> list[dict]:
    """Drop repo furniture, and drop .bin duplicates when safetensors exist."""
    keep = [f for f in files if not f["path"].endswith(SKIP_SUFFIX)]
    safet = {Path(f["path"]).stem for f in keep if f["path"].endswith(".safetensors")}
    out = []
    for f in keep:
        p = Path(f["path"])
        if p.suffix in (".bin", ".pth") and p.stem.replace("pytorch_model",
                                                           "model") in safet:
            continue
        out.append(f)
    return out


def human_size(n: float) -> str:
    """Bytes at a scale that reads. Fixing this at GB showed every repo under
    ten megabytes as "0.00 of 0.00 GB"."""
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e7:
        return f"{n / 1e6:.0f} MB"
    if n >= 1e6:
        # A decimal below ten megabytes, or a 1.8 MB wheel reads as "2 MB".
        return f"{n / 1e6:.1f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} kB"
    return f"{int(n)} B"


def download_file(cfg: dict, repo: str, path: str, dest: Path,
                  on_progress=None, should_cancel=None,
                  revision: str = "main", expected: int = 0) -> None:
    """Resumable single-file download: .part file, Range resume, atomic move.

    `expected` is the size the repo listing gave, when there is one. Nothing is
    moved into place until what arrived accounts for it.
    """
    url = f"{hf_endpoint(cfg)}/{repo}/resolve/{revision}/{path}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(hf_headers(cfg))
    if have:
        headers["Range"] = f"bytes={have}-"

    with requests.get(url, headers=headers, stream=True, timeout=60,
                      allow_redirects=True) as r:
        if r.status_code == 416:
            part.replace(dest)
            return
        if r.status_code in (401, 403):
            raise RuntimeError("HuggingFace refused the download. Add a token "
                               "with access to this repo.")
        r.raise_for_status()
        # A 206 means the server honoured the Range header and Content-Length
        # covers only what is left; a 200 means it ignored it and is sending
        # the whole file again, so what is already on disk does not count —
        # towards the total either, or the progress readout runs past 100%.
        resuming = bool(have) and r.status_code == 206
        mode = "ab" if resuming else "wb"
        if not resuming:
            have = 0
        total = int(r.headers.get("Content-Length", 0)) + have
        got, last = have, 0.0
        with open(part, mode) as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if should_cancel and should_cancel():
                    return
                if not chunk:
                    continue
                fh.write(chunk)
                got += len(chunk)
                now = time.time()
                if on_progress and now - last > 0.6:
                    last = now
                    on_progress(got, total)
    # A connection that drops mid-file ends that loop exactly like a clean
    # finish does. Renaming a short file into place makes it look complete for
    # good: the .part it would have resumed from is gone, and the folder counts
    # as installed while the weights in it are truncated. Keep the .part and
    # say so — the next attempt carries on from where this one stopped.
    want = expected or total
    landed = part.stat().st_size if part.exists() else 0
    if want and landed < want:
        raise RuntimeError(
            f"{path} stopped at {landed / 1e6:.1f} MB of {want / 1e6:.1f} MB — "
            "the connection dropped. Start the download again and it carries "
            "on from here.")
    part.replace(dest)
    if on_progress:
        on_progress(dest.stat().st_size, dest.stat().st_size)


def download_repo(cfg: dict, repo: str, models_dir: Path,
                  on_detail=None, should_cancel=None, engine: str = "") -> None:
    """Pull a whole model folder into the layout its own node searches."""
    target = model_dir(models_dir, repo, engine)
    files = wanted_files(hf_tree(cfg, repo))
    total_bytes = sum(f["size"] for f in files) or 1
    done_bytes = 0
    for i, f in enumerate(files, 1):
        dest = target / f["path"]
        if dest.exists() and f["size"] and dest.stat().st_size == f["size"]:
            done_bytes += f["size"]
            continue

        def prog(got, tot, _f=f, _i=i, _done=done_bytes):
            if on_detail:
                overall = (_done + got) / total_bytes * 100
                on_detail(f"{repo.split('/')[-1]} · file {_i}/{len(files)} · "
                          f"{human_size(_done + got)} of "
                          f"{human_size(total_bytes)} ({overall:.0f}%)",
                          overall)

        download_file(cfg, repo, f["path"], dest, prog, should_cancel,
                      expected=f["size"])
        if should_cancel and should_cancel():
            return
        done_bytes += f["size"]


# --------------------------------------------------------------------------- #
# ComfyUI process
# --------------------------------------------------------------------------- #
class ComfyProcess:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []
        self._lock = threading.Lock()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, python: str, comfy_dir: Path, port: int,
              prog: Progress) -> None:
        """Raises RuntimeError with a sentence a person can act on. A ComfyUI
        folder that has moved, or an interpreter that is gone, is an engine
        that cannot start — never a reason the whole app fails to boot."""
        if self.alive():
            return
        if not (comfy_dir / "main.py").exists():
            raise RuntimeError(
                f"There is no ComfyUI at {comfy_dir} any more — the folder has "
                "moved or been deleted. Run setup again from Settings.")
        cmd = [python, "main.py", "--listen", "127.0.0.1", "--port", str(port),
               "--disable-auto-launch"]
        prog.log("Launching ComfyUI: " + " ".join(cmd))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            if platform.system() == "Windows" else 0
        try:
            self.proc = subprocess.Popen(cmd, cwd=str(comfy_dir),
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True,
                                         bufsize=1, creationflags=flags)
        except OSError as exc:
            raise RuntimeError(
                f"ComfyUI could not be started with {python} — {exc}. "
                "Run setup again from Settings.") from exc
        threading.Thread(target=self._pump, args=(prog,), daemon=True).start()

    def _pump(self, prog: Progress) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.rstrip()
            with self._lock:
                self.lines.append(line)
                if len(self.lines) > 2000:
                    del self.lines[:1000]
            if any(k in line for k in ("Error", "Traceback", "error:",
                                       "Qwen", "Starting server",
                                       "IMPORT FAILED")):
                prog.log(f"ComfyUI: {line}")

    def tail(self, n: int = 40) -> list[str]:
        with self._lock:
            return self.lines[-n:]

    def stop(self) -> None:
        if self.alive():
            try:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


def comfy_online(url: str) -> bool:
    try:
        return requests.get(f"{url}/system_stats", timeout=3).status_code == 200
    except Exception:
        return False


def wait_for_comfy(url: str, timeout: int = 900) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if comfy_online(url):
            return True
        time.sleep(2)
    return False


# --------------------------------------------------------------------------- #
# pip
# --------------------------------------------------------------------------- #
def stream_lines(stream):
    """What a subprocess writes, with a carriage return counting as a break.

    pip redraws its progress over itself with \r. Iterating the pipe by line
    waits for a \n that only arrives when the download has already finished,
    which is why a 2.7 GB PyTorch showed "Collecting torch" and then nothing.
    """
    buffer = ""
    while True:
        char = stream.read(1)
        if not char:
            break
        if char in ("\r", "\n"):
            if buffer.strip():
                yield buffer.strip()
            buffer = ""
        else:
            buffer += char
    if buffer.strip():
        yield buffer.strip()


def pip_progress(line: str, state: dict):
    """One line of pip output -> (something worth reading, percent) or None.

    `--progress-bar raw` prints "Progress 123 of 456" as it goes, and that is
    the only account of a multi-gigabyte download that survives being piped:
    pip draws no bar at all unless it is talking to a terminal.
    """
    if line.startswith("Downloading "):
        parts = line.split()
        state["what"] = (parts[1].split("-")[0] if len(parts) > 1
                         else "package")
        state["since"] = time.time()
        state["phase"] = "download"
        # Shown as well as recorded: where pip is too old for raw progress
        # this line and the clock below are the whole account of a download.
        size = re.search(r"\(([\d.]+)\s*([kKMG]?)B\)", line)
        bytes_ = 0.0
        if size:
            scale = {"": 1, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9}
            bytes_ = float(size.group(1)) * scale[size.group(2)]
        return (f"Downloading {state['what']}"
                + (f" — {human_size(bytes_)}" if size else ""), None)
    if line.startswith("Installing collected packages"):
        # Nothing is printed again until this finishes, and for a 2.7 GB
        # PyTorch that is minutes of writing thousands of files. pip_install's
        # heartbeat measures site-packages growing so there is still a number
        # moving; all this can do is name the phase and count the packages.
        names = [n.strip() for n in line.split(":", 1)[-1].split(",")
                 if n.strip()] if ":" in line else []
        state["packages"] = len(names)
        state["phase"] = "install"
        state["what"] = (f"Unpacking {len(names)} packages" if len(names) > 1
                         else "Unpacking and installing — the slow part")
        return (state["what"], None)
    if line.startswith(("Building", "Preparing", "Getting requirements")):
        state["what"] = line[:60]
        return (line[:60], None)
    if not line.startswith("Progress "):
        return None
    parts = line.split()
    try:
        got, total = int(parts[1]), int(parts[3])
    except (IndexError, ValueError):
        return None
    what = state.get("what", "package")
    elapsed = max(time.time() - state.get("since", time.time()), 0.001)
    speed = got / elapsed
    if total <= 0:
        return (f"{what} — {got / 1e6:.0f} MB so far", None)
    if total >= 1e9:
        size = f"{got / 1e9:.2f} of {total / 1e9:.2f} GB"
    elif total >= 1e7:
        size = f"{got / 1e6:.0f} of {total / 1e6:.0f} MB"
    else:
        # Rounding to whole megabytes showed a 1.3 MB wheel as "0 of 1 MB",
        # which reads as a stuck download rather than a small one.
        size = f"{got / 1e6:.1f} of {total / 1e6:.1f} MB"
    # Clamped here so every consumer gets a sane number: Progress.detail
    # clamps its own, but a Task on the Engine page takes what it is given and
    # would set a bar to a negative width.
    pct = max(0.0, min(100.0, got * 100.0 / total))
    head = f"{what} — {size} ({pct:.0f}%)"
    # A rate measured over the first fraction of a second is nonsense — one
    # chunk arriving at once reads as several hundred MB/s.
    if elapsed < 1.5 or speed <= 0:
        return (head, pct)
    left = (total - got) / speed
    return (f"{head} · {speed / 1e6:.1f} MB/s · "
            f"{int(left // 60)}m {int(left % 60):02d}s left", pct)


_PIP_RAW: dict[str, list[str]] = {}


def pip_raw_progress(python: str, refresh: bool = False) -> list[str]:
    """`--progress-bar raw` if this pip offers it, asked rather than guessed.

    A version number is the wrong test: pip 24.0 takes only on/off and exits
    with "invalid choice: 'raw'" — passing it there does not merely lose the
    percentage, it fails the install. pip's own help lists the choices, so
    read them. Cached per interpreter, because the answer only changes when
    pip_ready() upgrades pip and then asks again.
    """
    key = str(python)
    if not refresh and key in _PIP_RAW:
        return _PIP_RAW[key]
    flags: list[str] = []
    try:
        out = _run([key, "-m", "pip", "install", "--help"], timeout=60)
        block = re.search(r"--progress-bar[^\n]*\n(?:\s{6,}[^\n]*\n)*",
                          out.stdout or "")
        if block and re.search(r"\braw\b", block.group(0)):
            flags = ["--progress-bar", "raw"]
    except Exception:  # noqa: BLE001
        pass
    _PIP_RAW[key] = flags
    return flags


_PIP_READY: set[str] = set()


def pip_ready(python: str, log) -> None:
    """Get this interpreter a pip that can report progress. Once, per pip.

    `--progress-bar raw` arrived in pip 24.1, and it is the only account of a
    2.7 GB download that survives being piped — pip draws nothing at all when
    it is not talking to a terminal. A venv ships whatever pip its base Python
    bundled, which for Python 3.11 is 24.0, so without this the longest step of
    the install is a blank panel for ten minutes. That is the bug report:
    "not showing me how much gigabyte the file is and how much is done".

    An upgrade that fails is not fatal — the install still runs, it just runs
    quietly — so this never raises.
    """
    key = str(python)
    if key in _PIP_READY:
        return
    _PIP_READY.add(key)
    if pip_raw_progress(key):
        return
    log("Updating pip first so the download can report a percentage…")
    try:
        out = _run([key, "-m", "pip", "install", "--upgrade", "pip",
                    "--disable-pip-version-check"], timeout=900)
        if out.returncode != 0:
            log("Could not update pip — the install will run without a "
                "percentage. " + (out.stderr or out.stdout or "")[-200:])
    except Exception as exc:  # noqa: BLE001
        log(f"Could not update pip ({exc}) — the install will run without a "
            "percentage.")
    pip_raw_progress(key, refresh=True)


_SITE: dict[str, str] = {}


def site_packages(python: str) -> str:
    """Where this interpreter unpacks wheels, asked once and remembered."""
    key = str(python)
    if key not in _SITE:
        try:
            out = _run([key, "-c", "import sysconfig;"
                                   "print(sysconfig.get_paths()['purelib'])"],
                       timeout=60)
            _SITE[key] = (out.stdout or "").strip().splitlines()[-1] \
                if out.returncode == 0 and (out.stdout or "").strip() else ""
        except Exception:  # noqa: BLE001
            _SITE[key] = ""
    return _SITE[key]


def dir_size(path, budget: int = 400_000) -> int:
    """Bytes under `path`, giving up after `budget` files.

    This runs on a timer while pip unpacks, and a site-packages with torch in
    it is fifty thousand files — the cap is there so a pathological tree can
    never turn the progress report into the slow part.
    """
    total, seen = 0, 0
    for root, _dirs, files in os.walk(str(path), onerror=lambda _e: None):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
            seen += 1
            if seen >= budget:
                return total
    return total


def quiet_detail(state: dict, written: int, seconds: int) -> str:
    """What to show while pip is saying nothing. Pulled out to be testable.

    No percentage, on purpose. A wheel unpacks to more than it downloads, by a
    ratio that varies per package, so a bar worked out from the download size
    would sit pinned at 100% for minutes — worse than no bar. A byte count that
    keeps climbing is the honest version of "how much of the progress is done".
    """
    head = state.get("what") or "Working"
    if written:
        # Bytes written, and deliberately not "of the 2.7 GB downloaded": a
        # wheel unpacks to more than it downloads, so that reads as 199 of 88
        # and looks like a bug rather than progress.
        head += f" — {human_size(written)} written"
    return f"{head} — {seconds // 60}m {seconds % 60:02d}s so far"


def pip_install(python: str, args: list[str], log, on_detail=None) -> None:
    """Run pip, reporting progress through on_detail(text, pct).

    pct is None when there is no number to show — pip says nothing measurable
    while it resolves dependencies or unpacks a wheel.
    """
    pip_ready(python, log)
    cmd = [python, "-m", "pip", "install"] + args + pip_raw_progress(python)
    log("$ " + " ".join(cmd[:8]) + (" …" if len(cmd) > 8 else ""))

    state: dict = {}
    last = [0.0]
    last_pct = [-1.0]
    started = time.time()
    stop = threading.Event()
    # Measured before pip runs, so what the unpacking step reports is bytes
    # this install wrote and not the size of everything already there.
    site = site_packages(python) if on_detail else ""
    base_size = dir_size(site) if site else 0

    def tick() -> None:
        # Unpacking a 2.7 GB wheel prints nothing for minutes. Keep a clock
        # running so the panel never looks like it has died — and, once pip
        # is writing, weigh site-packages so the number moves.
        while not stop.wait(5):
            if not on_detail or time.time() - last[0] < 5:
                continue
            written = (max(dir_size(site) - base_size, 0)
                       if site and state.get("phase") == "install" else 0)
            on_detail(quiet_detail(state, written,
                                   int(time.time() - started)), None)

    # The context manager closes the pipe and reaps the child even if reading
    # its output raises, which a bare Popen left to garbage collection did not.
    with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True,
                          bufsize=0) as proc:
        assert proc.stdout
        if on_detail:
            threading.Thread(target=tick, daemon=True).start()
        try:
            for line in stream_lines(proc.stdout):
                # The log gets its line whatever the panel does. These two
                # used to be an if/elif, so whether "Downloading torch
                # (2.7 GB)" or "Installing collected packages" reached the log
                # depended on whether the throttle below happened to swallow
                # the panel update — and that log is what people are told to
                # read when an install fails.
                if line.startswith(("Collecting", "Downloading", "Installing",
                                    "Successfully", "ERROR", "Building",
                                    "WARNING: ")):
                    log(line[:200])
                shown = pip_progress(line, state)
                if not (shown and on_detail):
                    continue
                # Throttled by time, except when the percentage actually
                # moved: a fast mirror can deliver a whole wheel in three
                # bursts, and time alone would swallow every one of them and
                # leave the bar at zero.
                moved = (shown[1] is not None
                         and abs(shown[1] - last_pct[0]) >= 1.0)
                if moved or time.time() - last[0] > 0.4:
                    last[0] = time.time()
                    if shown[1] is not None:
                        last_pct[0] = shown[1]
                    on_detail(shown[0], shown[1])
        finally:
            stop.set()
        code = proc.wait()
    if code != 0:
        raise RuntimeError("pip install failed — see the log.")


def _smi_candidates() -> list[str]:
    """Every place nvidia-smi might be, PATH first.

    The Windows driver drops it in System32, which is normally on PATH — but
    "normally" is what rule 5 already burned us on: a shortcut, a service, or a
    venv activated from a trimmed environment can hand the process a PATH
    without it, and shutil.which then reports no GPU on a machine that has one.
    """
    found = shutil.which("nvidia-smi")
    cands = [found] if found else []
    if platform.system() == "Windows":
        root = os.environ.get("SystemRoot") or r"C:\Windows"
        cands.append(str(Path(root) / "System32" / "nvidia-smi.exe"))
        for base in {os.environ.get("ProgramFiles") or r"C:\Program Files",
                     os.environ.get("ProgramW6432") or r"C:\Program Files"}:
            cands.append(str(Path(base) / "NVIDIA Corporation" / "NVSMI"
                             / "nvidia-smi.exe"))
    else:
        cands += ["/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi",
                  "/opt/bin/nvidia-smi"]
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _adapter_names() -> list[str]:
    """Display adapters, for the case where the driver is not installed.

    nvidia-smi ships with the driver, so its absence means "no usable GPU",
    not "no GPU" — and telling someone with an RTX 4060 that they have no GPU
    is both wrong and unactionable. These names come from the machine itself.
    """
    system = platform.system()
    try:
        if system == "Windows":
            out = _run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_VideoController | "
                        "Select-Object -ExpandProperty Name"], timeout=30)
            return [l.strip() for l in (out.stdout or "").splitlines() if l.strip()]
        if system == "Linux":
            names = []
            gpus = Path("/proc/driver/nvidia/gpus")
            for entry in gpus.iterdir() if gpus.is_dir() else []:
                text = (entry / "information").read_text(errors="replace") \
                    if (entry / "information").exists() else ""
                match = re.search(r"Model:\s*(.+)", text)
                if match:
                    names.append(match.group(1).strip())
            if names:
                return names
            out = _run(["lspci"], timeout=20)
            return [l.strip() for l in (out.stdout or "").splitlines()
                    if "VGA" in l or "3D controller" in l]
    except Exception:  # noqa: BLE001
        pass
    return []


_GPU: dict = {}


def nvidia_gpu(refresh: bool = False) -> dict:
    """What NVIDIA hardware is here, and whether its driver answers.

    Returns {"name": str, "driver": bool}. `driver` False with a name means
    the card is in the machine but nvidia-smi did not run, which is a driver
    problem and says so — a different sentence from "no GPU".

    Cached: dependencies() asks on every Recheck, and a PowerShell query per
    poll would be a second of the user's machine for an answer that cannot
    change without a reboot.
    """
    if _GPU and not refresh:
        return _GPU
    name, driver = "", False
    for smi in _smi_candidates():
        try:
            out = _run([smi, "--query-gpu=name", "--format=csv,noheader"],
                       timeout=30)
        except Exception:  # noqa: BLE001
            continue
        if out.returncode != 0:
            continue
        first = next((l.strip() for l in (out.stdout or "").splitlines()
                      if l.strip()), "")
        if first:
            name, driver = first, True
            break
    if not name:
        name = next((n for n in _adapter_names() if "nvidia" in n.lower()), "")
    _GPU.clear()
    _GPU.update({"name": name, "driver": driver})
    return _GPU


CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def torch_index(cfg: dict) -> str:
    if cfg.get("torch_index"):
        return cfg["torch_index"]
    if platform.system() == "Darwin":
        return ""
    return CUDA_INDEX if nvidia_gpu()["name"] else CPU_INDEX


def torch_build(index: str) -> str:
    """The local version tag the wheels at `index` carry: cu128, cpu, or "".

    pip treats torch 2.14.0+cpu as satisfying a plain `torch`, so asking the
    CUDA index for it changes nothing: the build has to be compared, and the
    old one uninstalled, before the right one can land.
    """
    match = re.search(r"/whl/(cu\d+|cpu|rocm[\d.]*)", index or "")
    return match.group(1) if match else ""


def installed_torch(python: str) -> str:
    """The torch already in this environment, "" if there is none."""
    try:
        out = _run([str(python), "-c", "import torch;print(torch.__version__)"],
                   timeout=180)
    except Exception:  # noqa: BLE001
        return ""
    if out.returncode != 0 or not (out.stdout or "").strip():
        return ""
    return out.stdout.strip().splitlines()[-1].strip()


def drop_mismatched_torch(python: str, index: str, log) -> bool:
    """Remove a torch whose build is not the one being asked for.

    pip treats `torch` as satisfied by torch 2.14.0+cpu, so pointing it at the
    CUDA index and asking again changes nothing at all — which is why pressing
    Reinstall on a CPU build left the CPU build in place and the GPU unused.
    The old build has to go first. Nothing is removed when the builds already
    agree, so a Reinstall that only wants to repair a broken install is still
    the cheap operation it looks like.
    """
    wanted = torch_build(index)
    if not wanted:
        return False
    have = installed_torch(python)
    if not have:
        return False
    current = have.split("+")[1] if "+" in have else ""
    # A wheel from the default PyPI index carries no local tag and is the CUDA
    # build on Linux, the CPU build on Windows — it cannot be matched against
    # a tag, so it is left alone rather than reinstalled on a guess.
    if not current or current == wanted:
        return False
    log(f"Installed torch is {have}, but the {wanted} build was asked for — "
        "removing it first, because pip counts the old one as good enough.")
    try:
        _run([str(python), "-m", "pip", "uninstall", "-y",
              "torch", "torchaudio", "torchvision"], timeout=900)
    except Exception as exc:  # noqa: BLE001
        log(f"Could not remove the old torch ({exc}) — carrying on.")
        return False
    return True


NODE_PROBE = """
import importlib.util, sys
root, pkg = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("qwen_tts_probe",
                                              pkg + "/__init__.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except BaseException as exc:
    print("FAILED " + type(exc).__name__ + ": " + str(exc)[:400])
    sys.exit(2)
print("OK")
"""


def node_import_error(python: str, comfy_dir: Path, engine: str = "qwen") -> str:
    """Why ComfyUI could not load an engine's nodes, in one sentence.

    ComfyUI prints its import failures to its own console and carries on, so
    "Installed but ComfyUI has not loaded them" was as far as the Engine panel
    could get — a dead end for anyone running it from a launcher with no
    console to read. Importing the package in the same interpreter, with the
    ComfyUI folder on sys.path the way ComfyUI puts it there, reproduces the
    failure and gets the actual exception back: almost always a package the
    node needs that is not installed.

    Returns "" when the import succeeds, which means the running ComfyUI is
    simply older than the install and wants a restart.
    """
    eng = ENGINES[engine]
    node_path = comfy_dir / "custom_nodes" / eng["node_dir"]
    if not (node_path / "__init__.py").exists():
        return f"The {eng['label']} nodes are not installed."
    try:
        out = _run([str(python), "-c", NODE_PROBE, str(comfy_dir),
                    str(node_path)], cwd=str(comfy_dir), timeout=600)
    except Exception as exc:  # noqa: BLE001
        return f"Could not test the import: {exc}"
    text = ((out.stdout or "") + (out.stderr or "")).strip()
    if out.returncode == 0 and text.endswith("OK"):
        return ""
    for line in reversed(text.splitlines()):
        if line.startswith("FAILED "):
            return line[len("FAILED "):]
    return text[-400:] or "The import failed without saying why."


# --------------------------------------------------------------------------- #
# setup run
# --------------------------------------------------------------------------- #
def run_setup(cfg: dict, prog: Progress, comfy: ComfyProcess,
              chosen_dir: str = "", mode: str = "auto") -> None:
    prog.running = True
    prog.done = False
    prog.error = None
    try:
        # 1. python -------------------------------------------------------- #
        prog.begin("python")
        if mode == "external":
            prog.finish("python", "Not needed — you run ComfyUI yourself")
            py = cfg.get("python") or sys.executable
        else:
            py = find_python(prog)
            cfg["python"] = py
            prog.finish("python", py)

        # 2. comfyui ------------------------------------------------------- #
        prog.begin("comfyui")
        if mode == "external":
            url = cfg["comfy_url"]
            if not comfy_online(url):
                raise RuntimeError(f"Nothing is answering at {url}. Start "
                                   "ComfyUI first, or let Script Builder "
                                   "install its own.")
            if not cfg.get("models_dir"):
                raise RuntimeError("Set the ComfyUI models folder in Settings "
                                   "so the voices land in the right place.")
            cfg["managed"] = False
            prog.finish("comfyui", url)
            comfy_dir = Path(cfg["comfy_dir"]) if cfg.get("comfy_dir") else None
        else:
            if chosen_dir:
                comfy_dir = Path(chosen_dir)
                cfg["managed"] = False
                prog.log(f"Using existing ComfyUI at {comfy_dir}")
            else:
                comfy_dir = APP_DIR / "ComfyUI"
                cfg["managed"] = True
                if not (comfy_dir / "main.py").exists():
                    if not have_git():
                        raise RuntimeError(
                            "Git is not installed, so ComfyUI cannot be "
                            "downloaded. Install Git from the Engine panel, or "
                            "point Script Builder at an existing ComfyUI.")
                    prog.detail("comfyui", "Downloading ComfyUI…")
                    res = _run(["git", "clone", "--depth", "1", COMFY_REPO,
                                str(comfy_dir)])
                    if res.returncode != 0:
                        raise RuntimeError("git clone failed: " +
                                           (res.stderr or res.stdout)[-600:])
                else:
                    prog.detail("comfyui", "Updating ComfyUI…")
                    _run(["git", "-C", str(comfy_dir), "pull", "--ff-only"])
            if not (comfy_dir / "main.py").exists():
                raise RuntimeError(f"No main.py in {comfy_dir} — that folder is "
                                   "not a ComfyUI install.")
            cfg["comfy_dir"] = str(comfy_dir)
            cfg["models_dir"] = str(comfy_dir / "models")
            prog.finish("comfyui", str(comfy_dir))

        models_dir = Path(cfg["models_dir"])

        # 3. custom nodes ---------------------------------------------------- #
        prog.begin("node")
        wanted_engines = [e for e in ENGINES.values()
                          if engine_enabled(cfg, e["id"])]
        if comfy_dir is None:
            prog.finish("node", "Install the speech nodes in your own ComfyUI")
        else:
            nodes_dir = comfy_dir / "custom_nodes"
            landed = []
            for eng in wanted_engines:
                node_path = nodes_dir / eng["node_dir"]
                if node_path.exists():
                    prog.detail("node", f"Updating the {eng['label']} nodes…")
                    _run(["git", "-C", str(node_path), "pull", "--ff-only"])
                else:
                    if not have_git():
                        raise RuntimeError(
                            f"Git is needed to install the {eng['label']} "
                            "nodes. Install it from the Engine panel.")
                    nodes_dir.mkdir(parents=True, exist_ok=True)
                    prog.detail("node", f"Downloading the {eng['label']} nodes…")
                    prog.log(f"git clone {eng['node_repo']}")
                    res = _run(["git", "clone", "--depth", "1",
                                eng["node_repo"], str(node_path)])
                    if res.returncode != 0:
                        raise RuntimeError("git clone failed: " +
                                           (res.stderr or res.stdout)[-600:])
                landed.append(eng["label"])
            prog.finish("node", " and ".join(landed) + f" in {nodes_dir}")

        # 4. dependencies --------------------------------------------------- #
        prog.begin("deps")
        if mode == "external":
            prog.finish("deps", "Handled by your own ComfyUI install")
        else:
            comfy_dir = Path(cfg["comfy_dir"])
            target = portable_python(comfy_dir)
            if target:
                prog.log(f"Portable ComfyUI detected — installing into {target}")
            elif not cfg.get("managed"):
                # Someone else's install already runs on its own environment,
                # with torch in it. Building a second one beside it would cost
                # gigabytes and put the node's requirements where ComfyUI never
                # looks, so the nodes would still fail to import.
                found = existing_python(comfy_dir)
                if not found:
                    raise RuntimeError(
                        f"Could not find the Python environment that the "
                        f"ComfyUI at {comfy_dir} runs on, so the Qwen-TTS "
                        "requirements have nowhere to go. Start that ComfyUI "
                        "yourself and pick 'Connect to a ComfyUI I start "
                        "myself', or let Script Builder install its own.")
                target = Path(found)
                prog.log(f"That install runs on {target} — using it as it is")
            else:
                vpy = venv_python(comfy_dir)
                if not vpy.exists():
                    prog.detail("deps", "Creating the Python environment…")
                    res = _run([py, "-m", "venv",
                                str(comfy_dir.parent / "comfy-venv")])
                    if res.returncode != 0:
                        raise RuntimeError("venv creation failed: " +
                                           (res.stderr or res.stdout)[-600:])
                target = vpy
                say = lambda t, pct: prog.detail("deps", t, pct)  # noqa: E731
                prog.detail("deps", "Installing PyTorch — the long one…")
                pip_install(str(target), ["--upgrade", "pip", "wheel"],
                            prog.log, say)
                idx = torch_index(cfg)
                gpu = nvidia_gpu()
                if gpu["name"]:
                    prog.log(f"Graphics: {gpu['name']}"
                             + ("" if gpu["driver"] else
                                " (driver not answering — nvidia-smi did not "
                                "run, so a CPU build may be the safe one)"))
                else:
                    prog.log("No NVIDIA GPU found — installing the CPU build.")
                # A second setup run over an environment that already has the
                # wrong build would otherwise change nothing: pip counts
                # torch+cpu as satisfying `torch`.
                drop_mismatched_torch(str(target), idx, prog.log)
                args = ["torch", "torchaudio"]
                if idx:
                    args += ["--index-url", idx]
                pip_install(str(target), args, prog.log, say)
                prog.detail("deps", "Installing ComfyUI requirements…")
                pip_install(str(target),
                            ["-r", str(comfy_dir / "requirements.txt")],
                            prog.log, say)
            cfg["python"] = str(target)
            for eng in ENGINES.values():
                if not engine_enabled(cfg, eng["id"]):
                    continue
                node_reqs = (Path(cfg["comfy_dir"]) / "custom_nodes"
                             / eng["node_dir"] / "requirements.txt")
                if node_reqs.exists():
                    prog.detail("deps",
                                f"Installing the {eng['label']} requirements…")
                    pip_install(str(target), ["-r", str(node_reqs)], prog.log,
                                lambda t, pct: prog.detail("deps", t, pct))
                else:
                    prog.log(f"No requirements.txt in {eng['node_dir']} — "
                             "skipping.")
            prog.finish("deps", f"Installed into {Path(cfg['python']).name}")

        # 5. models --------------------------------------------------------- #
        prog.begin("models")
        todo = missing_models(models_dir, cfg)
        if not todo:
            prog.finish("models", "Everything is already downloaded")
        else:
            prog.log(f"{len(todo)} model folder(s) to fetch")
            for i, m in enumerate(todo):
                prog.detail("models", f"Downloading {m['repo']}…")
                # download_repo has worked the percentage out all along; it
                # used to be handed to a lambda that dropped it on the floor.
                # Spread each folder across its share of the whole step, so
                # the bar crosses the run once instead of restarting per repo.
                span, base = 100.0 / len(todo), 100.0 * i / len(todo)
                download_repo(
                    cfg, m["repo"], models_dir, engine=m.get("engine", ""),
                    on_detail=lambda d, pct, _b=base, _s=span:
                        prog.detail("models", d, _b + (pct or 0) * _s / 100.0))
            prog.finish("models", "Voices and models ready")

        # 6. launch --------------------------------------------------------- #
        prog.begin("launch")
        url = cfg["comfy_url"]
        if mode == "external" or not cfg.get("auto_start_comfy", True):
            if not comfy_online(url):
                raise RuntimeError(f"ComfyUI is not answering at {url}.")
        elif comfy_online(url):
            # It was already up when this run installed the nodes into it, and
            # ComfyUI reads custom_nodes once, at startup — so it is running
            # without them, which is the "Nodes not loaded" the Engine panel
            # then reports with nothing the user can do about it. Bounce it.
            if comfy.alive():
                prog.detail("launch", "Restarting ComfyUI so it loads the "
                                      "Qwen-TTS nodes…")
                prog.log("Restarting ComfyUI so it picks up the nodes")
                comfy.stop()
                for _ in range(30):
                    if not comfy_online(url):
                        break
                    time.sleep(1)
                comfy.start(cfg["python"], Path(cfg["comfy_dir"]),
                            comfy_port(url), prog)
                if not wait_for_comfy(url, timeout=900):
                    raise RuntimeError(
                        "ComfyUI did not come back after the restart.\n"
                        + "\n".join(comfy.tail(25)))
            else:
                prog.log("ComfyUI is already running, and Script Builder did "
                         "not start it — restart it yourself so it loads the "
                         "Qwen-TTS nodes.")
        else:
            comfy.start(cfg["python"], Path(cfg["comfy_dir"]),
                        comfy_port(url), prog)
            prog.detail("launch", "Waiting for ComfyUI — the first start is slow…")
            if not wait_for_comfy(url, timeout=900):
                raise RuntimeError("ComfyUI did not start within 15 minutes.\n"
                                   + "\n".join(comfy.tail(25)))
        prog.finish("launch", url)

        cfg["setup_complete"] = True
        save_config(cfg)
        prog.done = True
        prog.log("Setup complete. Script Builder is ready.")
    except Exception as exc:  # noqa: BLE001
        prog.error = str(exc)
        if prog.step:
            prog.fail(prog.step, str(exc))
        prog.log(f"FAILED: {exc}")
    finally:
        prog.running = False
