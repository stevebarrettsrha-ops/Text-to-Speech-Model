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
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CONFIG_PATH = DATA_DIR / "config.json"

COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
NODE_REPO = "https://github.com/flybirdxx/ComfyUI-Qwen-TTS.git"
NODE_DIR_NAME = "ComfyUI-Qwen-TTS"

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
              "design": "want_voicedesign"}

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
    "setup_complete": False,
}

QWEN_SUBDIR = Path("qwen-tts")


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
        ("node", "Install the Qwen-TTS nodes"),
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
        self.steps = {k: {"label": v, "state": "pending", "detail": ""}
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

    def detail(self, key: str, detail: str) -> None:
        with self._lock:
            self.steps[key]["detail"] = detail

    def finish(self, key: str, detail: str = "") -> None:
        with self._lock:
            self.steps[key]["state"] = "done"
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


def qwen_model_dir(models_dir: Path, repo: str) -> Path:
    """models/qwen-tts/Qwen/<repo name> — the layout the node searches."""
    org, name = repo.split("/", 1)
    return models_dir / QWEN_SUBDIR / org / name


def model_installed(models_dir: Path, repo: str) -> bool:
    d = qwen_model_dir(models_dir, repo)
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


def wanted_models(cfg: dict) -> list[dict]:
    """The folders this setup actually needs, from what the person asked for."""
    out = []
    for m in MODEL_REPOS:
        flag = GROUP_FLAG.get(m["group"])
        if flag is None or cfg.get(flag):
            if m["group"] == "clone_hq" and not cfg.get("want_clone"):
                continue
            out.append(m)
    return out


def missing_models(models_dir: Path, cfg: dict) -> list[dict]:
    return [m for m in wanted_models(cfg)
            if not model_installed(models_dir, m["repo"])]


def node_installed(comfy_dir: Path) -> bool:
    return (comfy_dir / "custom_nodes" / NODE_DIR_NAME / "nodes.py").exists()


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
                  on_detail=None, should_cancel=None) -> None:
    """Pull a whole model folder into models/qwen-tts/Qwen/<name>."""
    target = qwen_model_dir(models_dir, repo)
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
                on_detail(f"{repo} · file {_i}/{len(files)} · "
                          f"{(_done + got)/1e9:.2f} of {total_bytes/1e9:.2f} GB",
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
def pip_install(python: str, args: list[str], log) -> None:
    cmd = [python, "-m", "pip", "install"] + args
    log("$ " + " ".join(cmd[:8]) + (" …" if len(cmd) > 8 else ""))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith(("Collecting", "Downloading", "Installing",
                            "Successfully", "ERROR", "Building",
                            "WARNING: ")):
            log(line[:200])
    if proc.wait() != 0:
        raise RuntimeError("pip install failed — see the log.")


def torch_index(cfg: dict) -> str:
    if cfg.get("torch_index"):
        return cfg["torch_index"]
    if platform.system() == "Darwin":
        return ""
    if shutil.which("nvidia-smi"):
        try:
            if _run(["nvidia-smi"], timeout=20).returncode == 0:
                return "https://download.pytorch.org/whl/cu128"
        except Exception:
            pass
    return "https://download.pytorch.org/whl/cpu"


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

        # 3. custom node --------------------------------------------------- #
        prog.begin("node")
        if comfy_dir is None:
            prog.finish("node", "Install the Qwen-TTS nodes in your own ComfyUI")
        else:
            nodes_dir = comfy_dir / "custom_nodes"
            node_path = nodes_dir / NODE_DIR_NAME
            if node_path.exists():
                prog.detail("node", "Updating the Qwen-TTS nodes…")
                _run(["git", "-C", str(node_path), "pull", "--ff-only"])
            else:
                if not have_git():
                    raise RuntimeError("Git is needed to install the Qwen-TTS "
                                       "nodes. Install it from the Engine panel.")
                nodes_dir.mkdir(parents=True, exist_ok=True)
                prog.detail("node", "Downloading the Qwen-TTS nodes…")
                prog.log(f"git clone {NODE_REPO}")
                res = _run(["git", "clone", "--depth", "1", NODE_REPO,
                            str(node_path)])
                if res.returncode != 0:
                    raise RuntimeError("git clone failed: " +
                                       (res.stderr or res.stdout)[-600:])
            prog.finish("node", str(node_path))

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
                prog.detail("deps", "Installing PyTorch — the long one…")
                pip_install(str(target), ["--upgrade", "pip", "wheel"], prog.log)
                args = ["torch", "torchaudio"]
                idx = torch_index(cfg)
                if idx:
                    args += ["--index-url", idx]
                pip_install(str(target), args, prog.log)
                prog.detail("deps", "Installing ComfyUI requirements…")
                pip_install(str(target),
                            ["-r", str(comfy_dir / "requirements.txt")],
                            prog.log)
            cfg["python"] = str(target)
            node_reqs = Path(cfg["comfy_dir"]) / "custom_nodes" / NODE_DIR_NAME \
                / "requirements.txt"
            if node_reqs.exists():
                prog.detail("deps", "Installing the Qwen-TTS requirements…")
                pip_install(str(target), ["-r", str(node_reqs)], prog.log)
            else:
                prog.log("No requirements.txt in the node folder — skipping.")
            prog.finish("deps", f"Installed into {Path(cfg['python']).name}")

        # 5. models --------------------------------------------------------- #
        prog.begin("models")
        todo = missing_models(models_dir, cfg)
        if not todo:
            prog.finish("models", "Everything is already downloaded")
        else:
            prog.log(f"{len(todo)} model folder(s) to fetch")
            for m in todo:
                prog.detail("models", f"Downloading {m['repo']}…")
                download_repo(cfg, m["repo"], models_dir,
                              on_detail=lambda d, _p: prog.detail("models", d))
            prog.finish("models", "Voices and models ready")

        # 6. launch --------------------------------------------------------- #
        prog.begin("launch")
        url = cfg["comfy_url"]
        if mode == "external" or not cfg.get("auto_start_comfy", True):
            if not comfy_online(url):
                raise RuntimeError(f"ComfyUI is not answering at {url}.")
        elif comfy_online(url):
            prog.log("ComfyUI is already running")
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
