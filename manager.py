"""
manager.py - what the front end needs to get the machine ready.

Dependencies: check Python, Git, ComfyUI, the Qwen-TTS nodes, PyTorch,
ComfyUI's packages, the node's own packages (transformers in particular), the
model folders and the running engine — and install any of them on request.

HuggingFace: browse a repo, pull a whole model folder into
ComfyUI/models/qwen-tts/Qwen/, show progress, cancel, delete. Repo, token and
mirror are all set from the page; nothing here needs a terminal.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import bootstrap
from bootstrap import (APP_DIR, ENGINES, comfy_python, have_git,
                       portable_python, venv_python)

DEFAULT_ENDPOINT = bootstrap.HF_BASE


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #
class Task:
    def __init__(self, kind: str, title: str, meta: dict | None = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.title = title
        self.meta = meta or {}
        self.state = "running"
        self.pct = 0.0
        self.detail = ""
        self.lines: list[str] = []
        self.created = time.time()
        self.cancel = False
        self._lock = threading.Lock()

    def log(self, msg: str) -> None:
        with self._lock:
            self.lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            if len(self.lines) > 1200:
                del self.lines[:600]

    def set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def view(self, since: int = 0) -> dict:
        with self._lock:
            return {"id": self.id, "kind": self.kind, "title": self.title,
                    "meta": self.meta, "state": self.state,
                    "pct": round(self.pct, 1), "detail": self.detail,
                    "created": self.created, "cursor": len(self.lines),
                    "lines": self.lines[since:]}


class Tasks:
    def __init__(self) -> None:
        self._items: dict[str, Task] = {}
        self._lock = threading.Lock()

    def add(self, task: Task) -> Task:
        with self._lock:
            self._items[task.id] = task
            finished = sorted((t for t in self._items.values()
                               if t.state != "running"), key=lambda t: t.created)
            for old in finished[:-40]:
                self._items.pop(old.id, None)
        return task

    def get(self, task_id: str) -> Task | None:
        return self._items.get(task_id)

    def list(self) -> list[Task]:
        with self._lock:
            return sorted(self._items.values(), key=lambda t: t.created,
                          reverse=True)

    def running(self, kind: str = "") -> list[Task]:
        return [t for t in self.list()
                if t.state == "running" and (not kind or t.kind == kind)]


TASKS = Tasks()


def spawn(kind: str, title: str, fn, meta: dict | None = None) -> Task:
    task = TASKS.add(Task(kind, title, meta))

    def wrapper():
        try:
            fn(task)
            if task.state == "running":
                task.set(state="done", pct=100)
        except Exception as exc:  # noqa: BLE001
            task.log(f"FAILED: {exc}")
            task.set(state="error", detail=str(exc))

    threading.Thread(target=wrapper, daemon=True).start()
    return task


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #
def stream(cmd: list[str], task: Task, keep: tuple[str, ...] = ()) -> int:
    task.log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        if not keep or line.startswith(keep):
            task.log(line[:220])
        if task.cancel:
            proc.terminate()
            task.set(state="cancelled", detail="Cancelled")
            return 1
    return proc.wait()


def _probe(python: str, code: str, timeout: int = 90) -> tuple[int, str]:
    if not python or not Path(python).exists():
        return 1, "no interpreter"
    try:
        out = subprocess.run([python, "-c", code], capture_output=True,
                             text=True, timeout=timeout)
        return out.returncode, (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


# --------------------------------------------------------------------------- #
# dependency report
# --------------------------------------------------------------------------- #
def no_cuda_reason(version: str) -> str:
    """Why torch cannot see a GPU, in the words that fit this machine.

    "no GPU found" was reported to someone holding an RTX 4060, because all
    the check knew was that torch.cuda.is_available() came back False. The
    usual cause is the build: a wheel tagged +cpu has no CUDA in it at all and
    never will, whatever hardware is underneath. Look at the machine before
    blaming it.
    """
    gpu = bootstrap.nvidia_gpu()
    build = version.split("+")[1] if "+" in version else ""
    if gpu["name"] and build == "cpu":
        return (f"torch {version} — this is the CPU-only build, but {gpu['name']} "
                "is here. Pick the NVIDIA build above and press Reinstall.")
    if gpu["name"] and not gpu["driver"]:
        return (f"torch {version} — {gpu['name']} is here but its driver is not "
                "answering. Install the NVIDIA driver, then press Recheck.")
    if gpu["name"]:
        return (f"torch {version} — {gpu['name']} is here but this build cannot "
                "use it. Pick the NVIDIA build above and press Reinstall.")
    if build == "cpu":
        return (f"torch {version} — the CPU-only build, and no NVIDIA GPU was "
                "found. Speech will be slow.")
    return f"torch {version} — no NVIDIA GPU found, speech will be slow."


def dependencies(cfg: dict, client=None) -> list[dict]:
    items: list[dict] = []

    try:
        py = bootstrap.find_python()
        items.append({"id": "python", "label": "Python 3.10+", "state": "ok",
                      "detail": py, "action": None})
    except Exception as exc:  # noqa: BLE001
        items.append({"id": "python", "label": "Python 3.10+",
                      "state": "missing", "detail": str(exc), "action": None,
                      "hint": "Install it from python.org, then press Recheck."})

    git = shutil.which("git") or ""
    items.append({"id": "git", "label": "Git", "state": "ok" if git else "missing",
                  "detail": git or "Needed to download ComfyUI and the nodes.",
                  "action": None if git else "install"})

    comfy_dir = Path(cfg["comfy_dir"]) if cfg.get("comfy_dir") else None
    if comfy_dir and (comfy_dir / "main.py").exists():
        items.append({"id": "comfyui", "label": "ComfyUI", "state": "ok",
                      "detail": str(comfy_dir), "action": "update"})
    else:
        items.append({"id": "comfyui", "label": "ComfyUI", "state": "missing",
                      "detail": "Not installed yet.", "action": "install"})

    # One row per engine. The ids are what /api/deps/<id>/install takes, so
    # they are the engine ids rather than a single "node".
    for eid, eng in ENGINES.items():
        dep_id = "node" if eid == "qwen" else f"node_{eid}"
        if not bootstrap.engine_enabled(cfg, eid):
            items.append({"id": dep_id, "label": f"{eng['label']} nodes",
                          "state": "off",
                          "detail": "Turned off in Settings.", "action": None})
            continue
        loaded = client.engine_ready(eid) if client else None
        if not (comfy_dir and bootstrap.node_installed(comfy_dir, eid)):
            if loaded:
                # The running ComfyUI has the classes, so they are installed —
                # we simply cannot see where, because this is someone's own
                # ComfyUI and setup was never told its folder. Reporting that
                # as "missing" put two red rows and an Install button in front
                # of someone whose engine was working perfectly.
                items.append({"id": dep_id, "label": f"{eng['label']} nodes",
                              "state": "ok",
                              "detail": "Loaded by the ComfyUI you are running. "
                                        "Set its folder in Settings to manage "
                                        "them from here.",
                              "action": None})
                continue
            items.append({"id": dep_id, "label": f"{eng['label']} nodes",
                          "state": "missing",
                          "detail": f"{eng['node_repo']} is not installed.",
                          "action": "install"})
            continue
        items.append({
            "id": dep_id, "label": f"{eng['label']} nodes",
            "state": "ok" if loaded is not False else "warn",
            # ComfyUI reads custom_nodes once, at startup, so the usual cause
            # is an engine that was already running when they were installed.
            # Restart is the fix and the diagnosis both: if they still do not
            # load, that task imports them and reports the real exception.
            "detail": (str(comfy_dir / "custom_nodes" / eng["node_dir"])
                       if loaded is not False else
                       "Installed, but this ComfyUI started before they were. "
                       "Restart it so it loads them."),
            "action": "update" if loaded is not False else "restart"})

    py_comfy = comfy_python(cfg)
    if py_comfy:
        kind = "portable python_embeded" if "python_embeded" in py_comfy \
            else "virtual environment"
        code, out = _probe(py_comfy,
                           "import torch,json;"
                           "print(json.dumps({'v':torch.__version__,"
                           "'cuda':torch.cuda.is_available(),"
                           "'dev':(torch.cuda.get_device_name(0) "
                           "if torch.cuda.is_available() else '')}))")
        if code != 0:
            items.append({"id": "torch", "label": "PyTorch", "state": "missing",
                          "detail": f"Not installed in the {kind}.",
                          "action": "install"})
        else:
            import json as _json
            try:
                d = _json.loads(out.splitlines()[-1])
                if d["cuda"]:
                    items.append({"id": "torch", "label": "PyTorch", "state": "ok",
                                  "detail": f"torch {d['v']} — GPU: {d['dev']}",
                                  "action": "reinstall"})
                else:
                    items.append({"id": "torch", "label": "PyTorch",
                                  "state": "warn",
                                  "detail": no_cuda_reason(d["v"]),
                                  "action": "reinstall"})
            except Exception:
                items.append({"id": "torch", "label": "PyTorch", "state": "unknown",
                              "detail": out[-140:], "action": "install"})
    else:
        items.append({"id": "torch", "label": "PyTorch", "state": "unknown",
                      "detail": "Install ComfyUI first.", "action": "install"})

    # The node's own packages. transformers is the one that actually breaks:
    # Qwen3-TTS wants 4.57.3, or 5.0 and up.
    if py_comfy:
        code, out = _probe(py_comfy,
                           "import transformers,librosa,accelerate;"
                           "print(transformers.__version__)")
        if code != 0:
            items.append({"id": "node_reqs", "label": "Qwen-TTS packages",
                          "state": "missing",
                          "detail": "transformers, librosa or accelerate is "
                                    "missing.", "action": "install"})
        else:
            ver = out.splitlines()[-1].strip()
            major = int(ver.split(".")[0]) if ver[:1].isdigit() else 0
            good = ver.startswith("4.57.3") or major >= 5
            items.append({
                "id": "node_reqs", "label": "Qwen-TTS packages",
                "state": "ok" if good else "warn",
                "detail": f"transformers {ver}" + ("" if good else
                          " — Qwen3-TTS needs 4.57.3, or 5.0 and up."),
                "action": "install"})
    else:
        items.append({"id": "node_reqs", "label": "Qwen-TTS packages",
                      "state": "unknown", "detail": "Install ComfyUI first.",
                      "action": "install"})

    models_dir = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    if models_dir and models_dir.is_dir():
        missing = bootstrap.missing_models(models_dir, cfg)
        need = [m for m in missing if m["group"] in ("core", "preset")]
        if need:
            items.append({"id": "models", "label": "Voices and models",
                          "state": "missing",
                          "detail": "Missing: " + ", ".join(m["repo"] for m in need),
                          "action": "models"})
        elif missing:
            items.append({"id": "models", "label": "Voices and models",
                          "state": "warn",
                          "detail": "Optional: " + ", ".join(m["repo"]
                                                             for m in missing),
                          "action": "models"})
        else:
            items.append({"id": "models", "label": "Voices and models",
                          "state": "ok", "detail": "All folders present.",
                          "action": "models"})
    else:
        items.append({"id": "models", "label": "Voices and models",
                      "state": "unknown", "detail": "Set the models folder first.",
                      "action": "models"})

    online = bootstrap.comfy_online(cfg["comfy_url"])
    items.append({"id": "engine", "label": "Engine",
                  "state": "ok" if online else "missing",
                  "detail": cfg["comfy_url"] if online
                  else "ComfyUI is not answering.",
                  "action": None if online else "start"})
    return items


# --------------------------------------------------------------------------- #
# installers
# --------------------------------------------------------------------------- #
def install_dependency(dep_id: str, cfg: dict, opts: dict) -> Task:
    titles = {"git": "Install Git", "comfyui": "Install ComfyUI",
              "node": "Install the Qwen3-TTS nodes",
              "node_moss": "Install the MOSS-TTS nodes",
              "torch": "Install PyTorch",
              "node_reqs": "Install the speech packages"}

    def run(task: Task) -> None:
        if dep_id == "git":
            _install_git(task)
        elif dep_id == "comfyui":
            _install_comfyui(task, cfg)
        elif dep_id == "node":
            _install_node(task, cfg, "qwen")
        elif dep_id.startswith("node_") and dep_id[5:] in ENGINES:
            _install_node(task, cfg, dep_id[5:])
        elif dep_id == "torch":
            _install_torch(task, cfg, opts)
        elif dep_id == "node_reqs":
            _install_node_reqs(task, cfg)
        else:
            raise RuntimeError(f"Nothing to install for '{dep_id}'.")

    return spawn("dependency", titles.get(dep_id, dep_id), run, {"dep": dep_id})


def _reporter(task: Task):
    """pip progress into the task the Engine panel is showing.

    A pct of None is 0 here, and the panel hides a bar at 0 — the same reason
    Progress.detail clears its own: a bar left at 100% while pip unpacks for
    ten silent minutes reads as a run that finished and hung.
    """
    def say(text: str, pct: float | None) -> None:
        task.set(detail=text, pct=0.0 if pct is None else pct)
    return say


def _install_git(task: Task) -> None:
    cmds = {"Windows": ["winget", "install", "--id", "Git.Git", "-e",
                        "--source", "winget", "--accept-package-agreements",
                        "--accept-source-agreements"],
            "Darwin": ["brew", "install", "git"],
            "Linux": ["sudo", "apt-get", "install", "-y", "git"]}
    cmd = cmds.get(platform.system())
    if not cmd or not shutil.which(cmd[0]):
        raise RuntimeError("Git has to be installed by hand on this system. "
                           "Install it, then press Recheck.")
    if stream(cmd, task) != 0:
        raise RuntimeError("The Git installer did not finish.")
    task.set(detail="Git installed. Restart Script Builder if it is still "
                    "reported as missing.")


def _install_comfyui(task: Task, cfg: dict) -> None:
    if not have_git():
        raise RuntimeError("Install Git first.")
    target = Path(cfg["comfy_dir"]) if cfg.get("comfy_dir") else APP_DIR / "ComfyUI"
    if (target / "main.py").exists():
        task.set(detail="Updating ComfyUI…")
        stream(["git", "-C", str(target), "pull", "--ff-only"], task)
    else:
        task.set(detail="Downloading ComfyUI…")
        if stream(["git", "clone", "--depth", "1", bootstrap.COMFY_REPO,
                   str(target)], task) != 0:
            raise RuntimeError("git clone failed — see the log.")
    cfg["comfy_dir"] = str(target)
    cfg["models_dir"] = cfg.get("models_dir") or str(target / "models")
    bootstrap.save_config(cfg)
    task.set(detail=str(target))


def _install_node(task: Task, cfg: dict, engine: str = "qwen") -> None:
    eng = ENGINES[engine]
    comfy_dir = Path(cfg.get("comfy_dir") or "")
    if not (comfy_dir / "main.py").exists():
        raise RuntimeError("Install ComfyUI first.")
    if not have_git():
        raise RuntimeError("Install Git first.")
    node_path = comfy_dir / "custom_nodes" / eng["node_dir"]
    if node_path.exists():
        task.set(detail=f"Updating the {eng['label']} nodes…")
        stream(["git", "-C", str(node_path), "pull", "--ff-only"], task)
    else:
        node_path.parent.mkdir(parents=True, exist_ok=True)
        task.set(detail=f"Downloading the {eng['label']} nodes…")
        if stream(["git", "clone", "--depth", "1", eng["node_repo"],
                   str(node_path)], task) != 0:
            raise RuntimeError("git clone failed — see the log.")
    _install_node_reqs(task, cfg, engine)
    task.set(detail="Installed. Restart ComfyUI so it loads the new nodes.")


def _install_torch(task: Task, cfg: dict, opts: dict) -> None:
    comfy_dir = Path(cfg.get("comfy_dir") or "")
    if not (comfy_dir / "main.py").exists():
        raise RuntimeError("Install ComfyUI first.")
    target = portable_python(comfy_dir)
    # An install we did not make runs on its own environment. PyTorch goes in
    # there, beside the ComfyUI that will import it — never into a second
    # environment ComfyUI never loads. Probed once: each call runs the
    # candidates to see which of them is real.
    own = "" if (target or cfg.get("managed")) \
        else bootstrap.existing_python(comfy_dir)
    if target:
        task.log(f"Portable ComfyUI — installing into {target}")
    elif own:
        target = Path(own)
        task.log(f"That ComfyUI runs on {target} — installing into it")
    else:
        vpy = venv_python(comfy_dir)
        if not vpy.exists():
            task.set(detail="Creating the Python environment…")
            base = bootstrap.find_python()
            if stream([base, "-m", "venv",
                       str(comfy_dir.parent / "comfy-venv")], task) != 0:
                raise RuntimeError("Could not create the environment.")
        target = vpy
    cfg["python"] = str(target)
    bootstrap.save_config(cfg)
    if opts.get("torch_index") is not None:
        cfg["torch_index"] = opts["torch_index"]
    index = bootstrap.torch_index(cfg)
    say = _reporter(task)
    task.set(detail="Installing PyTorch — this is the long one…")
    bootstrap.pip_install(str(target), ["--upgrade", "pip", "wheel"],
                          task.log, say)
    bootstrap.drop_mismatched_torch(str(target), index, task.log)
    args = ["torch", "torchaudio"]
    if index:
        args += ["--index-url", index]
    bootstrap.pip_install(str(target), args, task.log, say)
    task.set(detail="Installing ComfyUI requirements…")
    bootstrap.pip_install(str(target),
                          ["-r", str(comfy_dir / "requirements.txt")],
                          task.log, say)
    task.set(detail="PyTorch installed.")


def _install_node_reqs(task: Task, cfg: dict, engine: str = "") -> None:
    """Requirements for one engine, or for every engine that is installed."""
    comfy_dir = Path(cfg.get("comfy_dir") or "")
    py = comfy_python(cfg)
    if not py:
        raise RuntimeError("Install ComfyUI and PyTorch first.")
    todo = [ENGINES[engine]] if engine else [
        e for e in ENGINES.values()
        if bootstrap.engine_enabled(cfg, e["id"])
        and (comfy_dir / "custom_nodes" / e["node_dir"]
             / "requirements.txt").exists()]
    if not todo:
        raise RuntimeError("No speech nodes are installed yet.")
    for eng in todo:
        reqs = comfy_dir / "custom_nodes" / eng["node_dir"] / "requirements.txt"
        if not reqs.exists():
            raise RuntimeError(f"The {eng['label']} nodes are not installed yet.")
        task.set(detail=f"Installing the {eng['label']} requirements…")
        bootstrap.pip_install(py, ["-r", str(reqs)], task.log, _reporter(task))
    task.set(detail="Packages installed. Restart ComfyUI.")


# --------------------------------------------------------------------------- #
# huggingface
# --------------------------------------------------------------------------- #
def hf_browse(cfg: dict, repo: str, revision: str = "main") -> dict:
    files = bootstrap.hf_tree(cfg, repo, revision)
    keep = bootstrap.wanted_files(files)
    keep_paths = {f["path"] for f in keep}
    target = bootstrap.model_dir(Path(cfg["models_dir"]), repo) \
        if cfg.get("models_dir") else None
    for f in files:
        f["needed"] = f["path"] in keep_paths
        f["installed"] = bool(target and (target / f["path"]).exists())
    files.sort(key=lambda f: (-f["size"], f["path"]))
    return {"repo": repo, "revision": revision, "files": files,
            "total": sum(f["size"] for f in keep),
            "target": str(target) if target else ""}


def hf_download_repo(cfg: dict, repo: str) -> Task:
    if not cfg.get("models_dir"):
        raise RuntimeError("Set the ComfyUI models folder before downloading.")
    models_dir = Path(cfg["models_dir"])
    if any(t.meta.get("repo") == repo for t in TASKS.running("download")):
        raise RuntimeError(f"{repo} is already downloading.")

    engine = bootstrap.engine_of(repo)

    def run(task: Task) -> None:
        task.log(f"{repo} → {bootstrap.model_dir(models_dir, repo, engine)}")

        def detail(text: str, pct: float) -> None:
            task.set(detail=text, pct=pct)

        bootstrap.download_repo(cfg, repo, models_dir, on_detail=detail,
                                should_cancel=lambda: task.cancel,
                                engine=engine)
        if task.cancel:
            task.set(state="cancelled",
                     detail="Cancelled — what downloaded is kept, starting "
                            "again carries on from there.")
            return
        task.set(detail="Downloaded", pct=100)

    return spawn("download", repo.split("/")[-1], run, {"repo": repo})


def _folder_row(repo: str, engine: str, folder: Path) -> dict:
    size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    return {"repo": repo, "engine": engine, "size": size,
            "partial": any(f.suffix == ".part" for f in folder.rglob("*")),
            "path": str(folder)}


def local_models(cfg: dict) -> list[dict]:
    """Every model folder on disk, in both engines' layouts.

    The two are read differently on purpose: Qwen nests <Org>/<Name>, MOSS
    flattens to <Org>--<Name>. Walking one shape over the other lists nothing,
    which is how a downloaded MOSS folder would read as never downloaded.
    """
    base = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    out: list[dict] = []
    if not base:
        return out
    for eid, eng in ENGINES.items():
        root = base / eng["subdir"]
        if not root.is_dir():
            continue
        if eng["layout"] == "flat":
            for folder in sorted(p for p in root.iterdir() if p.is_dir()):
                out.append(_folder_row(folder.name.replace("--", "/", 1), eid,
                                       folder))
        else:
            for org in sorted(p for p in root.iterdir() if p.is_dir()):
                for folder in sorted(p for p in org.iterdir() if p.is_dir()):
                    out.append(_folder_row(f"{org.name}/{folder.name}", eid,
                                           folder))
    return out


def delete_model(cfg: dict, repo: str) -> None:
    if not cfg.get("models_dir"):
        raise RuntimeError("No models folder is set.")
    if "/" not in repo or ".." in repo:
        raise RuntimeError("That path is not allowed.")
    engine = bootstrap.engine_of(repo)
    root = (Path(cfg["models_dir"]) / ENGINES[engine]["subdir"]).resolve()
    target = bootstrap.model_dir(Path(cfg["models_dir"]), repo, engine).resolve()
    if not str(target).startswith(str(root)):
        raise RuntimeError("That path is outside the models folder.")
    if not target.is_dir():
        raise RuntimeError("That folder is already gone.")
    shutil.rmtree(target)


REQUIRED_GROUPS = ("core", "preset", "moss_core")


def curated(cfg: dict) -> list[dict]:
    """Every folder both engines know about, with what it is for and whether
    this setup has asked for it."""
    models_dir = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    wanted = {m["repo"] for m in bootstrap.wanted_models(cfg)}
    out = []
    for eid, eng in ENGINES.items():
        for m in eng["models"]:
            out.append({**m, "engine": eid, "engine_label": eng["label"],
                        "role": "required" if m["group"] in REQUIRED_GROUPS
                                else "wanted" if m["repo"] in wanted
                                else "optional",
                        "installed": bool(models_dir)
                        and bootstrap.model_installed(models_dir, m["repo"], eid)})
    return out
