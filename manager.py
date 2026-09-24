"""
manager.py - what the front end needs to get the machine ready.

Dependencies: check Python, Git, ComfyUI, the Qwen-TTS nodes, PyTorch,
ComfyUI's packages, the node's own packages (transformers in particular), the
model folders and the running engine — and install any of them on request.

HuggingFace: browse a repo, pull a whole model folder into
ComfyUI/models/qwen-tts/<Name>/, show progress, cancel, delete. Repo, token and
mirror are all set from the page; nothing here needs a terminal.
"""

from __future__ import annotations

import array
import io
import os
import platform
import shutil
import subprocess
import threading
import time
import uuid
import wave
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
def no_cuda_reason(version: str, cpu_only: bool | None = None) -> str:
    """Why torch cannot see a GPU, in the words that fit this machine.

    "no GPU found" was reported to someone holding an RTX 4060, because all
    the check knew was that torch.cuda.is_available() came back False. The
    usual cause is the build: a wheel tagged +cpu has no CUDA in it at all and
    never will, whatever hardware is underneath. Look at the machine before
    blaming it.

    `cpu_only` is what torch itself says (torch.version.cuda is None), and
    wins over the tag when given: PyPI's wheels carry no tag, so on Windows
    the CPU build reads as a plain "2.14.0".
    """
    gpu = bootstrap.nvidia_gpu()
    build = version.split("+")[1] if "+" in version else ""
    if cpu_only is None:
        cpu_only = build == "cpu"
    # What the picker is set to now. Told to "pick the NVIDIA build above" by a
    # panel whose picker already reads "Automatic — NVIDIA GeForce RTX 4060
    # (CUDA build)", the only honest next move is the one button that is left,
    # and the sentence should say so rather than send them hunting.
    fix = ("Press Reinstall." if bootstrap.torch_build(bootstrap.torch_index({}))
           .startswith("cu") else "Pick the NVIDIA build above and press "
                                  "Reinstall.")
    if gpu["name"] and cpu_only:
        return (f"torch {version} — this is the CPU-only build, but {gpu['name']} "
                f"is here. {fix}")
    if gpu["name"] and not gpu["driver"]:
        return (f"torch {version} — {gpu['name']} is here but its driver is not "
                "answering. Install the NVIDIA driver, then press Recheck.")
    if gpu["name"]:
        return (f"torch {version} — {gpu['name']} is here but this build cannot "
                f"use it. {fix}")
    if cpu_only:
        return (f"torch {version} — the CPU-only build, and no NVIDIA GPU was "
                "found. Speech will be slow.")
    return f"torch {version} — no NVIDIA GPU found, speech will be slow."


def _torch_row(py_comfy: str, suffix: str, label: str,
               cfg: dict | None = None) -> dict:
    """PyTorch as this engine's own environment has it."""
    if not py_comfy:
        return {"id": "torch" + suffix, "label": f"PyTorch · {label}",
                "state": "unknown", "detail": "Install ComfyUI first.",
                "action": "install"}
    kind = "portable python_embeded" if "python_embeded" in py_comfy \
        else "virtual environment"
    code, out = _probe(py_comfy,
                       "import torch,json;"
                       "print(json.dumps({'v':torch.__version__,"
                       "'cuda':torch.cuda.is_available(),"
                       "'built':torch.version.cuda,"
                       "'hip':getattr(torch.version,'hip',None),"
                       "'dev':(torch.cuda.get_device_name(0) "
                       "if torch.cuda.is_available() else '')}))")
    # Read before anything else: a damaged torch imports fine here and still
    # dies inside ComfyUI, and its version reads as exactly the right build —
    # which is how this row said "ok" over an engine that could not start.
    damage = bootstrap.torch_damage_summary(bootstrap.torch_damage(py_comfy))
    if damage:
        have = bootstrap.installed_torch(py_comfy)
        return {"id": "torch" + suffix, "label": f"PyTorch · {label}",
                "state": "warn", "repair": True, "action": "reinstall",
                "detail": (f"torch {have.get('version', '')} is damaged — "
                           f"{damage}. ComfyUI will not start on it. Press "
                           "Reinstall.")}
    if code != 0:
        have = bootstrap.installed_torch(py_comfy)
        if have:
            last = (out or "").strip().splitlines()[-1:] or [""]
            return {"id": "torch" + suffix, "label": f"PyTorch · {label}",
                    "state": "warn", "repair": True, "action": "reinstall",
                    "detail": (f"torch {have['version']} is installed but will "
                               f"not import: {last[0][:160]} Press Reinstall.")}
        return {"id": "torch" + suffix, "label": f"PyTorch · {label}",
                "state": "missing", "detail": f"Not installed in the {kind}.",
                "action": "install"}
    import json as _json
    try:
        d = _json.loads(out.splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"id": "torch" + suffix, "label": f"PyTorch · {label}",
                "state": "unknown", "detail": out[-140:], "action": "install"}
    if d["cuda"]:
        return {"id": "torch" + suffix, "label": f"PyTorch · {label}",
                "state": "ok", "detail": f"torch {d['v']} — GPU: {d['dev']}",
                "action": "reinstall"}
    cpu_only = not d.get("built") and not d.get("hip")
    row = {"id": "torch" + suffix, "label": f"PyTorch · {label}",
           "state": "warn", "detail": no_cuda_reason(d["v"], cpu_only),
           "action": "reinstall"}
    # A CPU-only build where there is an NVIDIA card and the CUDA build is
    # what the picker asks for is not a slow engine: it is one that stops as
    # it starts, and bootstrap.torch_launch refuses to launch it. So it raises
    # the Engine badge and Install everything missing repairs it, like a
    # missing row — a "warn" alone left that button saying nothing was wrong.
    if cpu_only and bootstrap.nvidia_gpu()["name"] and bootstrap.build_kind(
            bootstrap.torch_build(bootstrap.torch_index(cfg or {}))) == "cuda":
        row["repair"] = True
    return row


def same_install(comfy_dir: str, engine_root: str) -> bool:
    """Is the ComfyUI answering the address the one we are managing?"""
    if not comfy_dir or not engine_root:
        return True                      # nothing to compare: do not cry wolf
    try:
        return (Path(comfy_dir).resolve() == Path(engine_root).resolve()
                or os.path.normcase(os.path.normpath(comfy_dir))
                == os.path.normcase(os.path.normpath(engine_root)))
    except OSError:
        return True


def engine_row(label: str, suffix: str, url: str, online: bool,
               comfy_dir: str, engine_root: str) -> dict:
    """The Engine page's row for one engine, saying what actually answered.

    "ok" on its own reads as verified, and the identity check used to speak up
    only on a mismatch — so an engine that matches and an engine that will not
    say where it runs from looked identical. The second is exactly how another
    ComfyUI holding the port passes for a healthy one, with its nodes missing
    and the folders on disk all present.
    """
    if not online:
        return {"id": "engine" + suffix, "label": f"Engine · {label}",
                "state": "off",
                "detail": url + " — not running. Only the engine you are "
                                "using is kept up, so the other is not on the "
                                "card.",
                "action": "start"}
    if not same_install(comfy_dir, engine_root):
        return {"id": "engine" + suffix, "label": f"Engine · {label}",
                "state": "warn",
                "detail": f"{url} is answered by the ComfyUI in "
                          f"{engine_root}, not {label}'s own in {comfy_dir}. "
                          "Its nodes will read as missing however many times "
                          "they are installed. Stop that one, or give this "
                          "engine a free port in Settings.",
                "action": None}
    if engine_root:
        where = " — the ComfyUI in " + engine_root
    elif comfy_dir:
        where = (" — this engine does not say where it runs from, so it "
                 "cannot be confirmed as the one in " + comfy_dir)
    else:
        where = ""
    return {"id": "engine" + suffix, "label": f"Engine · {label}",
            "state": "ok", "detail": url + where, "action": None}


def dependencies(cfg: dict, clients=None, engine: str = "") -> list[dict]:
    """What each engine needs, engine by engine.

    They no longer share anything below ComfyUI — separate clones, separate
    environments, separate model folders, separate ports — so the report is
    per engine too. Python and Git are the only rows left that both use.

    `clients` is {engine id: ComfyClient} for the engines that are answering;
    a bare client is taken as the selected engine's, which is what callers
    written before the split still pass.
    """
    if clients is not None and not isinstance(clients, dict):
        clients = {engine or bootstrap.DEFAULT_ENGINE: clients}
    clients = clients or {}
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

    for eid, eng in ENGINES.items():
        label, suffix = eng["label"], "_" + eid
        if not bootstrap.engine_enabled(cfg, eid):
            items.append({"id": "comfyui" + suffix,
                          "label": f"ComfyUI · {label}", "state": "off",
                          "detail": "Turned off in Settings.", "action": None})
            continue
        slot = bootstrap.engine_cfg(cfg, eid)
        client = clients.get(eid)
        comfy_dir = Path(slot["comfy_dir"]) if slot.get("comfy_dir") else None

        # Its own ComfyUI ------------------------------------------------- #
        if comfy_dir and (comfy_dir / "main.py").exists():
            items.append({"id": "comfyui" + suffix,
                          "label": f"ComfyUI · {label}", "state": "ok",
                          "detail": str(comfy_dir), "action": "update"})
        else:
            items.append({"id": "comfyui" + suffix,
                          "label": f"ComfyUI · {label}", "state": "missing",
                          "detail": f"{label} has no ComfyUI of its own yet — "
                                    f"it would go in {APP_DIR / eng['dir_name']}.",
                          "action": "install"})

        # Its own nodes ---------------------------------------------------- #
        loaded = client.engine_ready(eid) if client else None
        if not (comfy_dir and bootstrap.node_installed(comfy_dir, eid)):
            if loaded:
                items.append({"id": "node" + suffix, "label": f"{label} nodes",
                              "state": "ok",
                              "detail": "Loaded by the ComfyUI you are "
                                        "running. Set its folder in Settings "
                                        "to manage them from here.",
                              "action": None})
            else:
                items.append({"id": "node" + suffix, "label": f"{label} nodes",
                              "state": "missing",
                              "detail": f"{eng['node_repo']} is not installed.",
                              "action": "install"})
        else:
            items.append({
                "id": "node" + suffix, "label": f"{label} nodes",
                "state": "ok" if loaded is not False else "warn",
                "detail": (str(comfy_dir / "custom_nodes" / eng["node_dir"])
                           if loaded is not False else
                           "Installed, but this ComfyUI started before they "
                           "were. Restart it so it loads them."),
                "action": "update" if loaded is not False else "restart"})

        # Its own environment ---------------------------------------------- #
        py_comfy = comfy_python(cfg, eid)
        items.append(_torch_row(py_comfy, suffix, label, cfg))

        if py_comfy:
            code, out = _probe(py_comfy,
                               "import transformers,librosa;"
                               "print(transformers.__version__)")
            if code != 0:
                items.append({"id": "node_reqs" + suffix,
                              "label": f"Speech packages · {label}",
                              "state": "missing",
                              "detail": "transformers or librosa is missing.",
                              "action": "install"})
            else:
                ver = out.splitlines()[-1].strip()
                major = int(ver.split(".")[0]) if ver[:1].isdigit() else 0
                # Qwen3-TTS is the strict one: 4.57.3, or 5.0 and up. MOSS asks
                # only for 4.40+, and now that they no longer share an
                # environment each is judged on its own floor.
                good = (ver.startswith("4.57.3") or major >= 5) if eid == "qwen" \
                    else (major >= 5 or ver >= "4.40")
                items.append({
                    "id": "node_reqs" + suffix,
                    "label": f"Speech packages · {label}",
                    "state": "ok" if good else "warn",
                    "detail": f"transformers {ver}" + ("" if good else
                              " — Qwen3-TTS needs 4.57.3, or 5.0 and up."),
                    "action": "install"})
        else:
            items.append({"id": "node_reqs" + suffix,
                          "label": f"Speech packages · {label}",
                          "state": "unknown", "detail": "Install ComfyUI first.",
                          "action": "install"})

        # Its own models ---------------------------------------------------- #
        models_dir = bootstrap.engine_models_dir(cfg, eid)
        if models_dir and models_dir.is_dir():
            missing = bootstrap.missing_models(models_dir, cfg, eid)
            need = [m for m in missing
                    if m["group"] in ("core", "preset", "moss_core")]
            if need:
                items.append({"id": "models" + suffix,
                              "label": f"Voices and models · {label}",
                              "state": "missing",
                              "detail": "Missing: " + ", ".join(m["repo"] for m in need),
                              "action": "models"})
            elif missing:
                items.append({"id": "models" + suffix,
                              "label": f"Voices and models · {label}",
                              "state": "warn",
                              "detail": "Optional: " + ", ".join(m["repo"]
                                                                 for m in missing),
                              "action": "models"})
            else:
                items.append({"id": "models" + suffix,
                              "label": f"Voices and models · {label}",
                              "state": "ok", "detail": "All folders present.",
                              "action": "models"})
        else:
            items.append({"id": "models" + suffix,
                          "label": f"Voices and models · {label}",
                          "state": "unknown",
                          "detail": "Set up this engine first.",
                          "action": "models"})

        # Is it up, and is it ours ------------------------------------------- #
        online = bootstrap.comfy_online(slot["comfy_url"])
        root = ""
        if online and client:
            try:
                root = client.engine_root()
            except Exception:  # noqa: BLE001
                root = ""
        items.append(engine_row(label, suffix, slot["comfy_url"], online,
                                slot.get("comfy_dir") or "", root))
    return items


# The installs that run pip in an engine's own environment.
PIP_STEPS = ("node", "torch", "node_reqs")
_INSTALL_LOCK = threading.Lock()


class InstallBusy(RuntimeError):
    """Another install is already writing this engine's environment."""


def dep_parts(dep_id: str) -> tuple[str, str]:
    """"torch_moss" -> ("torch", "moss"); a bare id is the default engine's."""
    base, _, eid = dep_id.rpartition("_")
    if eid not in ENGINES:
        base, eid = dep_id, bootstrap.DEFAULT_ENGINE
    return base, eid


def install_dependency(dep_id: str, cfg: dict, opts: dict,
                       stop_engine=None) -> Task:
    """Install one thing for one engine.

    Ids carry the engine — "torch_moss", "node_qwen" — because nothing below
    ComfyUI is shared any more. A bare id without a suffix is Qwen's, which is
    what a page written before the split would send.

    `stop_engine(engine)` stops that engine's ComfyUI if this app is running
    it, and says whether it did. Anything in PIP_STEPS calls it first: Windows
    will not let pip replace a file a running ComfyUI has loaded — torch's
    DLLs above all, which is exactly what a Reinstall has to replace — and the
    engine has to restart to use new packages anyway.

    One pip at a time per environment. A button that looked as if it had done
    nothing got pressed again, and the row below it too, and two pips writing
    the same site-packages — one of them uninstalling torch — break each
    other; on Windows the loser fails on a file the winner holds. So a second
    one is refused, naming the one that is running and how far it has got.
    """
    base, eid = dep_parts(dep_id)
    label = ENGINES[eid]["label"]
    titles = {"git": "Install Git",
              "comfyui": f"Install ComfyUI for {label}",
              "node": f"Install the {label} nodes",
              "torch": f"Install PyTorch for {label}",
              "node_reqs": f"Install the {label} packages"}
    opts = dict(opts, engine=eid)

    def run(task: Task) -> None:
        if base in PIP_STEPS and stop_engine and stop_engine(eid):
            task.log(f"Stopped {label}'s ComfyUI first — its packages are "
                     "about to change, and a running ComfyUI holds them open.")
        if base == "git":
            _install_git(task)
        elif base == "comfyui":
            _install_comfyui(task, cfg, eid)
        elif base == "node":
            _install_node(task, cfg, eid)
        elif base == "torch":
            _install_torch(task, cfg, opts)
        elif base == "node_reqs":
            _install_node_reqs(task, cfg, eid)
        else:
            raise RuntimeError(f"Nothing to install for '{dep_id}'.")

    with _INSTALL_LOCK:
        if base in PIP_STEPS:
            for other in TASKS.running("dependency"):
                kind, where = dep_parts(other.meta.get("dep", ""))
                if kind in PIP_STEPS and where == eid:
                    raise InstallBusy(
                        f"{other.title} is still running"
                        + (f" ({other.detail})" if other.detail else "")
                        + f" — one install at a time into {label}'s "
                          "environment. Wait for it to finish.")
        return spawn("dependency", titles.get(base, dep_id), run,
                     {"dep": dep_id, "engine": eid})


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


def _install_comfyui(task: Task, cfg: dict, engine: str = "qwen") -> None:
    if not have_git():
        raise RuntimeError("Install Git first.")
    slot = bootstrap.engine_cfg(cfg, engine)
    target = Path(slot["comfy_dir"]) if slot.get("comfy_dir") \
        else APP_DIR / ENGINES[engine]["dir_name"]
    if (target / "main.py").exists():
        task.set(detail="Updating ComfyUI…")
        stream(["git", "-C", str(target), "pull", "--ff-only"], task)
    else:
        task.set(detail="Downloading ComfyUI…")
        if stream(["git", "clone", "--depth", "1", bootstrap.COMFY_REPO,
                   str(target)], task) != 0:
            raise RuntimeError("git clone failed — see the log.")
    slot["comfy_dir"] = str(target)
    slot["models_dir"] = slot.get("models_dir") or str(target / "models")
    bootstrap.save_config(cfg)
    task.set(detail=str(target))


def _install_node(task: Task, cfg: dict, engine: str = "qwen") -> None:
    eng = ENGINES[engine]
    comfy_dir = Path(bootstrap.engine_cfg(cfg, engine).get("comfy_dir") or "")
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
    engine = opts.get("engine") or "qwen"
    slot = bootstrap.engine_cfg(cfg, engine)
    comfy_dir = Path(slot.get("comfy_dir") or "")
    if not (comfy_dir / "main.py").exists():
        raise RuntimeError("Install ComfyUI first.")
    target = portable_python(comfy_dir)
    # An install we did not make runs on its own environment. PyTorch goes in
    # there, beside the ComfyUI that will import it — never into a second
    # environment ComfyUI never loads. Probed once: each call runs the
    # candidates to see which of them is real.
    own = "" if (target or slot.get("managed")) \
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
            if stream([base, "-m", "venv", str(vpy.parents[1])], task) != 0:
                raise RuntimeError("Could not create the environment.")
        target = vpy
    slot["python"] = str(target)
    if opts.get("torch_index") is not None:
        cfg["torch_index"] = opts["torch_index"]
    bootstrap.save_config(cfg)
    say = _reporter(task)
    bootstrap.pip_install(str(target), ["--upgrade", "pip", "wheel"],
                          task.log, say)
    task.set(detail="Installing ComfyUI requirements…")
    # The build of PyTorch is what this button is for. A requirement of
    # ComfyUI's that will not install is worth reporting, but it is no reason
    # to leave the CPU build in place — which is what stopping here did, and
    # the row then read exactly as it had before the button was pressed.
    reqs_failed = ""
    try:
        bootstrap.pip_install(str(target),
                              ["-r", str(comfy_dir / "requirements.txt")],
                              task.log, say)
    except RuntimeError as exc:
        reqs_failed = str(exc)
        task.log(f"ComfyUI's requirements did not all install ({exc}) — "
                 "carrying on with PyTorch.")
    task.set(detail="Installing the selected PyTorch build — the long one…")
    bootstrap.install_requested_torch(str(target), cfg, task.log, say)
    if reqs_failed:
        raise RuntimeError("PyTorch is in place, but ComfyUI's own "
                           f"requirements did not all install — {reqs_failed}")
    task.set(detail="PyTorch installed. Start the engine to use it.")


def _install_node_reqs(task: Task, cfg: dict, engine: str = "") -> None:
    """Requirements for one engine's own ComfyUI."""
    engine = engine or "qwen"
    comfy_dir = Path(bootstrap.engine_cfg(cfg, engine).get("comfy_dir") or "")
    py = comfy_python(cfg, engine)
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
    # A node requirements file is allowed to name torch.  On Windows that can
    # silently swap a CUDA wheel for PyPI's CPU wheel, so restore the build the
    # GPU picker selected before calling the engine repaired.
    task.set(detail="Verifying the selected PyTorch build…")
    bootstrap.install_requested_torch(py, cfg, task.log, _reporter(task))
    task.set(detail="Packages installed. Restart ComfyUI.")



# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
# "Does this engine actually work?" is not a question the dependency report can
# answer. Every row there can read ok while the first take still fails: the
# folders can be present and truncated, the classes loaded from a version whose
# inputs have been renamed, the weights too big for the card. The only proof is
# one line of speech, generated here, on this machine.
#
# So this runs the whole path in order and stops at the first step that breaks,
# naming it. It also reads ComfyUI's own console over the run, which is the one
# way to see something the API never reports — a model reaching for HuggingFace
# mid-generation because a processor could not find its codec locally.

SELFTEST_LINE = "This is a short line, spoken once, to prove the engine works."

# What a folder must contain to be worth loading. .onnx and .gguf are here
# because a MOSS codec or a quantized backbone is no less a weight file.
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf",
                   ".onnx", ".npy"}


def _peak(raw: bytes, width: int) -> float:
    """Loudest sample as a fraction of full scale, 0.0 when it cannot be read.

    A file of the right length full of zeros decodes perfectly and plays
    nothing, which is exactly the failure worth catching here. audioop would
    do this in one call and was removed in Python 3.13.
    """
    if width != 2 or not raw:
        return -1.0
    try:
        samples = array.array("h")
        samples.frombytes(raw[:len(raw) - (len(raw) % 2)])
        return max(abs(s) for s in samples) / 32768.0 if samples else 0.0
    except Exception:  # noqa: BLE001
        return -1.0


class _Steps:
    """Ordered results, published to the task as each one lands."""

    def __init__(self, task: Task) -> None:
        self.task = task
        self.rows: list[dict] = []

    def add(self, sid: str, label: str, state: str, detail: str = "") -> dict:
        row = {"id": sid, "label": label, "state": state, "detail": detail}
        self.rows.append(row)
        self.task.log(f"{state.upper():5} {label}" + (f" — {detail}" if detail else ""))
        self.task.set(meta={**self.task.meta, "steps": list(self.rows)},
                      detail=f"{label}: {state}")
        return row

    def ok(self, sid, label, detail=""):
        return self.add(sid, label, "ok", detail)

    def fail(self, sid, label, detail=""):
        return self.add(sid, label, "fail", detail)

    def skip(self, sid, label, detail=""):
        return self.add(sid, label, "skip", detail)


def selftest(cfg: dict, client, engine: str, task: Task, tail=None) -> None:
    """Prove an engine end to end, or say exactly where it stops.

    `tail` returns the last lines of ComfyUI's console when we are the ones who
    started it; None when someone else's ComfyUI is in front of us and its
    output is theirs to read.
    """
    eng = ENGINES[engine]
    steps = _Steps(task)
    task.set(meta={**task.meta, "engine": engine, "steps": []})
    started_lines = len(tail(4000)) if tail else 0

    # 1. is anything there ---------------------------------------------- #
    here = bootstrap.engine_cfg(cfg, engine)
    if not bootstrap.comfy_online(here["comfy_url"]):
        steps.fail("engine", "ComfyUI is answering",
                   f"Nothing at {here['comfy_url']}. Start it from the Engine "
                   "panel first.")
        raise RuntimeError("ComfyUI is not running.")
    steps.ok("engine", "ComfyUI is answering", here["comfy_url"])

    # 2. did it load these nodes ----------------------------------------- #
    # Forced: the schema is cached for two minutes, and the whole point of
    # pressing Test is usually that something just changed — nodes installed,
    # ComfyUI restarted. Answering from a stale cache would report the state
    # of the world before the thing being tested.
    try:
        client.schema(force=True)
    except Exception as exc:  # noqa: BLE001
        steps.fail("nodes", f"{eng['label']} nodes are loaded",
                   f"Could not read ComfyUI's node list: {exc}"[:300])
        raise
    if not client.engine_ready(engine):
        why = ""
        if here.get("comfy_dir"):
            why = bootstrap.node_import_error(
                bootstrap.comfy_python(cfg, engine),
                Path(here["comfy_dir"]), engine)
        steps.fail("nodes", f"{eng['label']} nodes are loaded",
                   why or "ComfyUI has none of this engine's classes. If they "
                          "are installed, it started before they were — press "
                          "Restart engine.")
        raise RuntimeError(f"{eng['label']} nodes are not loaded.")
    steps.ok("nodes", f"{eng['label']} nodes are loaded")

    # 3. are the folders there, and whole -------------------------------- #
    models_dir = bootstrap.engine_models_dir(cfg, engine)
    if not models_dir or not models_dir.is_dir():
        steps.fail("models", "Model folders are on disk",
                   "No models folder is set — run setup, or set it in Settings.")
        raise RuntimeError("No models folder.")
    missing = bootstrap.missing_models(models_dir, cfg, engine)
    if missing:
        steps.fail("models", "Model folders are on disk",
                   "Missing: " + ", ".join(m["repo"] for m in missing))
        raise RuntimeError("Models are missing.")
    sizes = []
    for m in bootstrap.wanted_models(cfg, engine):
        folder = bootstrap.model_dir(models_dir, m["repo"], engine)
        sizes.append(f"{m['repo'].split('/')[-1]} "
                     f"{bootstrap.human_size(bootstrap.dir_size(folder))}")
        # model_installed() accepts a folder with only a config.json in it,
        # deliberately — a repo whose config landed first is still arriving.
        # By the time anyone presses Test, a folder with no weights in it is a
        # download that stopped, and it fails at load rather than here. Weight
        # files, not a byte count: the right floor for a tokenizer is not the
        # right floor for an 8B, and picking one number gets both wrong.
        if not any(f.suffix in WEIGHT_SUFFIXES for f in folder.rglob("*")):
            steps.fail("models", "Model folders are on disk",
                       f"{m['repo']} has no weights in it, only "
                       f"{', '.join(sorted({f.suffix or f.name for f in folder.rglob('*') if f.is_file()}))[:80]}"
                       " — delete it on the Models page and fetch it again.")
            raise RuntimeError("A model folder has no weights in it.")
    steps.ok("models", "Model folders are on disk", " · ".join(sizes))

    # 4. can we build a graph for it ------------------------------------- #
    opts = {"engine": engine, "prefer_wav": True, "style": ""}
    voice = {"kind": "preset"}
    if engine == "moss":
        opts["moss_dirs"] = {
            m["repo"]: str(bootstrap.model_dir(models_dir, m["repo"], "moss"))
            for m in bootstrap.wanted_models(cfg, "moss")
            if bootstrap.model_installed(models_dir, m["repo"], "moss")}
    else:
        speakers = client.speakers()
        voice["speaker"] = speakers[0] if speakers else ""
    try:
        built = client.build_line({"text": SELFTEST_LINE}, voice, opts)
    except Exception as exc:  # noqa: BLE001
        steps.fail("graph", "A graph can be built for it", str(exc)[:300])
        raise
    shape = " → ".join(built["prompt"][k]["class_type"]
                       for k in sorted(built["prompt"]))
    steps.ok("graph", "A graph can be built for it", shape)

    # 5. does ComfyUI accept it ------------------------------------------ #
    try:
        prompt_id = client.queue(built["prompt"])
    except Exception as exc:  # noqa: BLE001
        # This is where a node that renamed an input shows up: the graph is
        # well formed against the schema we read and rejected by the one
        # running.
        steps.fail("accepted", "ComfyUI accepts the graph", str(exc)[:400])
        raise
    steps.ok("accepted", "ComfyUI accepts the graph", f"prompt {prompt_id[:8]}")

    # 6. does audio come back -------------------------------------------- #
    task.set(detail="Generating one line — the first run loads the model, "
                    "which is the slow part…")
    began = time.time()
    outs: list[dict] = []
    while time.time() - began < 1800:
        if task.cancel:
            steps.skip("audio", "Speech comes back", "Cancelled.")
            return
        err = client.failed(prompt_id)
        if err:
            steps.fail("audio", "Speech comes back", err[:400])
            raise RuntimeError(err)
        outs = client.outputs(prompt_id)
        if outs:
            break
        time.sleep(2)
    took = time.time() - began
    if not outs:
        steps.fail("audio", "Speech comes back",
                   f"Nothing after {int(took / 60)} minutes.")
        raise RuntimeError("The engine produced nothing.")

    with client.view(outs[0]) as resp:
        resp.raise_for_status()
        raw = resp.content
    detail = f"{bootstrap.human_size(len(raw))} in {took:.0f}s"
    try:
        with wave.open(io.BytesIO(raw), "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
            width, chans = w.getsampwidth(), w.getnchannels()
            peak = _peak(w.readframes(frames), width)
        seconds = frames / float(rate or 1)
        detail = (f"{seconds:.1f}s of audio, {rate} Hz, "
                  f"{'mono' if chans == 1 else f'{chans}ch'} · {took:.0f}s to "
                  "generate")
        if seconds < 0.2:
            steps.fail("audio", "Speech comes back",
                       f"Only {seconds:.2f}s came back — too short to be the "
                       "line.")
            raise RuntimeError("The clip is too short.")
        if 0.0 <= peak < 0.005:
            # The right number of frames, all of them silence.
            steps.fail("audio", "Speech comes back",
                       f"{seconds:.1f}s of silence — the graph ran but the "
                       "model produced nothing audible.")
            raise RuntimeError("The clip is silent.")
        if peak >= 0:
            detail += f" · peak {peak * 100:.0f}%"
    except wave.Error:
        # Not a wav: SaveAudioAdvanced fell back to flac or opus, which is not
        # a failure — only a take that will be zipped rather than joined.
        detail += " (not wav — takes will be zipped instead of joined)"
    steps.ok("audio", "Speech comes back", detail)

    # 7. what ComfyUI said while it worked -------------------------------- #
    if tail:
        fresh = tail(4000)[started_lines:]
        pulled = [l for l in fresh
                  if "huggingface" in l.lower() or "Downloading" in l
                  or "%|" in l]
        if pulled:
            steps.add("console", "Ran without reaching for the network",
                      "warn",
                      "ComfyUI fetched something while generating — the model "
                      "found part of itself missing locally: "
                      + " / ".join(l.strip()[:80] for l in pulled[:3]))
        else:
            steps.ok("console", "Ran without reaching for the network")
    else:
        steps.skip("console", "Ran without reaching for the network",
                   "You start this ComfyUI yourself, so its console is not "
                   "ours to read.")
    task.set(detail=f"{eng['label']} works: " + detail)


# --------------------------------------------------------------------------- #
# huggingface
# --------------------------------------------------------------------------- #
def hf_browse(cfg: dict, repo: str, revision: str = "main") -> dict:
    files = bootstrap.hf_tree(cfg, repo, revision)
    keep = bootstrap.wanted_files(files)
    keep_paths = {f["path"] for f in keep}
    root = bootstrap.engine_models_dir(cfg, bootstrap.engine_of(repo))
    target = bootstrap.model_dir(root, repo) if root else None
    for f in files:
        f["needed"] = f["path"] in keep_paths
        f["installed"] = bool(target and (target / f["path"]).exists())
    files.sort(key=lambda f: (-f["size"], f["path"]))
    return {"repo": repo, "revision": revision, "files": files,
            "total": sum(f["size"] for f in keep),
            "target": str(target) if target else ""}


def hf_download_repo(cfg: dict, repo: str) -> Task:
    engine = bootstrap.engine_of(repo)
    models_dir = bootstrap.engine_models_dir(cfg, engine)
    if not models_dir:
        raise RuntimeError(
            f"{ENGINES[engine]['label']} has no models folder yet — set that "
            "engine up first, or set its folder in Settings.")
    if any(t.meta.get("repo") == repo for t in TASKS.running("download")):
        raise RuntimeError(f"{repo} is already downloading.")

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
            "partial": bootstrap.partial_download(folder),
            "path": str(folder)}


def local_models(cfg: dict) -> list[dict]:
    """Every model folder on disk, each engine read in its own install.

    Two things differ per engine and both matter: where the folder lives —
    each ComfyUI has its own models directory now — and its name. Qwen keeps
    <Name> alone, MOSS flattens to <Org>--<Name>. Reading one shape as the
    other gets every repo id wrong, which is how a downloaded model would read
    as never downloaded.
    """
    out: list[dict] = []
    for eid, eng in ENGINES.items():
        base = bootstrap.engine_models_dir(cfg, eid)
        if not base:
            continue
        root = base / eng["subdir"]
        if not root.is_dir():
            continue
        # A Qwen folder name has lost its org, so the table puts it back. One
        # the table does not know is given Qwen's, which is enough for
        # delete_model to find the same folder again. An org folder left over
        # from the old <Org>/<Name> shape is not itself a model.
        known = {m["repo"].split("/", 1)[1]: m["repo"] for m in eng["models"]}
        orgs = {m["repo"].split("/", 1)[0] for m in eng["models"]}
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            name = folder.name
            if name.startswith(".") or name in bootstrap.QWEN_RESERVED \
                    or name in orgs:
                continue
            if eng["layout"] == "org--name":
                repo = name.replace("--", "/", 1)
            else:
                repo = known.get(name) or f"Qwen/{name}"
            out.append(_folder_row(repo, eid, folder))
    return out


def delete_model(cfg: dict, repo: str) -> None:
    if "/" not in repo or ".." in repo:
        raise RuntimeError("That path is not allowed.")
    engine = bootstrap.engine_of(repo)
    base = bootstrap.engine_models_dir(cfg, engine)
    if not base:
        raise RuntimeError("No models folder is set for that engine.")
    root = (base / ENGINES[engine]["subdir"]).resolve()
    target = bootstrap.model_dir(base, repo, engine).resolve()
    if not str(target).startswith(str(root)):
        raise RuntimeError("That path is outside the models folder.")
    # A model is a folder *inside* the root. With no org folder in the way,
    # "Qwen/" would name the root itself, and "Qwen/voices" the node's own
    # saved voices.
    if target == root or target.parent != root \
            or target.name in bootstrap.QWEN_RESERVED:
        raise RuntimeError("That is not a model folder.")
    if not target.is_dir():
        raise RuntimeError("That folder is already gone.")
    shutil.rmtree(target)


REQUIRED_GROUPS = ("core", "preset", "moss_core")


def curated(cfg: dict, vram_mb: int = 0) -> list[dict]:
    """Every folder both engines know about, with what it is for, whether this
    setup has asked for it, and whether the card can actually run it."""
    wanted = {m["repo"] for m in bootstrap.wanted_models(cfg)}
    if not vram_mb:
        vram_mb = bootstrap.nvidia_gpu().get("vram_mb") or 0
    out = []
    for eid, eng in ENGINES.items():
        root = bootstrap.engine_models_dir(cfg, eid)
        for m in eng["models"]:
            out.append({**m, "engine": eid, "engine_label": eng["label"],
                        "role": "required" if m["group"] in REQUIRED_GROUPS
                                else "wanted" if m["repo"] in wanted
                                else "optional",
                        # None where the card is unknown: an unknown card is
                        # not a small one, and hiding a model because
                        # nvidia-smi was missing is rule 5b in a new coat.
                        "fits": bootstrap.fits_vram(m.get("vram_gb") or 0,
                                                    vram_mb),
                        "installed": bool(root)
                        and bootstrap.model_installed(root, m["repo"], eid)})
    return out
