"""
server.py - Script Builder backend.

Run:  python server.py        (opens http://127.0.0.1:7799)
"""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import threading
import time
import uuid
import wave
import webbrowser
import zipfile
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

import bootstrap
import manager
from bootstrap import (APP_DIR, DATA_DIR, ComfyProcess, Progress, clean_url,
                       comfy_online, comfy_port, detect_comfy_dirs,
                       load_config, save_config)
from comfy import ComfyClient, ComfyError

# DATA_DIR comes from bootstrap so the two cannot drift apart, and so
# SCRIPT_BUILDER_DATA moves both.
TAKES_DIR = DATA_DIR / "takes"
TAKES_PATH = DATA_DIR / "takes.json"
WEB_DIR = APP_DIR / "web"
PORT = int(os.environ.get("SCRIPT_BUILDER_PORT", "7799"))

app = Flask(__name__, static_folder=None)

cfg = load_config()
progress = Progress()
comfy_proc = ComfyProcess()
client = ComfyClient(cfg["comfy_url"])

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
takes_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# takes
# --------------------------------------------------------------------------- #
def _read_takes() -> list[dict]:
    """Callers hold takes_lock."""
    if not TAKES_PATH.exists():
        return []
    try:
        return json.loads(TAKES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_takes(items: list[dict]) -> None:
    """Callers hold takes_lock. Written beside the file and moved into place:
    takes.json is the whole library, and a half-written one is an empty one."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TAKES_PATH.with_name(TAKES_PATH.name + ".tmp")
    tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
    tmp.replace(TAKES_PATH)


def read_takes() -> list[dict]:
    with takes_lock:
        return _read_takes()


def write_takes(items: list[dict]) -> None:
    with takes_lock:
        _write_takes(items)


# Read, change, write — all inside one hold of the lock. Taking it twice with
# a gap in the middle meant two jobs finishing together each wrote the list
# they had read before the other's take was in it, and the loser vanished from
# the library while its audio stayed on disk for the orphan sweep to delete.
def add_take(take: dict) -> None:
    with takes_lock:
        items = _read_takes()
        items.insert(0, take)
        _write_takes(items[:200])


def remove_take(take_id: str) -> None:
    with takes_lock:
        _write_takes([t for t in _read_takes() if t["id"] != take_id])


def take_title(lines: list[dict]) -> str:
    for line in lines:
        text = (line.get("text") or "").strip()
        if text:
            words = text.split()
            return " ".join(words[:7]).strip(" ,.!?-") or "Untitled take"
    return "Untitled take"


def stitch_wavs(paths: list[Path], dest: Path, pause: float) -> bool:
    """Join clips into one wav with silence between. Returns False if the
    clips are not wav, or their formats do not line up."""
    try:
        with wave.open(str(paths[0]), "rb") as first:
            params = first.getparams()
        with wave.open(str(dest), "wb") as out:
            out.setparams(params)
            # Whole frames only. Rounding the byte count instead lets a pause
            # like 0.75s at 22050 Hz stereo end on half a frame, and every
            # sample after it lands in the wrong channel.
            frame = params.sampwidth * params.nchannels
            gap = b"\x00" * (int(params.framerate * max(pause, 0)) * frame)
            for i, p in enumerate(paths):
                with wave.open(str(p), "rb") as w:
                    if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != \
                            (params.nchannels, params.sampwidth, params.framerate):
                        # Raised, not returned: by the time a mismatch shows up
                        # the earlier clips are already written, and returning
                        # from inside the `with` left that half-built file on
                        # disk next to the zip the caller then made — a wav
                        # that looks like the take and holds one line of it.
                        raise ValueError("clip formats differ")
                    out.writeframes(w.readframes(w.getnframes()))
                if i < len(paths) - 1 and gap:
                    out.writeframes(gap)
        return True
    except Exception:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def zip_clips(paths: list[Path], dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.name)


# --------------------------------------------------------------------------- #
# generation job
# --------------------------------------------------------------------------- #
def wait_for_prompt(prompt_id: str, job_id: str, timeout: int = 900) -> list[dict]:
    started = time.time()
    while True:
        time.sleep(1.0)
        with jobs_lock:
            if jobs[job_id].get("cancelled"):
                client.interrupt()
                raise ComfyError("Cancelled")
        err = client.failed(prompt_id)
        if err:
            raise ComfyError(err)
        outs = client.outputs(prompt_id)
        if outs:
            return outs
        if time.time() - started > timeout:
            raise ComfyError("That line took more than 15 minutes. Check the "
                             "ComfyUI console.")


def gpu_vram(live=None) -> int:
    """The card's memory in MB, nvidia-smi first, ComfyUI second, 0 if unknown."""
    mb = bootstrap.nvidia_gpu().get("vram_mb") or 0
    if not mb and live is not None:
        try:
            mb = live.vram_mb()
        except Exception:  # noqa: BLE001
            mb = 0
    return mb


def moss_dirs() -> dict:
    """repo id -> the folder it is really in, for the folders that are there.

    The MOSS loader treats local_model_path as a path only when it can stat it
    and as a HuggingFace repo id otherwise, so handing it a folder that has not
    been downloaded turns into snapshot_download("D:\\...\\MOSS-TTS"), which is
    not a repo id and fails. A folder that is absent is left out here, and the
    node then fetches the model itself.
    """
    base = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    if not base:
        return {}
    out = {}
    for m in bootstrap.ENGINES["moss"]["models"]:
        folder = bootstrap.model_dir(base, m["repo"], "moss")
        if folder.is_dir() and bootstrap.model_installed(base, m["repo"], "moss"):
            out[m["repo"]] = str(folder)
    return out


def run_job(job_id: str, payload: dict) -> None:
    def set_state(**kw):
        with jobs_lock:
            jobs[job_id].update(kw)

    take_id = uuid.uuid4().hex[:12]
    folder = TAKES_DIR / take_id
    try:
        lines = payload.get("lines") or []
        speakers = payload.get("speakers") or {}
        engine = payload.get("engine") or cfg.get("engine") \
            or bootstrap.DEFAULT_ENGINE
        if engine not in bootstrap.ENGINES:
            engine = bootstrap.DEFAULT_ENGINE
        opts = {
            "engine": engine,
            "style": payload.get("style", ""),
            "model": payload.get("model", ""),
            "attention": payload.get("attention", "auto"),
            "unload": bool(payload.get("unload")),
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "language": payload.get("language", ""),
            "prefer_wav": True,
        }
        if engine == "moss":
            opts["moss_model"] = payload.get("moss_model") or ""
            opts["moss_dirs"] = moss_dirs()
        pause = float(payload.get("pause") or 0.5)
        folder.mkdir(parents=True, exist_ok=True)
        clips: list[Path] = []
        meta_lines: list[dict] = []

        for i, line in enumerate(lines):
            key = str(line.get("speaker", 1))
            # JSON object keys are strings, but a take loaded back can carry
            # integer ones. A key that is neither is a speaker we do not have,
            # not a reason to fail the whole job.
            voice = speakers.get(key) or {}
            if not voice and key.isdigit():
                voice = speakers.get(int(key)) or {}
            set_state(stage=f"Line {i + 1} of {len(lines)} · "
                            f"{voice.get('name') or 'Speaker ' + key}",
                      pct=round(i / max(len(lines), 1) * 100, 1),
                      line_index=i)

            built = client.build_line(line, voice, opts)
            prompt_id = client.queue(built["prompt"])
            outs = wait_for_prompt(prompt_id, job_id)
            item = outs[0]
            ext = Path(item["filename"]).suffix or ".wav"
            dest = folder / f"line_{i:03d}{ext}"
            with client.view(item) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(1024 * 256):
                        fh.write(chunk)
            clips.append(dest)
            meta_lines.append({"index": i, "speaker": int(key) if key.isdigit() else 1,
                               "text": line.get("text", ""), "file": dest.name})

        if not clips:
            raise ComfyError("There is nothing in the script to say.")

        set_state(stage="Joining the lines", pct=97)
        joined = folder / "take.wav"
        single = clips[0].suffix == ".wav" and stitch_wavs(clips, joined, pause)
        bundle = ""
        if not single:
            joined = folder / "take.zip"
            zip_clips(clips, joined)
            bundle = "zip"

        take = {
            "id": take_id, "title": payload.get("title") or take_title(lines),
            "style": payload.get("style", ""),
            "mode": payload.get("mode", "multi"),
            "created": time.time(),
            "engine": engine,
            "pause": pause, "model": opts["model"],
            "speakers": {k: {"name": v.get("name"), "kind": v.get("kind"),
                             "speaker": v.get("speaker")}
                         for k, v in speakers.items()},
            "lines": meta_lines,
            "file": joined.name, "bundle": bundle,
            "duration": None,
        }
        if single:
            try:
                with wave.open(str(joined), "rb") as w:
                    take["duration"] = round(w.getnframes() / w.getframerate(), 1)
            except Exception:
                pass
        add_take(take)
        set_state(status="done", pct=100, stage="Ready", take=take)
    # A job that does not finish records no take, so the clips it did fetch are
    # unreachable: nothing in the library lists them and no Delete can remove
    # them. Left behind, every failed run — and out of memory on line four is
    # the failure this app documents — costs another few megabytes for good.
    except ComfyError as exc:
        shutil.rmtree(folder, ignore_errors=True)
        if str(exc) == "Cancelled":
            set_state(status="cancelled", stage="Cancelled")
        else:
            set_state(status="error", error=str(exc), stage="Failed")
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(folder, ignore_errors=True)
        set_state(status="error", error=f"{type(exc).__name__}: {exc}",
                  stage="Failed")


# --------------------------------------------------------------------------- #
# app shell
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/web/<path:name>")
def web_asset(name: str):
    return send_from_directory(WEB_DIR, name)


# --------------------------------------------------------------------------- #
# status / setup
# --------------------------------------------------------------------------- #
@app.get("/api/status")
def api_status():
    online = comfy_online(cfg["comfy_url"])
    models_dir = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    # Whether the voices could be checked at all, which is not the same as
    # finding none missing. With no models folder set there is nowhere to look,
    # and an empty "missing" list used to read as "all present" — so a machine
    # with ComfyUI up, the nodes loaded and not one voice on disk reported the
    # engine ready and let someone press Read.
    models_known = bool(models_dir and models_dir.is_dir())
    engine = request.args.get("engine") or cfg.get("engine") \
        or bootstrap.DEFAULT_ENGINE
    if engine not in bootstrap.ENGINES:
        engine = bootstrap.DEFAULT_ENGINE
    payload = {
        "comfy_online": online,
        "engine": engine,
        "engines": [{"id": e["id"], "label": e["label"], "blurb": e["blurb"],
                     "enabled": bootstrap.engine_enabled(cfg, e["id"])}
                    for e in bootstrap.ENGINES.values()],
        "setup_complete": bool(cfg.get("setup_complete")),
        # So the Create page can say "Setting up…" rather than offer a setup
        # that is already running.
        "setup_running": bool(progress.running),
        "models_known": models_known,
        "detected": detect_comfy_dirs(),
        "config": {k: cfg.get(k) for k in
                   ("comfy_url", "comfy_dir", "models_dir", "managed",
                    "auto_start_comfy", "torch_index", "want_clone",
                    "want_17b", "want_voicedesign", "want_moss",
                    "want_moss_8b", "want_moss_design", "engine")},
        "nodes_ready": False, "ready": False,
    }
    # Readiness is per engine: with MOSS selected, a missing Qwen folder is
    # not what stands between this script and a take, and reporting it as one
    # sends people to download a model they are not about to use.
    missing = [m["repo"] for m in
               bootstrap.missing_models(models_dir, cfg, engine)] \
        if models_known else []
    payload["missing_models"] = missing
    if online:
        try:
            payload["nodes_ready"] = client.engine_ready(engine)
            payload["capabilities"] = client.capabilities(engine)
            payload["engine_nodes"] = {
                e: client.engine_ready(e) for e in bootstrap.ENGINES}
        except Exception as exc:  # noqa: BLE001
            payload["schema_error"] = str(exc)
    payload["ready"] = bool(online and payload["nodes_ready"]
                            and models_known and not missing)
    return jsonify(payload)


@app.get("/api/voices")
def api_voices():
    if not comfy_online(cfg["comfy_url"]):
        return jsonify({"error": "ComfyUI is not running."}), 503
    engine = request.args.get("engine") or cfg.get("engine") \
        or bootstrap.DEFAULT_ENGINE
    try:
        if engine == "moss":
            # No speaker enum exists on any MOSS node, so an empty list here is
            # the truth rather than a failed read — "fallback" stays False so
            # the page does not offer three Qwen names it cannot use.
            #
            # The picker carries repo ids, not the loader's display names: the
            # display name a repo maps to is read off the node when the graph
            # is built, and sending it back and forth through the browser would
            # be one more place for the two to drift apart.
            here = moss_dirs()
            vram = gpu_vram(client)
            models = []
            for m in bootstrap.ENGINES["moss"]["models"]:
                if m["group"] == "moss_core" and "Tokenizer" in m["repo"]:
                    continue
                fits = bootstrap.fits_vram(m.get("vram_gb") or 0, vram)
                label = f"{m['repo'].split('/')[-1]} · {m['params']}"
                if fits is False:
                    label += f" — needs ~{m['vram_gb']} GB, will not fit"
                elif m["repo"] not in here:
                    label += " (not downloaded)"
                models.append({"value": m["repo"], "label": label,
                               "fits": fits, "vram_gb": m.get("vram_gb")})
            return jsonify({
                "engine": "moss", "speakers": [], "fallback": False,
                "models": models,
                "variants": client.moss_variants(),
                "attentions": [],
                "capabilities": client.capabilities("moss"),
            })
        speakers = client.speakers()
        return jsonify({
            "engine": "qwen",
            "speakers": speakers or [],
            "fallback": not speakers,
            "models": client.models(),
            "attentions": client.attentions(),
            "capabilities": client.capabilities("qwen"),
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/setup/start")
def api_setup_start():
    if progress.running:
        return jsonify({"error": "Setup is already running."}), 409
    body = request.get_json(silent=True) or {}
    for key in ("comfy_url", "models_dir", "want_clone", "want_17b",
                "want_voicedesign", "want_moss", "want_moss_8b",
                "want_moss_design"):
        if key in body:
            cfg[key] = body[key]
    cfg["comfy_url"] = clean_url(cfg.get("comfy_url")) or client.url
    client.url = cfg["comfy_url"]
    save_config(cfg)
    progress.__init__()
    threading.Thread(target=bootstrap.run_setup,
                     args=(cfg, progress, comfy_proc, body.get("comfy_dir", ""),
                           body.get("mode", "auto")), daemon=True).start()
    return jsonify({"ok": True})


@app.get("/api/setup/state")
def api_setup_state():
    snap = progress.snapshot(int(request.args.get("since", 0)))
    snap["comfy_tail"] = comfy_proc.tail(12)
    return jsonify(snap)


@app.post("/api/comfy/start")
def api_comfy_start():
    if comfy_online(cfg["comfy_url"]):
        return jsonify({"ok": True, "already": True})
    py = bootstrap.comfy_python(cfg)
    if not cfg.get("comfy_dir") or not py:
        return jsonify({"error": "Run setup first."}), 400
    try:
        comfy_proc.start(py, Path(cfg["comfy_dir"]),
                         comfy_port(cfg["comfy_url"]), progress)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True})


@app.post("/api/comfy/restart")
def api_comfy_restart():
    """Stop ComfyUI and start it again, then say whether the nodes loaded.

    ComfyUI reads custom_nodes once, at startup. Installing the Qwen-TTS nodes
    into an engine that is already running leaves it running without them, and
    "Installed but ComfyUI has not loaded them" is not something anyone can act
    on from a launcher with no console. This is the act.
    """
    py = bootstrap.comfy_python(cfg)
    if not cfg.get("comfy_dir") or not py:
        return jsonify({"error": "Run setup first."}), 400
    url = cfg["comfy_url"]
    if comfy_online(url) and not comfy_proc.alive():
        return jsonify({"error": "Something else started that ComfyUI, so "
                                 "Script Builder cannot restart it. Restart it "
                                 "yourself so it loads the nodes."}), 400

    def run(task: manager.Task) -> None:
        if comfy_proc.alive():
            task.set(detail="Stopping ComfyUI…")
            comfy_proc.stop()
            for _ in range(30):
                if not comfy_online(url):
                    break
                time.sleep(1)
        task.set(detail="Starting ComfyUI — the first start is slow…")
        comfy_proc.start(py, Path(cfg["comfy_dir"]), comfy_port(url), progress)
        if not bootstrap.wait_for_comfy(url, timeout=900):
            raise RuntimeError("ComfyUI did not come back.\n"
                               + "\n".join(comfy_proc.tail(25)))
        # force=True: the schema is cached for two minutes, and two minutes of
        # "still missing" after a restart that fixed it is the wrong answer.
        client.schema(force=True)
        wanted = [e for e in bootstrap.ENGINES
                  if bootstrap.engine_enabled(cfg, e)
                  and bootstrap.node_installed(Path(cfg["comfy_dir"]), e)]
        short = [e for e in wanted if not client.engine_ready(e)]
        if not short:
            task.set(detail="Restarted — " + (", ".join(
                bootstrap.ENGINES[e]["label"] for e in wanted) or "ComfyUI")
                + " loaded.")
            return
        task.set(detail="ComfyUI is up but some nodes are still missing — "
                        "finding out why…")
        for line in comfy_proc.tail(80):
            if "IMPORT FAILED" in line or "Qwen" in line or "Moss" in line:
                task.log(line)
        reasons = []
        for eid in short:
            why = bootstrap.node_import_error(py, Path(cfg["comfy_dir"]), eid)
            reasons.append(f"{bootstrap.ENGINES[eid]['label']}: "
                           + (why or "imports fine by hand, so something else "
                                     "in custom_nodes is failing first"))
        raise RuntimeError("; ".join(reasons))

    return jsonify({"ok": True,
                    "task": manager.spawn("engine", "Restart ComfyUI",
                                          run).view()})


@app.post("/api/selftest/<engine>")
def api_selftest(engine: str):
    """Prove an engine end to end on this machine, or name the step that stops.

    Every row of the dependency report can read ok while the first take still
    fails — folders present and truncated, classes loaded from a version whose
    inputs were renamed, weights larger than the card. One line of speech,
    generated here, is the only answer that settles it.
    """
    if engine not in bootstrap.ENGINES:
        return jsonify({"error": f"There is no '{engine}' engine."}), 400
    if not bootstrap.engine_enabled(cfg, engine):
        return jsonify({"error": f"{bootstrap.ENGINES[engine]['label']} is "
                                 "turned off in Settings."}), 400
    if manager.TASKS.running("selftest"):
        return jsonify({"error": "A self-test is already running."}), 409

    # Only our own ComfyUI's console is ours to read; someone else's belongs
    # to them, and the test says so rather than reporting an empty tail as a
    # clean run.
    tail = comfy_proc.tail if comfy_proc.alive() else None

    def run(task: manager.Task) -> None:
        manager.selftest(cfg, client, engine, task, tail)

    label = bootstrap.ENGINES[engine]["label"]
    return jsonify({"ok": True,
                    "task": manager.spawn("selftest", f"Test {label}", run,
                                          {"engine": engine}).view()})


@app.get("/api/moss/8b")
def api_moss_8b():
    """What running the MOSS 8B on a small card would actually take.

    Reports; installs nothing. Two of the five prerequisites cannot be
    downloaded at all — llama.cpp is compiled from source and the TensorRT
    engines are built against the card in front of you — so anything that
    claimed first launch could fetch its way to a working 8B would be lying.
    The two HuggingFace repos are looked up rather than taken on trust.
    """
    vram = gpu_vram(client if comfy_online(cfg["comfy_url"]) else None)
    entry = next((m for m in bootstrap.ENGINES["moss"]["models"]
                  if m["repo"] == "OpenMOSS-Team/MOSS-TTS"), {})
    body = {
        "gpu": bootstrap.nvidia_gpu(),
        "vram_mb": vram,
        "through_comfyui": {
            "fits": bootstrap.fits_vram(entry.get("vram_gb") or 18, vram),
            "why": "The ComfyUI node loads bf16 weights with "
                   "AutoModel.from_pretrained — no GGUF, no ONNX, no "
                   "low-memory mode — so the 8B wants about 18 GB here.",
        },
        "through_llama_cpp": {
            "fits_claim": "OpenMOSS measure a 5.6 GB peak with "
                          "configs/llama_cpp/trt-8gb.yaml: Q4_K_M weights, "
                          "staged loading, numpy LM heads.",
            "steps": bootstrap.GGUF_STEPS,
        },
    }
    if request.args.get("check") == "1":
        body["weights"] = bootstrap.gguf_available(cfg)
    return jsonify(body)


@app.post("/api/config")
def api_config():
    body = request.get_json(silent=True) or {}
    for key in ("comfy_url", "comfy_dir", "models_dir", "auto_start_comfy",
                "torch_index", "want_clone", "want_17b", "want_voicedesign",
                "want_moss", "want_moss_8b", "want_moss_design", "engine"):
        if key in body:
            cfg[key] = body[key]
    cfg["comfy_url"] = clean_url(cfg.get("comfy_url")) or client.url
    client.url = cfg["comfy_url"]
    save_config(cfg)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# dependencies / tasks
# --------------------------------------------------------------------------- #
@app.get("/api/deps")
def api_deps():
    live = client if comfy_online(cfg["comfy_url"]) else None
    # The GPU answer is cached — it costs a PowerShell query on Windows and
    # cannot change without a reboot. Recheck asks again anyway, because
    # installing the driver is exactly what someone does between two presses.
    gpu = dict(bootstrap.nvidia_gpu(refresh=request.args.get("fresh") == "1"))
    if not gpu.get("vram_mb") and live:
        # nvidia-smi missing but ComfyUI running: it carries its own CUDA and
        # knows the card, which is exactly the gap rule 5b is about.
        gpu["vram_mb"] = live.vram_mb()
    return jsonify({"items": manager.dependencies(cfg, live),
                    "torch_index": cfg.get("torch_index", ""),
                    "gpu": gpu,
                    "torch_auto": bootstrap.torch_index({})})


@app.post("/api/deps/<dep_id>/install")
def api_dep_install(dep_id: str):
    body = request.get_json(silent=True) or {}
    if body.get("torch_index") is not None:
        cfg["torch_index"] = body["torch_index"]
        save_config(cfg)
    try:
        return jsonify({"ok": True,
                        "task": manager.install_dependency(dep_id, cfg, body).view()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.get("/api/tasks")
def api_tasks():
    task_id = request.args.get("id", "")
    since = int(request.args.get("since", 0))
    if task_id:
        task = manager.TASKS.get(task_id)
        if not task:
            return jsonify({"error": "No such task."}), 404
        return jsonify(task.view(since))
    return jsonify([t.view(t.view()["cursor"]) for t in manager.TASKS.list()[:25]])


@app.post("/api/tasks/<task_id>/cancel")
def api_task_cancel(task_id: str):
    task = manager.TASKS.get(task_id)
    if task:
        task.cancel = True
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# huggingface
# --------------------------------------------------------------------------- #
@app.get("/api/hf/settings")
def api_hf_settings():
    token = cfg.get("hf_token") or ""
    return jsonify({"endpoint": cfg.get("hf_endpoint") or manager.DEFAULT_ENDPOINT,
                    "token_set": bool(token),
                    "token_hint": ("…" + token[-4:]) if len(token) > 4 else "",
                    "repo": cfg.get("hf_repo") or bootstrap.MODEL_REPOS[0]["repo"],
                    "curated": manager.curated(cfg, gpu_vram(
                        client if comfy_online(cfg["comfy_url"]) else None)),
                    "vram_mb": gpu_vram(
                        client if comfy_online(cfg["comfy_url"]) else None),
                    "gpu_name": bootstrap.nvidia_gpu().get("name", ""),
                    "models_dir": cfg.get("models_dir", "")})


@app.post("/api/hf/settings")
def api_hf_settings_save():
    body = request.get_json(silent=True) or {}
    if "token" in body:
        cfg["hf_token"] = (body["token"] or "").strip()
    if body.get("endpoint") is not None:
        cfg["hf_endpoint"] = body["endpoint"].strip() or manager.DEFAULT_ENDPOINT
    if body.get("repo"):
        cfg["hf_repo"] = body["repo"].strip()
    if body.get("models_dir"):
        cfg["models_dir"] = body["models_dir"].strip()
    save_config(cfg)
    return jsonify({"ok": True})


@app.get("/api/hf/browse")
def api_hf_browse():
    repo = (request.args.get("repo") or cfg.get("hf_repo") or "").strip()
    try:
        data = manager.hf_browse(cfg, repo)
        cfg["hf_repo"] = repo
        save_config(cfg)
        return jsonify(data)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/hf/download")
def api_hf_download():
    body = request.get_json(silent=True) or {}
    repo = (body.get("repo") or "").strip()
    if not repo:
        return jsonify({"error": "Pick a model folder to download."}), 400
    try:
        return jsonify({"ok": True,
                        "task": manager.hf_download_repo(cfg, repo).view()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.get("/api/hf/local")
def api_hf_local():
    return jsonify({"models": manager.local_models(cfg),
                    "models_dir": cfg.get("models_dir", "")})


@app.delete("/api/hf/local")
def api_hf_delete():
    body = request.get_json(silent=True) or {}
    try:
        manager.delete_model(cfg, body.get("repo", ""))
        return jsonify({"ok": True})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


# --------------------------------------------------------------------------- #
# speech
# --------------------------------------------------------------------------- #
@app.post("/api/speak")
def api_speak():
    payload = request.get_json(silent=True) or {}
    lines = [l for l in (payload.get("lines") or [])
             if (l.get("text") or "").strip()]
    if not lines:
        return jsonify({"error": "Write at least one line of dialogue."}), 400
    if not comfy_online(cfg["comfy_url"]):
        return jsonify({"error": "ComfyUI is not running. Start it from the "
                                 "Engine panel."}), 503
    payload["lines"] = lines
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        # Every other buffer in this app is capped; this one was not. A
        # finished job keeps the whole take, and /api/jobs walks the lot once
        # a second while a run is going, so a long session paid for takes
        # nothing has looked at in hours.
        finished = sorted((j for j in jobs.values() if j["status"] != "running"),
                          key=lambda j: j["created"])
        for old_job in finished[:-25]:
            jobs.pop(old_job["id"], None)
        jobs[job_id] = {"id": job_id, "status": "running", "pct": 0,
                        "stage": "Starting", "created": time.time(),
                        "total": len(lines),
                        "title": payload.get("title") or take_title(lines)}
    threading.Thread(target=run_job, args=(job_id, payload), daemon=True).start()
    return jsonify({"job": job_id})


@app.get("/api/jobs")
def api_jobs():
    with jobs_lock:
        active = [j for j in jobs.values()
                  if j["status"] == "running" or time.time() - j["created"] < 180]
        return jsonify(sorted(active, key=lambda j: j["created"], reverse=True))


@app.post("/api/jobs/<job_id>/cancel")
def api_job_cancel(job_id: str):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["cancelled"] = True
    client.interrupt()
    return jsonify({"ok": True})


@app.post("/api/upload-reference")
def api_upload_reference():
    if "file" not in request.files:
        return jsonify({"error": "No file received."}), 400
    try:
        return jsonify({"ok": True,
                        "name": client.upload_audio(request.files["file"])})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


# --------------------------------------------------------------------------- #
# takes
# --------------------------------------------------------------------------- #
@app.get("/api/takes")
def api_takes():
    return jsonify(read_takes())


def _find_take(take_id: str) -> dict | None:
    for t in read_takes():
        if t["id"] == take_id:
            return t
    return None


@app.get("/api/take/<take_id>")
def api_take(take_id: str):
    take = _find_take(take_id)
    if not take:
        return jsonify({"error": "Take not found."}), 404
    path = TAKES_DIR / take_id / take["file"]
    if not path.exists():
        return jsonify({"error": "The audio file is missing."}), 404
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return send_file(path, mimetype=mime, conditional=True,
                     download_name=f"{take['title']}{path.suffix}")


@app.get("/api/take/<take_id>/line/<int:index>")
def api_take_line(take_id: str, index: int):
    take = _find_take(take_id)
    if not take:
        return jsonify({"error": "Take not found."}), 404
    for line in take["lines"]:
        if line["index"] == index:
            path = TAKES_DIR / take_id / line["file"]
            if not path.exists():
                return jsonify({"error": "That clip is missing."}), 404
            mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
            return send_file(path, mimetype=mime, conditional=True)
    return jsonify({"error": "No such line."}), 404


@app.delete("/api/take/<take_id>")
def api_take_delete(take_id: str):
    remove_take(take_id)
    shutil.rmtree(TAKES_DIR / take_id, ignore_errors=True)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
def sweep_orphan_takes() -> int:
    """Drop clip folders that takes.json does not list.

    takes.json is the record of what exists. A folder missing from it is one a
    run never finished — killed part way through, or left by a version that did
    not clean up after a failure — and nothing in the app can reach it again.
    """
    known = {t["id"] for t in read_takes()}
    gone = 0
    for folder in TAKES_DIR.iterdir() if TAKES_DIR.is_dir() else []:
        if folder.is_dir() and folder.name not in known:
            shutil.rmtree(folder, ignore_errors=True)
            gone += 1
    return gone


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TAKES_DIR.mkdir(parents=True, exist_ok=True)
    swept = sweep_orphan_takes()
    if swept:
        progress.log(f"Cleared {swept} unfinished take folder(s).")
    if cfg.get("setup_complete") and cfg.get("auto_start_comfy", True) \
            and cfg.get("comfy_dir") and bootstrap.comfy_python(cfg) \
            and not comfy_online(cfg["comfy_url"]):
        progress.log("Restarting ComfyUI from the last setup…")
        # An engine that cannot be started is an engine the Engine panel
        # reports as offline, never a reason the whole app fails to boot.
        try:
            comfy_proc.start(bootstrap.comfy_python(cfg),
                             Path(cfg["comfy_dir"]),
                             comfy_port(cfg["comfy_url"]), progress)
        except RuntimeError as exc:
            progress.log(f"Could not restart ComfyUI: {exc}")
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  Script Builder  →  {url}\n")
    if os.environ.get("SCRIPT_BUILDER_NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False)
    finally:
        comfy_proc.stop()


if __name__ == "__main__":
    main()
