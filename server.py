"""
server.py - Script Builder backend.

Run:  python server.py        (opens http://127.0.0.1:7799)
"""

from __future__ import annotations

import hashlib
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

import requests
from flask import Flask, jsonify, request, send_file, send_from_directory

import bootstrap
import manager
from bootstrap import (APP_DIR, DATA_DIR, ComfyProcess, Progress, clean_url,
                       comfy_online, comfy_port, detect_comfy_dirs,
                       load_config, save_config)
from comfy import ComfyClient, ComfyError, root_from_argv

# DATA_DIR comes from bootstrap so the two cannot drift apart, and so
# SCRIPT_BUILDER_DATA moves both.
TAKES_DIR = DATA_DIR / "takes"
TAKES_PATH = DATA_DIR / "takes.json"
REFS_DIR = DATA_DIR / "references"
REFS_KEEP = 100
WEB_DIR = APP_DIR / "web"
PORT = int(os.environ.get("SCRIPT_BUILDER_PORT", "7799"))

app = Flask(__name__, static_folder=None)

cfg = load_config()
progress = Progress()
# One ComfyUI per engine, each its own process on its own port with its own
# environment — the whole point of the split. Two of them that have both
# generated will each be holding models in their own VRAM, and neither can free
# the other's: ComfyUI's unload_all_models only reaches inside one process. On
# an 8 GB card that is an out-of-memory waiting to happen, so only the engine
# being used is left running, and `activate` is what enforces it.
PROCS = {eid: ComfyProcess() for eid in bootstrap.ENGINES}
CLIENTS = {eid: ComfyClient(bootstrap.engine_url(cfg, eid))
           for eid in bootstrap.ENGINES}
# Held across "stop the other, start this one, generate", so two takes started
# together cannot leave both engines resident.
engine_lock = threading.RLock()


def current_engine() -> str:
    eid = cfg.get("engine") or bootstrap.DEFAULT_ENGINE
    return eid if eid in bootstrap.ENGINES else bootstrap.DEFAULT_ENGINE


def for_engine(engine: str = "") -> ComfyClient:
    eid = engine or current_engine()
    c = CLIENTS[eid]
    c.url = bootstrap.engine_url(cfg, eid)
    return c


def proc_for(engine: str = "") -> ComfyProcess:
    return PROCS[engine or current_engine()]


def engine_online(engine: str = "") -> bool:
    return comfy_online(bootstrap.engine_url(cfg, engine or current_engine()))


def started_elsewhere(slot: dict) -> bool:
    """Is this engine one the person starts themselves?

    The external route records managed False and leaves comfy_dir empty; an
    engine that was never set up has managed True. Telling the second group to
    run setup is right, and telling the first group to is rule 18 in reverse —
    setup cannot start someone else's ComfyUI, and the address can.
    """
    return slot.get("managed") is False and not slot.get("comfy_dir")


def _note(engine: str, msg: str) -> None:
    """Say it in the engine's own console as well as the activity log.

    What the app does *to* an engine belongs next to what the engine says
    about itself, in one window and in order — a takeover split across two
    panels reads as two unrelated stories.
    """
    PROCS[engine].note(msg)
    progress.log(msg)


def _refresh_schema_when_up(engine: str) -> None:
    """Drop the cached schema the moment the engine answers again.

    The cache lasts two minutes (comfy.py), and the reason anyone starts or
    restarts an engine is that something just changed — so without this the
    fresh read hides behind the stale one for the two minutes that matter
    most. `/api/comfy/restart`'s own task already forces a read; this is for
    the paths that have no task to hang it on.
    """
    url = bootstrap.engine_url(cfg, engine)

    def wait() -> None:
        if bootstrap.wait_for_comfy(url, timeout=900):
            try:
                for_engine(engine).schema(force=True)
            except Exception:  # noqa: BLE001
                pass            # a warm cache is a nicety, never a precondition

    threading.Thread(target=wait, daemon=True).start()


def take_over_port(url: str, port: int, engine: str):
    """Close whatever ComfyUI is answering on this engine's port.

    Returns ("manager-reboot", None) when ComfyUI-Manager rebooted it in
    place, ("freed", None) when the port is now empty, or (None, advice) when
    it cannot be done — and the advice names the obstacle that was actually
    hit, because "close it yourself" against a windowless python sends someone
    hunting through Task Manager for one of several identical rows.
    """
    _note(engine, "This ComfyUI was not started here — taking it over.")
    try:
        r = requests.post(f"{url}/manager/reboot", json={}, timeout=5)
        accepted = r.status_code in (200, 201, 204)
    except requests.exceptions.RequestException:
        accepted = True          # the connection dropping *is* the reboot
    if accepted:
        deadline = time.time() + 10
        while time.time() < deadline:
            if not comfy_online(url):
                _note(engine, "ComfyUI-Manager took the reboot; waiting for "
                              "the engine to come back.")
                return "manager-reboot", None
            time.sleep(0.5)
        _note(engine, "ComfyUI-Manager did not take the reboot; stopping the "
                      "process instead.")

    def settled_free() -> bool:
        # A supervisor — ComfyUI Desktop, a launcher .bat — respawns in well
        # under a second, so a port that has gone quiet is only free once it
        # has *stayed* quiet. Without the wait, a respawn lands between the
        # check and the start and the app reports success over a port it
        # never took.
        time.sleep(2.0)
        return not comfy_online(url) and not bootstrap.port_pids(port)

    first_pids: list[int] = []
    denied = False
    for attempt in range(3):
        pids = bootstrap.port_pids(port)
        if attempt == 0:
            first_pids = pids
        if not pids:
            if not comfy_online(url) and settled_free():
                return "freed", None
            if not comfy_online(url):
                _note(engine, "It came straight back — something restarted it.")
                continue
            return None, (f"Something answers on port {port} but its process "
                          "could not be found — it may belong to another user "
                          "account. Close it yourself, then press Start "
                          "ComfyUI.")
        for pid in pids:
            cmd = bootstrap.pid_cmdline(pid)
            _note(engine, f"Port {port} is held by pid {pid}"
                  + (f": {cmd[:120]}" if cmd else " (command line unreadable)"))
            # Never close something that is not a ComfyUI. The port is only
            # this engine's by convention, and a database or another app's
            # dev server on it is a settings mistake, not an orphan.
            if cmd and not any(k in cmd.lower()
                               for k in ("python", "main.py", "comfy")):
                return None, (f"Port {port} is held by something that does not "
                              f"look like ComfyUI ({cmd[:90]}). Close it "
                              "yourself, or give this engine a different "
                              "address in Settings.")
        for pid in pids:
            said = bootstrap.kill_pid(pid)
            _note(engine, f"Stopping pid {pid} — {said or 'no reply'}")
            if "denied" in (said or "").lower() \
                    or "access" in (said or "").lower():
                denied = True
        deadline = time.time() + 8
        while comfy_online(url) and time.time() < deadline:
            time.sleep(0.5)
        if not comfy_online(url):
            if settled_free():
                return "freed", None
            _note(engine, "It came straight back — something restarted it.")
            continue
        _note(engine, "Still answering — trying again.")

    now = bootstrap.port_pids(port)
    if denied:
        return None, ("The system refused to stop it (access denied) — it was "
                      "started by another user, or as administrator. Run "
                      "Script Builder as administrator once, or close that "
                      "ComfyUI yourself, then press Start ComfyUI.")
    if now and set(now) != set(first_pids):
        return None, ("It keeps coming back under a new process id — "
                      "something is supervising it (ComfyUI Desktop, or a "
                      "launcher script). Close that application, then press "
                      "Start ComfyUI.")
    return None, ("It would not close. The Engine console shows what was "
                  "tried; close that ComfyUI yourself, then press Start "
                  "ComfyUI.")


def busy_elsewhere(engine: str) -> str:
    """Why another engine cannot be stopped for this one right now, or ""."""
    if cfg.get("run_both_engines"):
        return ""
    with jobs_lock:
        running = [j for j in jobs.values() if j.get("status") == "running"
                   and j.get("engine") and j.get("engine") != engine]
    if not running:
        return ""
    other = bootstrap.ENGINES[running[0]["engine"]]["label"]
    return (f"{other} is still reading “{running[0].get('title') or 'a take'}”"
            f" — let it finish or press Stop, then switch to "
            f"{bootstrap.ENGINES[engine]['label']}. Only one engine holds "
            "the card at a time.")


def activate(engine: str, prog=None, wait: bool = True) -> str:
    """Make this the engine that is running, and the only one.

    Returns "" when it is up, or a sentence saying why it is not. Stopping the
    others is not tidiness: a ComfyUI that has generated keeps its model in
    VRAM until something in *its* process frees it, so two live engines on an
    8 GB card means the second one fails to allocate.
    """
    def stop_others() -> None:
        if cfg.get("run_both_engines"):
            return
        for other, proc in PROCS.items():
            if other != engine and proc.alive():
                (prog or progress).log(
                    f"Stopping {bootstrap.ENGINES[other]['label']} so "
                    f"{bootstrap.ENGINES[engine]['label']} has the card "
                    "to itself.")
                proc.stop()

    with engine_lock:
        url = bootstrap.engine_url(cfg, engine)
        # Asked first, because the answer does not depend on anything below:
        # switching engines on the Create page mid-take stopped the engine the
        # take was on, and it failed with "ComfyUI stopped answering".
        busy = busy_elsewhere(engine)
        if busy:
            return busy
        if comfy_online(url):
            stop_others()
            return ""
        slot = bootstrap.engine_cfg(cfg, engine)
        py = bootstrap.comfy_python(cfg, engine)
        # Everything that can refuse is asked before the other engine is
        # stopped: switching to an engine that cannot start used to take the
        # working one down with it and leave nothing running at all.
        if started_elsewhere(slot):
            return (f"{bootstrap.ENGINES[engine]['label']}'s ComfyUI is not "
                    f"answering at {url}. Start it, or change the address in "
                    "Settings.")
        if not slot.get("comfy_dir") or not py:
            return (f"{bootstrap.ENGINES[engine]['label']} has no ComfyUI set "
                    "up yet — run setup for it from the Engine panel.")
        if not slot.get("auto_start", True):
            return (f"{bootstrap.ENGINES[engine]['label']}'s ComfyUI is not "
                    f"running at {url}, and Script Builder is set not to "
                    "start it.")
        flags, refusal = bootstrap.torch_launch(py, cfg, engine)
        if refusal:
            return refusal
        stop_others()
        try:
            PROCS[engine].start(py, Path(slot["comfy_dir"]),
                                comfy_port(url), prog or progress,
                                cfg=cfg, engine=engine, extra=flags)
        except RuntimeError as exc:
            return str(exc)
        if wait and not bootstrap.wait_for_comfy(
                url, timeout=900, alive=PROCS[engine].alive):
            if PROCS[engine].crashed():
                return bootstrap.crash_reason(PROCS[engine].tail(120), engine)
            return (f"{bootstrap.ENGINES[engine]['label']}'s ComfyUI did not "
                    "come up.\n" + "\n".join(PROCS[engine].tail(20)))
        # Warming the schema cache is a nicety, not a precondition. With
        # wait=False the engine has only just been launched and is not
        # answering yet, so this read raised ComfyError straight out of
        # activate() — and the one caller that passes wait=False is the boot
        # path, which is how auto-starting a slow engine stopped the app from
        # booting at all, a few lines under a comment promising it could not.
        try:
            if comfy_online(url):
                for_engine(engine).schema(force=True)
        except Exception:  # noqa: BLE001
            pass
        return ""

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


# Run in the engine's own interpreter. Current ComfyUI saves audio as flac,
# mp3 or opus and never wav, so without this every take became a zip of flac
# clips: no joined file, no duration, a download that has to be unpacked. The
# interpreter that wrote the flac through PyAV can always read it back, and
# this app still needs nothing beyond `wave` (see Stitching in CLAUDE.md).
FLAC_TO_WAV = r"""
import sys, wave
import av
for src in sys.argv[1:]:
    dst = src.rsplit(".", 1)[0] + ".wav"
    with av.open(src) as f:
        stream = f.streams.audio[0]
        rate = stream.codec_context.sample_rate
        to16 = None
        pcm = []
        channels = 0
        for frame in f.decode(stream):
            if to16 is None:
                to16 = av.AudioResampler(format="s16", layout=frame.layout,
                                         rate=rate)
            for out in to16.resample(frame):
                channels = channels or len(out.layout.channels)
                pcm.append(out.to_ndarray().astype("<i2").tobytes())
        if to16 is not None:
            for out in to16.resample(None):
                pcm.append(out.to_ndarray().astype("<i2").tobytes())
    with wave.open(dst, "wb") as w:
        w.setnchannels(channels or 1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(pcm))
    print(dst)
"""


def to_wav(clips: list[Path], engine: str) -> list[Path]:
    """The clips as wav, converted in the engine's interpreter where needed.

    Gives back the clips unchanged when that cannot be done — a ComfyUI
    someone else started, whose interpreter this app does not know — and the
    caller then zips them as before.
    """
    todo = [c for c in clips if c.suffix.lower() != ".wav"]
    if not todo:
        return clips
    python = bootstrap.comfy_python(cfg, engine)
    if not python or not Path(python).exists():
        return clips
    try:
        out = bootstrap._run([python, "-c", FLAC_TO_WAV]
                             + [str(c) for c in todo], timeout=300)
    except Exception:  # noqa: BLE001
        return clips
    wavs = [c.with_suffix(".wav") for c in clips]
    if out.returncode != 0 or not all(w.exists() for w in wavs):
        for w, c in zip(wavs, clips):
            if w != c:
                w.unlink(missing_ok=True)
        return clips
    for c in todo:
        c.unlink(missing_ok=True)
    return wavs


# --------------------------------------------------------------------------- #
# generation job
# --------------------------------------------------------------------------- #
def wait_for_prompt(prompt_id: str, job_id: str, engine: str,
                    timeout: int = 900) -> list[dict]:
    """Wait on one line, on the engine that queued it.

    The engine has to be passed in: there are two clients now, and reading the
    selected one here would poll the wrong ComfyUI the moment someone switched
    engines mid-take.
    """
    started = time.time()
    # A short line on a warm model is back in well under a second, and a
    # whole second of polling latency per line was most of the gap between
    # lines. The wait backs off to a second once a line is clearly working.
    wait = 0.2
    while True:
        time.sleep(wait)
        wait = min(wait * 1.5, 1.0)
        with jobs_lock:
            if jobs[job_id].get("cancelled"):
                for_engine(engine).interrupt(prompt_id)
                raise ComfyError("Cancelled")
        outs, err = for_engine(engine).result(prompt_id)
        if err:
            raise ComfyError(err)
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
    base = bootstrap.engine_models_dir(cfg, "moss")
    if not base:
        return {}
    out = {}
    for m in bootstrap.ENGINES["moss"]["models"]:
        folder = bootstrap.model_dir(base, m["repo"], "moss")
        if folder.is_dir() and bootstrap.model_installed(base, m["repo"], "moss"):
            out[m["repo"]] = str(folder)
    return out


def generation_order(voices: list[dict], opts: dict) -> list[int]:
    """The order to speak the lines in, so each checkpoint loads once a take.

    Each engine holds one model at a time, so a preset speaker answering a
    cloned one swapped weights on every line — seconds of disk read apiece,
    and on an 8 GB card most of the take. Lines are grouped by the weights
    they need, in the order each is first needed, and kept in script order
    within a group; the take is still joined in script order.
    """
    groups: dict[tuple, list[int]] = {}
    for i, voice in enumerate(voices):
        groups.setdefault(ComfyClient.line_weights(voice, opts),
                          []).append(i)
    return [i for group in groups.values() for i in group]


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
        # The slider goes down to 0, and `or 0.5` read that as "not given".
        pause = payload.get("pause")
        pause = 0.5 if pause in (None, "") else max(float(pause), 0.0)
        folder.mkdir(parents=True, exist_ok=True)

        keys, voices = [], []
        for line in lines:
            key = str(line.get("speaker", 1))
            # JSON object keys are strings, but a take loaded back can carry
            # integer ones. A key that is neither is a speaker we do not have,
            # not a reason to fail the whole job.
            voice = speakers.get(key) or {}
            if not voice and key.isdigit():
                voice = speakers.get(int(key)) or {}
            keys.append(key)
            voices.append(voice)
        for voice in voices:
            if voice.get("kind") == "clone" and voice.get("ref_audio"):
                ensure_reference(engine, voice["ref_audio"])
        order = generation_order(voices, opts)
        # Asked for once per take, on its last line. Sent with every line it
        # made the node drop its weights after each one and read them back
        # from disk for the next: a take paid a model load per line for a
        # switch that says "after each run".
        unload = opts["unload"]
        made: dict[int, Path] = {}

        for done, i in enumerate(order):
            line, key, voice = lines[i], keys[i], voices[i]
            set_state(stage=f"Line {i + 1} of {len(lines)} · "
                            f"{voice.get('name') or 'Speaker ' + key}",
                      pct=round(done / max(len(lines), 1) * 100, 1),
                      line_index=i)

            opts["unload"] = unload and done == len(order) - 1
            built = for_engine(engine).build_line(line, voice, opts)
            prompt_id = for_engine(engine).queue(built["prompt"])
            set_state(engine=engine, prompt_id=prompt_id)
            outs = wait_for_prompt(prompt_id, job_id, engine)
            item = outs[0]
            ext = Path(item["filename"]).suffix or ".wav"
            dest = folder / f"line_{i:03d}{ext}"
            with for_engine(engine).view(item) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(1024 * 256):
                        fh.write(chunk)
            made[i] = dest

        if made:
            set_state(stage="Joining the lines", pct=97)
        order = sorted(made)
        made = dict(zip(order, to_wav([made[i] for i in order], engine)))
        clips = [made[i] for i in order]
        meta_lines = [{"index": i,
                       "speaker": int(keys[i]) if keys[i].isdigit() else 1,
                       "text": lines[i].get("text", ""),
                       "file": made[i].name} for i in order]

        if not clips:
            raise ComfyError("There is nothing in the script to say.")

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
    engine = request.args.get("engine") or cfg.get("engine") \
        or bootstrap.DEFAULT_ENGINE
    if engine not in bootstrap.ENGINES:
        engine = bootstrap.DEFAULT_ENGINE
    # Everything below is about *this* engine's install: its ComfyUI, its
    # models folder, its port. The other engine's state is its own business
    # and is reported separately in engine_nodes.
    slot = bootstrap.engine_cfg(cfg, engine)
    online = comfy_online(slot["comfy_url"])
    models_dir = bootstrap.engine_models_dir(cfg, engine)
    # Whether the voices could be checked at all, which is not the same as
    # finding none missing. With no models folder set there is nowhere to look,
    # and an empty "missing" list used to read as "all present" — so a machine
    # with ComfyUI up, the nodes loaded and not one voice on disk reported the
    # engine ready and let someone press Read.
    models_known = bool(models_dir and models_dir.is_dir())
    payload = {
        "comfy_online": online,
        "engine": engine,
        "engines": [{"id": e["id"], "label": e["label"], "blurb": e["blurb"],
                     "role": e.get("role", "secondary"),
                     "enabled": bootstrap.engine_enabled(cfg, e["id"])}
                    for e in bootstrap.ENGINES.values()],
        "primary_engine": bootstrap.start_engine(cfg),
        "setup_complete": bool(cfg.get("setup_complete")),
        # So the Create page can say "Setting up…" rather than offer a setup
        # that is already running.
        "setup_running": bool(progress.running),
        "models_known": models_known,
        "detected": detect_comfy_dirs(),
        "config": dict({k: cfg.get(k) for k in
                        ("torch_index", "want_clone", "want_17b",
                         "want_voicedesign", "want_moss", "want_moss_8b",
                         "want_moss_design", "engine", "run_both_engines")},
                       # The selected engine's own install, flattened under the
                       # names the page has always used, so one panel edits one
                       # engine rather than a shape it has to understand.
                       comfy_url=slot["comfy_url"],
                       comfy_dir=slot["comfy_dir"],
                       models_dir=slot["models_dir"],
                       managed=slot["managed"],
                       auto_start_comfy=slot["auto_start"]),
        # `stopped` is why an engine started here has exited on its own, in
        # a sentence — the page shows it instead of "its last words are
        # below" over a stack trace.
        "installs": {e: dict(bootstrap.engine_cfg(cfg, e),
                             running=PROCS[e].alive(),
                             stopped=(bootstrap.crash_reason(
                                 PROCS[e].tail(120), e)
                                 if PROCS[e].crashed() else ""),
                             online=comfy_online(
                                 bootstrap.engine_cfg(cfg, e)["comfy_url"]))
                     for e in bootstrap.ENGINES},
        "nodes_ready": False, "ready": False,
    }
    # Readiness is per engine: with MOSS selected, a missing Qwen folder is
    # not what stands between this script and a take, and reporting it as one
    # sends people to download a model they are not about to use.
    missing = [m["repo"] for m in bootstrap.engine_missing(cfg, engine)] \
        if models_known else []
    payload["missing_models"] = missing
    root = ""
    if online:
        try:
            payload["nodes_ready"] = for_engine(engine).engine_ready(engine)
            payload["capabilities"] = for_engine(engine).capabilities(engine)
            payload["engine_nodes"] = {
                e: (for_engine(e).engine_ready(e) if engine_online(e) else False)
                for e in bootstrap.ENGINES}
        except Exception as exc:  # noqa: BLE001
            payload["schema_error"] = str(exc)
        # The three silent "nothing works" states, named, so the Engine
        # console can say which one it is instead of showing a healthy-looking
        # panel over a dead app. One /system_stats read serves both of the
        # identity answers and the `foreign` sentence below.
        argv, root = engine_identity(engine)
        payload["engine_argv"] = argv
        payload["engine_mismatch"] = bool(
            root and slot.get("comfy_dir")
            and not manager.same_install(slot["comfy_dir"], root))
        payload["engine_managed"] = PROCS[engine].alive()
        payload["stale_reason"] = stale_engine(
            engine, bool(models_known and not missing))
        payload["stale_models"] = bool(payload["stale_reason"])
    # Only when the nodes are not loaded: that is the one symptom another
    # ComfyUI holding the port produces. Restarting ours would not fix it, so
    # the page must not offer that as the way out.
    payload["foreign"] = (foreign_engine(engine, root)
                          if online and not payload["nodes_ready"] else "")
    payload["ready"] = bool(online and payload["nodes_ready"]
                            and models_known and not missing)
    return jsonify(payload)


@app.get("/api/voices")
def api_voices():
    want = request.args.get("engine") or current_engine()
    if want not in bootstrap.ENGINES:
        want = bootstrap.DEFAULT_ENGINE
    if not engine_online(want):
        return jsonify({"error": f"{bootstrap.ENGINES[want]['label']}'s "
                                 "ComfyUI is not running."}), 503
    client = for_engine(want)
    engine = want
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
    for key in ("want_clone", "want_17b", "want_voicedesign", "want_moss",
                "want_moss_8b", "want_moss_design"):
        if key in body:
            cfg[key] = body[key]
    # An address or folder in the setup body belongs to one engine's install.
    # "qwen_comfy_url" names it outright; a bare "comfy_url" is the older shape
    # and means the default engine, which is the only one that existed then.
    for eid in bootstrap.ENGINES:
        slot = bootstrap.engine_cfg(cfg, eid)
        for key in ("comfy_url", "models_dir"):
            value = body.get(f"{eid}_{key}") or (
                body.get(key) if eid == bootstrap.DEFAULT_ENGINE else "")
            if value:
                slot[key] = value
        slot["comfy_url"] = clean_url(slot["comfy_url"]) \
            or f"http://127.0.0.1:{bootstrap.ENGINES[eid]['port']}"
    save_config(cfg)
    progress.__init__()
    threading.Thread(target=bootstrap.run_setup,
                     args=(cfg, progress, PROCS, body.get("comfy_dir", ""),
                           body.get("mode", "auto")), daemon=True).start()
    return jsonify({"ok": True})


@app.get("/api/setup/state")
def api_setup_state():
    snap = progress.snapshot(int(request.args.get("since", 0)))
    # With `engine`, only that engine's console. Start the engine streams this
    # while it waits, and two engines' output interleaved is not a console
    # anyone can read — it is the first start's model load that has to be
    # legible, and that belongs to one of them.
    want = request.args.get("engine")
    if want in bootstrap.ENGINES:
        snap["comfy_tail"] = PROCS[want].tail(40)
    else:
        snap["comfy_tail"] = [l for e in bootstrap.ENGINES
                              for l in PROCS[e].tail(6)]
    return jsonify(snap)


def engine_identity(engine: str) -> tuple[str, str]:
    """(argv[0], the folder it runs from) for whatever is answering, or ("","").

    One read of /system_stats, shared by the three callers that used to ask
    separately — the page polls status every few seconds and each of those was
    a request of its own.
    """
    stats = bootstrap.comfy_stats(bootstrap.engine_url(cfg, engine)) or {}
    argv = stats.get("argv") or []
    return ((argv[0] if argv else ""), root_from_argv(argv))


def stale_engine(engine: str, weights_ready: bool) -> str:
    """Why the engine answering cannot reach what is already on disk, or "".

    The usual meaning of a flag by that name is a model scan: ComfyUI lists
    its model folders once, at startup, so weights that landed afterwards are invisible
    until a restart. Neither of this app's node packs works that way — both
    resolve a checkpoint folder per call, freshly, so a voice downloaded
    behind a running engine is found without restarting anything. What does go
    stale here is the half of the same disease that rule 17 already names:
    ComfyUI reads `custom_nodes` once, at startup. An engine started before
    its node pack landed is a complete install with nothing in it — every
    folder present, every download finished, no classes, and a Create page
    that fails on "the nodes are not loaded" however many times they are
    installed.

    MOSS carries the model-list half as well, because its loader publishes the
    only enum in either pack that names checkpoints. Qwen declares no marker
    (see ENGINES) and is judged on its nodes alone.
    """
    if not weights_ready:
        return ""
    slot = bootstrap.engine_cfg(cfg, engine)
    label = bootstrap.ENGINES[engine]["label"]
    client = for_engine(engine)
    try:
        loaded = client.engine_ready(engine)
        listed = client.model_list(engine)
    except Exception:  # noqa: BLE001
        return ""            # an engine that will not answer is not "stale"
    if slot.get("comfy_dir") and not loaded and bootstrap.node_installed(
            Path(slot["comfy_dir"]), engine):
        return (f"the {label} nodes are on disk but this ComfyUI started "
                "before they were installed")
    marker = bootstrap.ENGINES[engine].get("model_marker") or ""
    if loaded and marker and not any(marker in x.lower() for x in listed):
        return (f"this ComfyUI lists no {label} checkpoint it could load")
    return ""


def foreign_engine(engine: str, root: str | None = None) -> str:
    """Why the ComfyUI answering this engine's address is not this engine's.

    Empty when it is ours, when there is no managed install to compare
    against, or when the engine will not say where it runs from. 8188 is the
    port every ComfyUI picks by default, so on this app's own port the one
    answering is quite often somebody else's — and that looks exactly like
    nodes that will not load, however many times they are installed.
    """
    slot = bootstrap.engine_cfg(cfg, engine)
    if not slot.get("comfy_dir"):
        return ""
    if root is None:
        try:
            root = for_engine(engine).engine_root()
        except Exception:  # noqa: BLE001
            return ""
    if not root or manager.same_install(slot["comfy_dir"], root):
        return ""
    return (f"{slot['comfy_url']} is answered by the ComfyUI in {root}, not "
            f"{bootstrap.ENGINES[engine]['label']}'s own in "
            f"{slot['comfy_dir']}.")


@app.post("/api/comfy/start")
def api_comfy_start():
    engine = request.args.get("engine") or current_engine()
    if engine not in bootstrap.ENGINES:
        return jsonify({"error": f"There is no '{engine}' engine."}), 400
    if engine_online(engine):
        # Nothing is started when the address already answers. Saying
        # "starting" here is how this button came to look broken: pressed,
        # claims to work, changes nothing, and never mentions that something
        # else holds the port.
        return jsonify({"ok": True, "already": True,
                        "foreign": foreign_engine(engine)})
    why = activate(engine, wait=False)
    if why:
        # In the engine's console too, where the page says "Offline" — the
        # toast is gone in seconds and the reason is the one thing to keep.
        _note(engine, f"Could not start it: {why}")
        return jsonify({"error": why}), 400
    _refresh_schema_when_up(engine)
    return jsonify({"ok": True})


@app.post("/api/comfy/restart")
def api_comfy_restart():
    """Stop ComfyUI and start it again, then say whether the nodes loaded.

    ComfyUI reads custom_nodes once, at startup. Installing the Qwen-TTS nodes
    into an engine that is already running leaves it running without them, and
    "Installed but ComfyUI has not loaded them" is not something anyone can act
    on from a launcher with no console. This is the act.

    An engine this app did not start — an orphan from an earlier launch, a
    ComfyUI Desktop, one started by hand on the same port — used to be a dead
    end here: Start said "already running", Restart said "not started by this
    app", and the only advice left was to go and find a windowless python in
    Task Manager. It is taken over instead, through ComfyUI-Manager's own
    reboot where that is installed and by closing the process where it is not,
    and every refusal is a 409 that names the obstacle it actually hit.

    `how` says which of the four routes ran: managed, started, takeover or
    manager-reboot.
    """
    engine = request.args.get("engine") or current_engine()
    if engine not in bootstrap.ENGINES:
        return jsonify({"error": f"There is no '{engine}' engine."}), 400
    slot = bootstrap.engine_cfg(cfg, engine)
    comfy_proc = PROCS[engine]
    py = bootstrap.comfy_python(cfg, engine)
    if started_elsewhere(slot):
        # External mode is a deliberate choice: there is no install here to
        # put back, so closing that ComfyUI would leave nothing at all.
        return jsonify({"error": "Script Builder did not start that ComfyUI, "
                                 "so it cannot restart it. Start it again at "
                                 f"{slot['comfy_url']} so it loads the "
                                 "nodes."}), 400
    if not slot.get("comfy_dir") or not py:
        # Same reason, one step earlier: taking a port from someone and having
        # no engine to start in its place is not a restart, it is a hole.
        return jsonify({"error": "Run setup first."}), 400
    # Asked before anything is stopped or taken over: an engine whose PyTorch
    # cannot start is not one to swap for the one answering now (rule 33a).
    flags, refusal = bootstrap.torch_launch(py, cfg, engine)
    if refusal:
        return jsonify({"error": refusal}), 409
    url = slot["comfy_url"]
    port = comfy_port(url)

    how = "managed"
    if not comfy_proc.alive():
        if not comfy_online(url):
            how = "started"
        else:
            kind, advice = take_over_port(url, port, engine)
            if advice:
                return jsonify({"error": advice}), 409
            if kind == "manager-reboot":
                # It is restarting itself, in place, with our models folder
                # and our custom_nodes — there is nothing left to start.
                _refresh_schema_when_up(engine)
                return jsonify({"ok": True, "how": "manager-reboot"})
            how = "takeover"

    def run(task: manager.Task) -> None:
        if comfy_proc.alive():
            task.set(detail="Stopping ComfyUI…")
            comfy_proc.stop()
            for _ in range(30):
                if not comfy_online(url):
                    break
                time.sleep(1)
        task.set(detail="Starting ComfyUI — the first start is slow…")
        comfy_proc.start(py, Path(slot["comfy_dir"]), comfy_port(url),
                         progress, cfg=cfg, engine=engine, extra=flags)
        if not bootstrap.wait_for_comfy(url, timeout=900,
                                        alive=comfy_proc.alive):
            if comfy_proc.crashed():
                raise RuntimeError(bootstrap.crash_reason(
                    comfy_proc.tail(120), engine))
            raise RuntimeError("ComfyUI did not come back.\n"
                               + "\n".join(comfy_proc.tail(25)))
        # force=True: the schema is cached for two minutes, and two minutes of
        # "still missing" after a restart that fixed it is the wrong answer.
        client = for_engine(engine)
        client.schema(force=True)
        wanted = [engine] if bootstrap.node_installed(
            Path(slot["comfy_dir"]), engine) else []
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
            why = bootstrap.node_import_error(py, Path(slot["comfy_dir"]), eid)
            reasons.append(f"{bootstrap.ENGINES[eid]['label']}: "
                           + (why or "imports fine by hand, so something else "
                                     "in custom_nodes is failing first"))
        raise RuntimeError("; ".join(reasons))

    return jsonify({"ok": True, "how": how,
                    "task": manager.spawn(
                        "engine",
                        f"Restart {bootstrap.ENGINES[engine]['label']}",
                                          run).view()})


@app.get("/api/comfy/log")
def api_comfy_log():
    """One engine's own console — the visible cue that it is starting, up, or
    saying exactly which import failed.

    `/api/setup/state` carries a tail too, but that one belongs to a setup run
    and stops when the run does. This is the engine panel's live feed, and it
    is also where `note()` puts what the app did to the engine.
    """
    engine = request.args.get("engine") or current_engine()
    if engine not in bootstrap.ENGINES:
        return jsonify({"error": f"There is no '{engine}' engine."}), 400
    # Clamped, and never int() on raw input: a value typed into a URL is not a
    # reason for a 500 (rule 9, in a smaller place).
    try:
        n = int(request.args.get("n", 80))
    except (TypeError, ValueError):
        n = 80
    n = min(max(n, 1), 400)
    return jsonify({"engine": engine,
                    "lines": PROCS[engine].tail(n),
                    "running": PROCS[engine].alive(),
                    "stopped": (bootstrap.crash_reason(
                        PROCS[engine].tail(120), engine)
                        if PROCS[engine].crashed() else ""),
                    "online": comfy_online(bootstrap.engine_url(cfg, engine))})


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
    # One engine on the card at a time — the same rule the take path follows.
    why = activate(engine)
    if why:
        return jsonify({"error": why}), 400
    tail = PROCS[engine].tail if PROCS[engine].alive() else None

    def run(task: manager.Task) -> None:
        manager.selftest(cfg, for_engine(engine), engine, task, tail)

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
    vram = gpu_vram(for_engine() if engine_online() else None)
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
    # An engine can be named, so Settings can edit either install; without one
    # the edit lands on whichever engine is selected.
    target = body.get("for_engine") or body.get("engine") or current_engine()
    if target not in bootstrap.ENGINES:
        target = bootstrap.DEFAULT_ENGINE
    slot = bootstrap.engine_cfg(cfg, target)
    for key, into in (("comfy_url", "comfy_url"), ("comfy_dir", "comfy_dir"),
                      ("models_dir", "models_dir"),
                      ("auto_start_comfy", "auto_start")):
        if key in body:
            slot[into] = body[key]
    slot["comfy_url"] = clean_url(slot.get("comfy_url")) \
        or f"http://127.0.0.1:{bootstrap.ENGINES[target]['port']}"
    for key in ("torch_index", "want_clone", "want_17b", "want_voicedesign",
                "want_moss", "want_moss_8b", "want_moss_design", "engine",
                "run_both_engines"):
        if key in body:
            cfg[key] = body[key]
    save_config(cfg)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# dependencies / tasks
# --------------------------------------------------------------------------- #
@app.get("/api/deps")
def api_deps():
    # Every engine that is answering, so each row is judged against its own
    # ComfyUI rather than the selected one's.
    live = {e: for_engine(e) for e in bootstrap.ENGINES if engine_online(e)}
    any_live = live.get(current_engine()) or next(iter(live.values()), None)
    # The GPU answer is cached — it costs a PowerShell query on Windows and
    # cannot change without a reboot. Recheck asks again anyway, because
    # installing the driver is exactly what someone does between two presses.
    gpu = dict(bootstrap.nvidia_gpu(refresh=request.args.get("fresh") == "1"))
    if not gpu.get("vram_mb") and any_live:
        # nvidia-smi missing but ComfyUI running: it carries its own CUDA and
        # knows the card, which is exactly the gap rule 5b is about.
        gpu["vram_mb"] = any_live.vram_mb()
    return jsonify({"items": manager.dependencies(cfg, live, current_engine()),
                    "torch_index": cfg.get("torch_index", ""),
                    "gpu": gpu,
                    "torch_auto": bootstrap.torch_index({})})


def _stop_for_install(engine: str) -> bool:
    """Stop this engine's ComfyUI before pip changes what it has loaded."""
    proc = PROCS.get(engine)
    if not proc or not proc.alive():
        return False
    _note(engine, "Stopping this engine while its packages change — a running "
                  "ComfyUI holds them open. Start it again once the install "
                  "is done.")
    proc.stop()
    return True


@app.post("/api/deps/<dep_id>/install")
def api_dep_install(dep_id: str):
    body = request.get_json(silent=True) or {}
    if body.get("torch_index") is not None:
        cfg["torch_index"] = body["torch_index"]
        save_config(cfg)
    try:
        return jsonify({"ok": True,
                        "task": manager.install_dependency(
                            dep_id, cfg, body,
                            stop_engine=_stop_for_install).view()})
    except manager.InstallBusy as exc:
        return jsonify({"error": str(exc)}), 409
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
                        for_engine() if engine_online() else None)),
                    "vram_mb": gpu_vram(
                        for_engine() if engine_online() else None),
                    "gpu_name": bootstrap.nvidia_gpu().get("name", ""),
                    "models_dir": str(bootstrap.engine_models_dir(
                        cfg, current_engine()) or "")})


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
        bootstrap.engine_cfg(cfg, current_engine())["models_dir"] = \
            body["models_dir"].strip()
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
                    "models_dir": str(bootstrap.engine_models_dir(
                        cfg, current_engine()) or "")})


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
    want = payload.get("engine") or current_engine()
    if want not in bootstrap.ENGINES:
        want = bootstrap.DEFAULT_ENGINE
    payload["lines"] = lines
    payload["engine"] = want
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
                        "engine": want,
                        "total": len(lines),
                        "title": payload.get("title") or take_title(lines)}
    # Bring this engine up and put the other one down before a single line is
    # queued. Two ComfyUIs that have both generated each hold their models in
    # their own VRAM and neither can free the other's, so on 8 GB the second
    # take is the one that fails to allocate. The job is registered first, so
    # a switch pressed while this one starts sees it and leaves it alone.
    why = activate(want)
    if why:
        with jobs_lock:
            jobs.pop(job_id, None)
        return jsonify({"error": why}), 503
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
    # Only a job that is running, and only on the engine it is running on.
    # This used to interrupt whatever the selected engine was doing whether or
    # not the job was alive — and the player's Stop sends the last job's id,
    # so stopping playback killed a self-test or a preview mid-line.
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or job.get("status") != "running":
            return jsonify({"ok": True, "running": False})
        job["cancelled"] = True
        engine, prompt_id = job.get("engine"), job.get("prompt_id")
    if engine and prompt_id:
        for_engine(engine).interrupt(prompt_id)
    return jsonify({"ok": True, "running": True})


@app.post("/api/upload-reference")
def api_upload_reference():
    """Keep the clip here, under a name taken from its contents.

    Each engine is its own ComfyUI with its own input folder (rule 30), and
    the clip used to go only to the one showing: switch engines and every
    cloned line failed "Invalid audio file". And uploads overwrote by file
    name, so two speakers' "recording.wav" became one voice. The copy here is
    what `ensure_reference` hands to whichever engine speaks the line.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file received."}), 400
    f = request.files["file"]
    data = f.read()
    if not data:
        return jsonify({"error": "That file is empty."}), 400
    ext = Path(f.filename or "").suffix.lower()
    if not ext or len(ext) > 6 or not ext[1:].isalnum():
        ext = ".wav"
    name = f"sb-ref-{hashlib.sha1(data).hexdigest()[:16]}{ext}"
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    (REFS_DIR / name).write_bytes(data)
    # Rule 14. Newest kept; a speaker still pointing at a pruned clip keeps
    # working on any engine that already has it.
    kept = sorted(REFS_DIR.glob("sb-ref-*"), key=lambda p: p.stat().st_mtime)
    for old in kept[:-REFS_KEEP]:
        old.unlink(missing_ok=True)
    try:
        if engine_online():
            ensure_reference(current_engine(), name)
    except Exception:  # noqa: BLE001
        pass            # the take sends it again; the copy here is what counts
    return jsonify({"ok": True, "name": name})


def ensure_reference(engine: str, name: str) -> None:
    """Put a kept reference clip in that engine's ComfyUI before a take.

    Sent every take: a clip is a few megabytes, and a copy remembered as sent
    is wrong the day that ComfyUI's input folder is not the one it was. A name
    this app did not keep (one from before the copy existed) is left as it
    is — it may already be there.
    """
    local = REFS_DIR / Path(name).name
    if not local.is_file():
        return
    mime = mimetypes.guess_type(local.name)[0] or "audio/wav"
    for_engine(engine).upload_bytes(local.name, local.read_bytes(), mime)


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


def ensure_engine_at_boot() -> None:
    """A launch ends with a working engine, without a button pressed.

    Four outcomes, and each one is said out loud in the engine console:

      offline            start the engine this library opens on;
      online and healthy adopt it, and say so — a ComfyUI somebody left
                         running is not a problem to be solved;
      online and useless replace it, through the same looks-like-ComfyUI
                         guard the Restart button uses;
      external mode      never touched. That engine is the person's own; the
                         app says what is wrong with it and leaves it alone.

    Only the engine a launch opens on (rule 18g), and only ever one at a time
    (rule 31) — bringing the other up costs the card for nothing.
    """
    if not cfg.get("setup_complete"):
        return
    engine = current_engine()
    slot = bootstrap.engine_cfg(cfg, engine)
    url = slot["comfy_url"]
    port = comfy_port(url)
    if not slot.get("auto_start", True):
        return

    if not comfy_online(url):
        _note(engine, "Starting the engine this library was last set to…")
        why = activate(engine, wait=False)
        if why:
            # An engine that cannot start is an engine the Engine panel
            # reports as offline, never a reason the app fails to boot.
            _note(engine, f"Could not start it: {why}")
        else:
            _refresh_schema_when_up(engine)
        return

    # Something already answers. Adopt it, or replace it — but decide on
    # evidence, not on who started it.
    if started_elsewhere(slot):
        _note(engine, f"Adopting the ComfyUI already running at {url} — it is "
                      "yours, not this app's.")
        return
    try:
        for_engine(engine).schema(force=True)
    except Exception as exc:  # noqa: BLE001
        _note(engine, "The engine already running would not describe itself "
                      f"({exc}) — leaving it alone.")
        return

    models_dir = bootstrap.engine_models_dir(cfg, engine)
    weights_ready = bool(models_dir and models_dir.is_dir()
                         and not bootstrap.engine_missing(cfg, engine))
    reasons = []
    why = stale_engine(engine, weights_ready)
    if why:
        reasons.append(why)
    _, root = engine_identity(engine)
    if root and slot.get("comfy_dir") \
            and not manager.same_install(slot["comfy_dir"], root):
        reasons.append(f"a different install is answering the address ({root})")

    if not reasons:
        _note(engine, f"Adopting the ComfyUI already running at {url}.")
        return
    py = bootstrap.comfy_python(cfg, engine)
    if not slot.get("managed", True) or not slot.get("comfy_dir") or not py:
        _note(engine, "The engine already running has problems ("
                      + "; ".join(reasons) + ") but this app has no ComfyUI of "
                      "its own to put in its place — restart it yourself, or "
                      "press Restart ComfyUI.")
        return
    _, refusal = bootstrap.torch_launch(py, cfg, engine)
    if refusal:
        _note(engine, "The engine already running has problems ("
                      + "; ".join(reasons) + "), but this app's own could not "
                      "start in its place: " + refusal)
        return
    _note(engine, "The engine already running is no use as it stands — "
                  + "; ".join(reasons) + ". Replacing it.")
    kind, advice = take_over_port(url, port, engine)
    if advice:
        _note(engine, advice)
        return
    if kind == "manager-reboot":
        _refresh_schema_when_up(engine)
        return
    _note(engine, "Starting a managed engine in its place…")
    problem = activate(engine, wait=False)
    if problem:
        _note(engine, f"Could not start it: {problem}")
    else:
        _refresh_schema_when_up(engine)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TAKES_DIR.mkdir(parents=True, exist_ok=True)
    swept = sweep_orphan_takes()
    if swept:
        progress.log(f"Cleared {swept} unfinished take folder(s).")
    # Before the engine starts, so its first line finds the folders where it
    # looks rather than downloading them a second time.
    bootstrap.migrate_qwen_layout(
        bootstrap.engine_models_dir(cfg, "qwen"),
        lambda msg: _note("qwen", msg))
    # Every launch opens on the primary engine, whichever one the last session
    # ended on. MOSS is a deliberate switch made on the Create page, and it
    # lasts that session: an app that quietly came back up holding the
    # secondary engine's models is one that chose for you, and on 8 GB that
    # choice costs the card.
    primary = bootstrap.start_engine(cfg)
    if cfg.get("engine") != primary:
        was = cfg.get("engine")
        cfg["engine"] = primary
        save_config(cfg)
        if was:
            progress.log(f"Opening on {bootstrap.ENGINES[primary]['label']}, "
                         f"the primary engine — the last session ended on "
                         f"{bootstrap.ENGINES.get(was, {}).get('label', was)}.")
    # On its own thread: taking a port off an orphan can take half a minute,
    # and the page has to be openable while it happens — the console it
    # narrates into is on that page.
    threading.Thread(target=ensure_engine_at_boot, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  Script Builder  →  {url}\n")
    if os.environ.get("SCRIPT_BUILDER_NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False)
    finally:
        for proc in PROCS.values():
            proc.stop()


if __name__ == "__main__":
    main()
