"""Unit tests for the parts that are easy to get quietly wrong.

Standard library only — `python -m unittest discover tests` needs nothing that
requirements.txt does not already install.

Every test here stands for a fault that actually shipped at some point, so the
names say what would break rather than what the function is called.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import struct
import sys
import tempfile
import subprocess
import threading
import unittest
from unittest import mock
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# server.py reads DATA_DIR at import, so point it somewhere disposable before
# importing it — otherwise running the tests would adopt the real library.
_SANDBOX = tempfile.mkdtemp(prefix="sb-tests-")
os.environ["SCRIPT_BUILDER_DATA"] = _SANDBOX

import bootstrap  # noqa: E402
import comfy  # noqa: E402
import manager  # noqa: E402
import server  # noqa: E402


def tearDownModule():
    shutil.rmtree(_SANDBOX, ignore_errors=True)


def make_clip(path: Path, channels=1, width=2, rate=24000, frames=2000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x01\x02" * (frames * channels * width // 2))
    return path


class Stitching(unittest.TestCase):
    """CLAUDE.md: the gap is whole frames, and a refusal leaves nothing."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_gap_is_a_whole_number_of_frames(self):
        # 0.75s at 22050 Hz stereo is the case that used to land on half a
        # frame and swap the channels for the rest of the file.
        clips = [make_clip(self.dir / f"{i}.wav", channels=2, rate=22050)
                 for i in range(3)]
        out = self.dir / "joined.wav"
        self.assertTrue(server.stitch_wavs(clips, out, 0.75))
        with wave.open(str(out)) as w:
            frames, rate = w.getnframes(), w.getframerate()
            data = w.readframes(frames)
        self.assertEqual(len(data) % (2 * 2), 0, "data is not whole frames")
        self.assertEqual(frames, 3 * 2000 + 2 * int(rate * 0.75))

    def test_the_gap_is_actually_silent_and_in_the_right_place(self):
        clips = [make_clip(self.dir / f"m{i}.wav", frames=1000) for i in range(2)]
        out = self.dir / "m.wav"
        self.assertTrue(server.stitch_wavs(clips, out, 0.5))
        with wave.open(str(out)) as w:
            rate = w.getframerate()
            samples = struct.unpack(f"<{w.getnframes()}h",
                                    w.readframes(w.getnframes()))
        gap = samples[1000:1000 + int(rate * 0.5)]
        self.assertEqual(max(map(abs, gap)), 0, "gap is not silence")
        self.assertGreater(max(map(abs, samples[:1000])), 0, "audio is silent")

    def test_mismatched_formats_are_refused_and_leave_no_file(self):
        for name, kw in (("rate", {"rate": 16000}), ("channels", {"channels": 2}),
                         ("width", {"width": 1})):
            with self.subTest(differs=name):
                clips = [make_clip(self.dir / f"a_{name}.wav"),
                         make_clip(self.dir / f"b_{name}.wav", **kw)]
                out = self.dir / f"out_{name}.wav"
                self.assertFalse(server.stitch_wavs(clips, out, 0.5))
                # It used to return from inside the `with`, leaving a wav that
                # looked like the take beside the zip the caller then made.
                self.assertFalse(out.exists(), "half-built file left behind")

    def test_a_mismatch_on_the_last_clip_is_still_caught(self):
        clips = [make_clip(self.dir / "p.wav"), make_clip(self.dir / "q.wav"),
                 make_clip(self.dir / "r.wav", rate=8000)]
        out = self.dir / "late.wav"
        self.assertFalse(server.stitch_wavs(clips, out, 0.25))
        self.assertFalse(out.exists())


class EngineAddress(unittest.TestCase):
    """CLAUDE.md rule 9: a typed address must not stop the server booting."""

    def test_addresses_people_actually_type(self):
        cases = {
            "http://127.0.0.1:8188": ("http://127.0.0.1:8188", 8188),
            "http://127.0.0.1:8188/": ("http://127.0.0.1:8188", 8188),
            "  http://host:9000/  ": ("http://host:9000", 9000),
            "localhost:8188": ("http://localhost:8188", 8188),
            "http://localhost": ("http://localhost", 8188),
            "": ("", 8188),
            "garbage://x": ("", 8188),
            "http://a b:1": ("", 8188),
        }
        for raw, (url, port) in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(bootstrap.clean_url(raw), url)
                self.assertEqual(bootstrap.comfy_port(raw), port)

    def test_comfy_port_never_raises(self):
        for raw in (None, "", "http://", ":::", "http://h:notaport", "x" * 500):
            with self.subTest(raw=raw):
                self.assertIsInstance(bootstrap.comfy_port(raw or ""), int)

    def test_a_saved_address_is_healed_on_load(self):
        data = Path(_SANDBOX) / "heal"
        data.mkdir(parents=True, exist_ok=True)
        old_dir, old_path = bootstrap.DATA_DIR, bootstrap.CONFIG_PATH
        bootstrap.DATA_DIR, bootstrap.CONFIG_PATH = data, data / "config.json"
        try:
            bootstrap.CONFIG_PATH.write_text('{"comfy_url": "127.0.0.1:8188/"}')
            self.assertEqual(bootstrap.load_config()["comfy_url"],
                             "http://127.0.0.1:8188")
            bootstrap.CONFIG_PATH.write_text('{ "comfy_url": ')  # truncated
            self.assertEqual(bootstrap.load_config()["comfy_url"],
                             bootstrap.DEFAULT_COMFY_URL)
        finally:
            bootstrap.DATA_DIR, bootstrap.CONFIG_PATH = old_dir, old_path


class ModelDeletes(unittest.TestCase):
    """CLAUDE.md rule 8: a delete stays under models_dir/qwen-tts."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.models = self.root / "models"
        (self.models / "qwen-tts" / "Qwen" / "Real").mkdir(parents=True)
        (self.models / "qwen-tts" / "Qwen" / "Real" / "w.safetensors").write_text("x")
        (self.models / "checkpoints").mkdir(parents=True)
        self.precious = self.models / "checkpoints" / "keep.safetensors"
        self.precious.write_text("do not delete")
        self.outside = self.root / "elsewhere"
        self.outside.mkdir()
        (self.outside / "f.txt").write_text("x")
        # Qwen's own install, since that is whose folder a Qwen repo lives in.
        self.cfg = dict(bootstrap.DEFAULT_CONFIG)
        self.cfg["engines"] = {
            eid: dict(bootstrap.engine_defaults(eid),
                      comfy_dir=str(self.root), models_dir=str(self.models))
            for eid in bootstrap.ENGINES}

    def test_traversal_is_refused(self):
        for repo in ("Qwen/../../checkpoints", "Qwen/../../../elsewhere",
                     "../../../../etc", "Qwen/./../../checkpoints",
                     r"Qwen\..\..\checkpoints", "Qwen", "nothing",
                     str(self.outside), "/etc/passwd"):
            with self.subTest(repo=repo):
                with self.assertRaises(Exception):
                    manager.delete_model(self.cfg, repo)
        self.assertTrue(self.precious.exists())
        self.assertTrue((self.outside / "f.txt").exists())

    def test_a_symlink_out_of_the_tree_is_refused(self):
        link = self.models / "qwen-tts" / "Qwen" / "Escape"
        link.symlink_to(self.outside, target_is_directory=True)
        with self.assertRaises(Exception):
            manager.delete_model(self.cfg, "Qwen/Escape")
        self.assertTrue((self.outside / "f.txt").exists())

    def test_a_real_delete_still_works(self):
        manager.delete_model(self.cfg, "Qwen/Real")
        self.assertFalse((self.models / "qwen-tts" / "Qwen" / "Real").exists())
        self.assertTrue(self.precious.exists())


class ModelInstalled(unittest.TestCase):
    """CLAUDE.md rule 6: a folder with a .part in it is not installed."""

    def setUp(self):
        self.models = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.models, ignore_errors=True)

    def _folder(self, repo):
        d = bootstrap.qwen_model_dir(self.models, repo)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_weights_present_counts_as_installed(self):
        d = self._folder("Qwen/A")
        (d / "model.safetensors").write_text("w")
        self.assertTrue(bootstrap.model_installed(self.models, "Qwen/A"))

    def test_a_part_file_means_not_installed(self):
        # config.json lands first and is perfectly good, which is exactly why
        # this has to be checked: the weights are still half here.
        d = self._folder("Qwen/B")
        (d / "config.json").write_text("{}")
        (d / "model.safetensors.part").write_text("half")
        self.assertFalse(bootstrap.model_installed(self.models, "Qwen/B"))

    def test_a_missing_folder_is_not_installed(self):
        self.assertFalse(bootstrap.model_installed(self.models, "Qwen/Nope"))


class PipProgress(unittest.TestCase):
    """CLAUDE.md rule 16: the percentage behind the setup panel's bar."""

    def test_a_multi_gigabyte_download_reports_sensibly(self):
        state, total = {}, 2_713_000_000
        self.assertIsNotNone(bootstrap.pip_progress(
            "Downloading torch-2.5.1+cu128-cp312-cp312-win_amd64.whl (2.7 GB)",
            state))
        out = bootstrap.pip_progress(f"Progress {total // 2} of {total}", state)
        self.assertIsNotNone(out)
        text, pct = out
        self.assertAlmostEqual(pct, 50.0, places=0)
        self.assertIn("torch", text)
        self.assertIn("GB", text)

    def test_the_percentage_cannot_leave_its_ends(self):
        for line, want in (("Progress -5 of 100", 0.0),
                           ("Progress 500 of 100", 100.0),
                           ("Progress 50 of 100", 50.0)):
            with self.subTest(line=line):
                self.assertEqual(bootstrap.pip_progress(line, {})[1], want)

    def test_malformed_output_cannot_throw(self):
        for junk in ("", "Progress", "Progress abc of def", "Progress 10 of",
                     "Progress 5 of 0", "Downloading", "random chatter",
                     "Installing collected packages: torch"):
            with self.subTest(junk=junk):
                bootstrap.pip_progress(junk, {})   # must not raise

    def test_carriage_returns_count_as_line_breaks(self):
        # pip redraws progress with \r; splitting on \n alone means nothing
        # appears until the download has already finished.
        import io
        stream = io.StringIO("Progress 1 of 9\rProgress 5 of 9\rdone\n")
        self.assertEqual(list(bootstrap.stream_lines(stream)),
                         ["Progress 1 of 9", "Progress 5 of 9", "done"])

    def test_human_size_scales(self):
        self.assertEqual(bootstrap.human_size(512), "512 B")
        self.assertEqual(bootstrap.human_size(921_200), "921 kB")
        self.assertEqual(bootstrap.human_size(16_900_000), "17 MB")
        self.assertEqual(bootstrap.human_size(1_800_000), "1.8 MB")
        self.assertEqual(bootstrap.human_size(2_713_000_000), "2.71 GB")


class SetupSteps(unittest.TestCase):
    """CLAUDE.md rule 10: steps cross the wire in run order, not sorted."""

    def test_steps_are_a_list_in_the_order_they_run(self):
        prog = bootstrap.Progress()
        snap = prog.snapshot()
        self.assertIsInstance(snap["steps"], list)
        self.assertEqual([s["key"] for s in snap["steps"]],
                         ["python", "comfyui", "node", "deps", "models",
                          "launch"])

    def test_a_step_has_no_percentage_until_there_is_a_real_one(self):
        prog = bootstrap.Progress()
        prog.begin("deps")
        self.assertIsNone(prog.snapshot()["steps"][3]["pct"])
        prog.detail("deps", "downloading", 42.0)
        self.assertEqual(prog.snapshot()["steps"][3]["pct"], 42.0)
        # Not "keeps the last number", which is what it used to do: pip goes
        # quiet for the whole of the unpacking step, and a bar frozen at the
        # download's last percentage reads as a run that finished and hung.
        prog.detail("deps", "unpacking", None)
        self.assertIsNone(prog.snapshot()["steps"][3]["pct"])
        prog.finish("deps", "done")
        self.assertIsNone(prog.snapshot()["steps"][3]["pct"])

    def test_a_percentage_is_clamped(self):
        prog = bootstrap.Progress()
        prog.begin("models")
        prog.detail("models", "x", 999)
        self.assertEqual(prog.snapshot()["steps"][4]["pct"], 100.0)
        prog.detail("models", "x", -50)
        self.assertEqual(prog.snapshot()["steps"][4]["pct"], 0.0)


class TakesRecord(unittest.TestCase):
    """CLAUDE.md rule 12: read, change and write under one hold of the lock."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self._old = server.TAKES_PATH, server.DATA_DIR
        server.DATA_DIR = self.dir
        server.TAKES_PATH = self.dir / "takes.json"

    def tearDown(self):
        server.TAKES_PATH, server.DATA_DIR = self._old

    def test_simultaneous_adds_do_not_lose_takes(self):
        # Two jobs finishing together each used to write the list they had read
        # before the other's take was in it. Forty adds kept eleven.
        n = 40
        gate = threading.Barrier(n)

        def add(i):
            gate.wait()
            server.add_take({"id": f"t{i:03d}", "title": str(i), "lines": [],
                             "file": "take.wav", "bundle": "", "created": 0})

        threads = [threading.Thread(target=add, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        got = server.read_takes()
        self.assertEqual(len(got), n)
        self.assertEqual({t["id"] for t in got},
                         {f"t{i:03d}" for i in range(n)})

    def test_the_file_survives_concurrent_adds_and_deletes(self):
        stop = threading.Event()
        errors: list = []

        def churn(fn, tag):
            try:
                i = 0
                while not stop.is_set():
                    fn(f"{tag}{i}")
                    i += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def adder(ident):
            server.add_take({"id": ident, "title": "a", "lines": [],
                             "file": "f", "bundle": "", "created": 0})

        def deleter(_ident):
            cur = server.read_takes()
            if cur:
                server.remove_take(cur[0]["id"])

        # Each adder gets its own id prefix, so a duplicate really would mean
        # two writers had trodden on each other rather than just counting alike.
        threads = [threading.Thread(target=churn, args=(adder, "a")),
                   threading.Thread(target=churn, args=(deleter, "d")),
                   threading.Thread(target=churn, args=(adder, "b"))]
        for t in threads:
            t.start()
        threading.Event().wait(1.5)
        stop.set()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        ids = [t["id"] for t in server.read_takes()]   # parses, so not corrupt
        self.assertEqual(len(ids), len(set(ids)), "duplicate ids")
        self.assertFalse(list(self.dir.glob("*.tmp")), "left a .tmp behind")

    def test_the_library_is_capped(self):
        for i in range(210):
            server.add_take({"id": f"c{i}", "title": "c", "lines": [],
                             "file": "f", "bundle": "", "created": 0})
        self.assertEqual(len(server.read_takes()), 200)


ATTENTIONS = ["auto", "sdpa"]


def custom_voice(required_extra=None, drop_required=(), optional_extra=None,
                 drop_optional=()):
    required = {
        "text": ["STRING", {"default": ""}],
        "speaker": [["Aiden", "Serena"], {"default": "Aiden"}],
        "model_choice": [["0.6B", "1.7B"], {"default": "1.7B"}],
        "language": [["Auto"], {"default": "Auto"}],
    }
    optional = {
        "instruct": ["STRING", {"default": ""}],
        "attention": [ATTENTIONS, {"default": "auto"}],
        "unload_model_after_generate": ["BOOLEAN", {"default": False}],
        "seed": ["INT", {"default": 0}],
        "temperature": ["FLOAT", {"default": 0.9}],
    }
    required.update(required_extra or {})
    optional.update(optional_extra or {})
    for k in drop_required:
        required.pop(k, None)
    for k in drop_optional:
        optional.pop(k, None)
    return {"input": {"required": required, "optional": optional}}


SAVE = {"SaveAudioAdvanced": {"input": {"required": {
    "audio": ["AUDIO"], "filename_prefix": ["STRING", {"default": "x"}],
    "format": [["flac", "wav"], {"default": "flac"}]}}}}

OPTS = {"style": "warm", "model": "0.6B", "attention": "sdpa", "unload": True,
        "temperature": 0.7, "prefer_wav": True}


def split_cfg(root: Path, **extra) -> dict:
    """A config in the per-engine shape: each engine its own ComfyUI folder,
    its own models folder and its own port."""
    cfg = dict(bootstrap.DEFAULT_CONFIG, **extra)
    cfg["engines"] = {}
    for eid, eng in bootstrap.ENGINES.items():
        d = root / eng["dir_name"]
        (d / "models").mkdir(parents=True, exist_ok=True)
        cfg["engines"][eid] = dict(bootstrap.engine_defaults(eid),
                                   comfy_dir=str(d),
                                   models_dir=str(d / "models"))
    return cfg


def client_for(schema):
    c = comfy.ComfyClient()
    c._schema = dict(schema)
    c._schema_at = 9e18          # never refetch during a test
    return c


class GraphBuilding(unittest.TestCase):
    """CLAUDE.md rules 2 and 3: graphs come from the live schema, through
    candidate-name lists, so a node that renames an input still works."""

    def _built(self, schema):
        g = client_for(schema).build_line(
            {"text": "Hello."}, {"kind": "preset", "speaker": "Serena"}, OPTS)
        return g["prompt"]["2"]["inputs"]

    def test_the_node_as_it_ships_today(self):
        ins = self._built({"CustomVoiceNode": custom_voice(), **SAVE})
        self.assertEqual(ins["text"], "Hello.")
        self.assertEqual(ins["speaker"], "Serena")
        self.assertEqual(ins["model_choice"], "0.6B")
        self.assertEqual(ins["attention"], "sdpa")
        self.assertTrue(ins["unload_model_after_generate"])
        self.assertEqual(ins["instruct"], "warm")

    def test_renamed_inputs_are_still_found(self):
        renames = [
            ({"speaker_name": [["Aiden", "Serena"], {}]}, ("speaker",), "speaker_name", "Serena"),
            ({"voice": [["Aiden", "Serena"], {}]}, ("speaker",), "voice", "Serena"),
            ({"target_text": ["STRING", {"default": ""}]}, ("text",), "target_text", "Hello."),
            ({"model": [["0.6B", "1.7B"], {}]}, ("model_choice",), "model", "0.6B"),
        ]
        for extra, dropped, key, want in renames:
            with self.subTest(renamed_to=key):
                schema = {"CustomVoiceNode": custom_voice(
                    required_extra=extra, drop_required=dropped), **SAVE}
                self.assertEqual(self._built(schema)[key], want)

    def test_a_renamed_optional_input_is_still_found(self):
        schema = {"CustomVoiceNode": custom_voice(
            optional_extra={"unload_model": ["BOOLEAN", {"default": False}]},
            drop_optional=("unload_model_after_generate",)), **SAVE}
        self.assertTrue(self._built(schema)["unload_model"])

    def test_an_unknown_required_input_picks_up_its_own_default(self):
        schema = {"CustomVoiceNode": custom_voice(required_extra={
            "emotion_scale": ["FLOAT", {"default": 1.25}],
            "dialect": [["neutral", "rp"], {"default": "rp"}]}), **SAVE}
        ins = self._built(schema)
        self.assertEqual(ins["emotion_scale"], 1.25)
        self.assertEqual(ins["dialect"], "rp")

    def test_a_missing_essential_input_is_a_readable_refusal(self):
        schema = {"CustomVoiceNode": custom_voice(drop_required=("speaker",)),
                  **SAVE}
        with self.assertRaises(comfy.ComfyError) as caught:
            self._built(schema)
        self.assertIn("speaker", str(caught.exception))

    def test_nodes_not_loaded_says_so(self):
        with self.assertRaises(comfy.ComfyError) as caught:
            self._built(dict(SAVE))
        self.assertIn("not loaded", str(caught.exception))

    def test_it_falls_back_from_saveaudioadvanced_to_saveaudio(self):
        schema = {"CustomVoiceNode": custom_voice(),
                  "SaveAudio": {"input": {"required": {
                      "audio": ["AUDIO"],
                      "filename_prefix": ["STRING", {"default": "x"}]}}}}
        g = client_for(schema).build_line(
            {"text": "Hi."}, {"kind": "preset", "speaker": "Serena"}, OPTS)
        self.assertEqual(g["prompt"]["3"]["class_type"], "SaveAudio")

    def test_no_save_node_at_all_says_so(self):
        with self.assertRaises(comfy.ComfyError):
            self._built({"CustomVoiceNode": custom_voice()})

    def test_every_line_gets_its_own_seed(self):
        # A fixed seed made a take byte-identical to the last one, so retrying
        # a line gave back exactly what it gave before.
        c = client_for({"CustomVoiceNode": custom_voice(), **SAVE})
        seeds = {c.build_line({"text": "x"},
                              {"kind": "preset", "speaker": "Serena"},
                              OPTS)["prompt"]["2"]["inputs"]["seed"]
                 for _ in range(8)}
        self.assertGreater(len(seeds), 6, "seeds are not varying")

    def test_a_designed_voice_never_asks_for_the_0_6b_build(self):
        # VoiceDesignNode raises on 0.6B, and 0.6B is what the model picker
        # offers first because cloning has one.
        schema = {"VoiceDesignNode": {"input": {"required": {
            "text": ["STRING", {"default": ""}],
            "instruct": ["STRING", {"default": ""}],
            "model_choice": [["0.6B", "1.7B"], {"default": "1.7B"}],
            "language": [["Auto"], {"default": "Auto"}]}}}, **SAVE}
        for picked in ("0.6B", "1.7B", ""):
            with self.subTest(picker=picked):
                g = client_for(schema).build_line(
                    {"text": "hi"}, {"kind": "design", "instruct": "gruff"},
                    dict(OPTS, model=picked))
                self.assertEqual(g["prompt"]["2"]["inputs"]["model_choice"],
                                 "1.7B")

    def test_the_1_7b_choice_is_read_off_the_node_not_typed_in(self):
        schema = {"VoiceDesignNode": {"input": {"required": {
            "text": ["STRING", {"default": ""}],
            "instruct": ["STRING", {"default": ""}],
            "model_choice": [["1.7B-VoiceDesign"], {}]}}}, **SAVE}
        g = client_for(schema).build_line(
            {"text": "hi"}, {"kind": "design", "instruct": "gruff"},
            dict(OPTS, model="0.6B"))
        self.assertEqual(g["prompt"]["2"]["inputs"]["model_choice"],
                         "1.7B-VoiceDesign")

    def test_a_cloned_voice_wires_loadaudio_into_the_clone_node(self):
        schema = {"VoiceCloneNode": {"input": {"required": {
            "ref_audio": ["AUDIO"], "ref_text": ["STRING", {"default": ""}],
            "target_text": ["STRING", {"default": ""}],
            "model_choice": [["0.6B", "1.7B"], {"default": "0.6B"}]}}},
            "LoadAudio": {"input": {"required": {
                "audio": [["ref.wav"], {"audio_upload": True}]}}}, **SAVE}
        g = client_for(schema).build_line(
            {"text": "hi"},
            {"kind": "clone", "ref_audio": "ref.wav", "ref_text": "spoken"},
            OPTS)["prompt"]
        self.assertEqual(g["1"]["class_type"], "LoadAudio")
        self.assertEqual(g["2"]["inputs"]["ref_audio"], ["1", 0])
        self.assertEqual(g["2"]["inputs"]["target_text"], "hi")

    def test_a_clone_with_no_reference_audio_says_so(self):
        schema = {"VoiceCloneNode": {"input": {"required": {
            "ref_audio": ["AUDIO"],
            "target_text": ["STRING", {"default": ""}]}}}, **SAVE}
        with self.assertRaises(comfy.ComfyError) as caught:
            client_for(schema).build_line({"text": "hi"}, {"kind": "clone"},
                                          OPTS)
        self.assertIn("reference audio", str(caught.exception))

    def test_an_empty_line_is_refused(self):
        with self.assertRaises(comfy.ComfyError):
            client_for({"CustomVoiceNode": custom_voice(), **SAVE}).build_line(
                {"text": "   "}, {"kind": "preset", "speaker": "Serena"}, OPTS)


class DeadEngine(unittest.TestCase):
    """CLAUDE.md rule 15: a dead engine is a sentence, not a stack trace."""

    def test_a_refused_connection_reads_as_english(self):
        # Nothing is listening on this port.
        c = comfy.ComfyClient("http://127.0.0.1:1")
        with self.assertRaises(comfy.ComfyError) as caught:
            c.queue({"1": {"class_type": "X", "inputs": {}}})
        message = str(caught.exception)
        self.assertIn("stopped answering", message)
        self.assertNotIn("HTTPConnectionPool", message)


class WhichEngineALaunchOpensOn(unittest.TestCase):
    """Qwen is primary: every launch opens on it, whichever engine the last
    session ended on. MOSS is a switch made on the Create page and lasting that
    session — an app that quietly came back up holding the secondary engine's
    models has chosen for you, and on 8 GB that choice costs the card. Being
    secondary is about what is loaded at launch, never about what is
    installed: first run still fetches both."""

    def test_the_primary_is_where_a_launch_starts(self):
        self.assertEqual(bootstrap.start_engine({}), "qwen")
        self.assertEqual(bootstrap.PRIMARY_ENGINE, "qwen")

    def test_last_session_ending_on_moss_does_not_move_it(self):
        self.assertEqual(bootstrap.start_engine({"engine": "moss"}), "qwen")

    def test_turning_the_primary_off_falls_to_one_that_is_on(self):
        # Qwen cannot be turned off today, so this is the shape of the answer
        # rather than a live case: never return an engine that is disabled.
        with mock.patch.object(bootstrap, "engine_enabled",
                               side_effect=lambda c, e: e == "moss"):
            self.assertEqual(bootstrap.start_engine({}), "moss")

    def test_both_engines_are_still_wanted_on_a_first_run(self):
        # Secondary is not "optional": the setup sheet ticks MOSS by default
        # and its models are in the first-run list.
        wanted = {m["repo"] for m in bootstrap.wanted_models(
            dict(bootstrap.DEFAULT_CONFIG))}
        self.assertTrue(any(r.startswith("OpenMOSS-Team/") for r in wanted))
        self.assertTrue(any(r.startswith("Qwen/") for r in wanted))

    def test_the_roles_are_on_the_engines_themselves(self):
        self.assertEqual(bootstrap.ENGINES["qwen"]["role"], "primary")
        self.assertEqual(bootstrap.ENGINES["moss"]["role"], "secondary")


class AutoStartingASlowEngine(unittest.TestCase):
    """Starting an engine is not the same as it being ready, and the boot path
    asks for exactly that: launch it, do not wait. Reading the schema straight
    afterwards raised out of activate() — a few lines under a comment promising
    that an engine which cannot start is never a reason the app fails to boot,
    which is precisely what it became."""

    def setUp(self):
        self.saved = copy.deepcopy(server.cfg)
        self.root = Path(tempfile.mkdtemp(prefix="sb-slow-"))
        (self.root / "main.py").write_text("")

    def tearDown(self):
        server.cfg.clear()
        server.cfg.update(self.saved)
        shutil.rmtree(self.root, ignore_errors=True)

    def test_activate_returns_rather_than_raising_while_it_comes_up(self):
        slot = bootstrap.engine_cfg(server.cfg, "qwen")
        slot.update({"comfy_dir": str(self.root), "python": sys.executable,
                     "managed": True, "auto_start": True,
                     # Nothing is listening, and nothing will be for a while.
                     "comfy_url": "http://127.0.0.1:1"})
        with mock.patch.object(server.PROCS["qwen"], "start"), \
             mock.patch.object(server, "comfy_online", return_value=False):
            self.assertEqual(server.activate("qwen", wait=False), "")


class WhoIsAnsweringThePort(unittest.TestCase):
    """8188 is the port every ComfyUI picks, so the one answering is often
    somebody else's — and that has every symptom of nodes that failed to load:
    the folders are all on disk, the install is complete, and the classes are
    not there. The engine row has to say which install answered."""

    def _client(self, payload):
        c = comfy.ComfyClient("http://127.0.0.1:1")

        class Resp:
            def raise_for_status(self): pass
            def json(self): return payload

        with mock.patch.object(comfy.requests, "get", return_value=Resp()):
            return c.engine_root()

    def test_the_folder_comes_off_the_reported_argv(self):
        self.assertEqual(
            self._client({"system": {"argv": ["/opt/ComfyUI/main.py", "--port"]}}),
            "/opt/ComfyUI")

    def test_a_windows_path_is_read_on_any_platform(self):
        self.assertEqual(
            self._client({"system": {"argv": ["D:\\AI\\ComfyUI\\main.py"]}}),
            "D:\\AI\\ComfyUI")

    def test_a_build_that_does_not_report_argv_is_not_a_mismatch(self):
        # Older ComfyUI says nothing here. Empty must never read as "someone
        # else's" — crying wolf about the usual case teaches people to ignore
        # the row that matters.
        self.assertEqual(self._client({"system": {"os": "posix"}}), "")
        self.assertTrue(manager.same_install("/opt/ComfyUI", ""))

    def test_the_same_folder_written_two_ways_is_one_install(self):
        self.assertTrue(manager.same_install("/opt/ComfyUI", "/opt/ComfyUI/"))
        self.assertTrue(manager.same_install("/opt/./ComfyUI", "/opt/ComfyUI"))
        self.assertFalse(manager.same_install("/opt/ComfyUI", "/opt/OtherUI"))

    def test_a_green_row_says_which_install_answered(self):
        row = manager.engine_row("Qwen3-TTS", "_qwen", "http://127.0.0.1:8188",
                                 True, "/opt/Qwen", "/opt/Qwen")
        self.assertEqual(row["state"], "ok")
        self.assertIn("/opt/Qwen", row["detail"])

    def test_an_unverifiable_engine_says_so_instead_of_only_ok(self):
        row = manager.engine_row("Qwen3-TTS", "_qwen", "http://127.0.0.1:8188",
                                 True, "/opt/Qwen", "")
        self.assertEqual(row["state"], "ok")
        self.assertIn("cannot be confirmed", row["detail"])

    def test_a_different_ComfyUI_on_the_port_is_a_warning_naming_both(self):
        row = manager.engine_row("Qwen3-TTS", "_qwen", "http://127.0.0.1:8188",
                                 True, "/opt/Qwen", "/home/me/ComfyUI")
        self.assertEqual(row["state"], "warn")
        self.assertIn("/home/me/ComfyUI", row["detail"])
        self.assertIn("/opt/Qwen", row["detail"])

    def test_an_engine_that_is_off_still_offers_start(self):
        row = manager.engine_row("MOSS-TTS", "_moss", "http://127.0.0.1:8189",
                                 False, "/opt/Moss", "")
        self.assertEqual(row["state"], "off")
        self.assertEqual(row["action"], "start")


class ComfyUISomeoneElseStarted(unittest.TestCase):
    """CLAUDE.md rule 18: the sentence names the blocker it can actually clear.

    Connecting to a ComfyUI you start yourself records managed False with no
    comfy_dir. When that ComfyUI stops answering, "run setup for it from the
    Engine panel" sends you to a dialog that cannot start someone else's
    process — the address is the thing that can be acted on.
    """

    def setUp(self):
        self.saved = copy.deepcopy(server.cfg)

    def tearDown(self):
        server.cfg.clear()
        server.cfg.update(self.saved)

    def _external(self, engine="qwen"):
        slot = bootstrap.engine_cfg(server.cfg, engine)
        slot.update({"managed": False, "comfy_dir": "", "python": "",
                     "auto_start": True,
                     # Nothing is listening here.
                     "comfy_url": "http://127.0.0.1:1"})
        return slot

    def test_a_ComfyUI_that_stopped_names_its_address(self):
        slot = self._external()
        why = server.activate("qwen")
        self.assertIn(slot["comfy_url"], why)
        self.assertNotIn("run setup", why.lower())

    def test_an_engine_never_set_up_still_says_run_setup(self):
        slot = bootstrap.engine_cfg(server.cfg, "qwen")
        slot.update({"managed": True, "comfy_dir": "", "python": "",
                     "comfy_url": "http://127.0.0.1:1"})
        self.assertIn("run setup", server.activate("qwen").lower())

    def test_one_started_elsewhere_is_told_apart_from_one_on_disk(self):
        # Picking a ComfyUI already on disk also records managed False, but it
        # has a folder — that one Script Builder can start.
        self.assertTrue(server.started_elsewhere(
            {"managed": False, "comfy_dir": ""}))
        self.assertFalse(server.started_elsewhere(
            {"managed": False, "comfy_dir": "/opt/ComfyUI"}))
        self.assertFalse(server.started_elsewhere(
            {"managed": True, "comfy_dir": ""}))

    def test_restart_says_who_has_to_do_it(self):
        self._external()
        with server.app.test_client() as web:
            r = web.post("/api/comfy/restart?engine=qwen")
        self.assertEqual(r.status_code, 400)
        self.assertIn("127.0.0.1:1", r.get_json()["error"])


class Interpreters(unittest.TestCase):
    """CLAUDE.md rules 4 and 5: found by execution, and an existing install
    keeps its own environment."""

    def test_an_existing_install_is_found_by_running_it(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        comfy_dir = root / "ComfyUI"
        (comfy_dir / "venv" / "bin").mkdir(parents=True)
        (comfy_dir / "main.py").write_text("")
        target = comfy_dir / "venv" / "bin" / "python"
        if os.name == "nt":                 # pragma: no cover - posix CI
            self.skipTest("posix layout only")
        os.symlink(sys.executable, target)
        self.assertEqual(bootstrap.existing_python(comfy_dir), str(target))

    def test_a_path_that_does_not_run_is_not_accepted(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        comfy_dir = root / "ComfyUI"
        (comfy_dir / "venv" / "bin").mkdir(parents=True)
        (comfy_dir / "main.py").write_text("")
        # Present on disk, not executable — the Windows Store stub problem.
        (comfy_dir / "venv" / "bin" / "python").write_text("not a program")
        self.assertEqual(bootstrap.existing_python(comfy_dir), "")


class DownloadFiltering(unittest.TestCase):
    """CLAUDE.md rule 6: .bin duplicates of safetensors and repo furniture."""

    def test_bin_duplicates_and_furniture_are_skipped(self):
        files = [{"path": "config.json", "size": 1},
                 {"path": "model.safetensors", "size": 9},
                 {"path": "pytorch_model.bin", "size": 9},
                 {"path": "README.md", "size": 1},
                 {"path": ".gitattributes", "size": 1},
                 {"path": "tokenizer.json", "size": 2}]
        kept = {f["path"] for f in bootstrap.wanted_files(files)}
        self.assertEqual(kept, {"config.json", "model.safetensors",
                                "tokenizer.json"})

    def test_a_bin_with_no_safetensors_twin_is_kept(self):
        files = [{"path": "pytorch_model.bin", "size": 9}]
        self.assertEqual([f["path"] for f in bootstrap.wanted_files(files)],
                         ["pytorch_model.bin"])


class GpuDetection(unittest.TestCase):
    """torch 2.14.0+cpu on a machine holding an RTX 4060, because the only
    test for a GPU was shutil.which("nvidia-smi")."""

    def setUp(self):
        bootstrap._GPU.clear()

    def tearDown(self):
        bootstrap._GPU.clear()

    def test_nvidia_smi_is_looked_for_off_PATH_too(self):
        # The regression exactly: which() comes back empty and the machine is
        # declared GPU-less. There must still be somewhere left to look.
        with mock.patch.object(bootstrap.shutil, "which", return_value=None):
            self.assertTrue(bootstrap._smi_candidates())

    def test_a_card_that_answers_is_a_card_with_a_driver(self):
        done = subprocess.CompletedProcess([], 0,
                                           "NVIDIA GeForce RTX 4060, 8188\n", "")
        with mock.patch.object(bootstrap, "_run", return_value=done):
            gpu = bootstrap.nvidia_gpu(refresh=True)
        self.assertEqual(gpu, {"name": "NVIDIA GeForce RTX 4060",
                               "driver": True, "vram_mb": 8188})

    def test_a_card_with_no_driver_is_still_a_card(self):
        with mock.patch.object(bootstrap, "_run",
                               return_value=subprocess.CompletedProcess([], 1, "", "")), \
             mock.patch.object(bootstrap, "_adapter_names",
                               return_value=["NVIDIA GeForce RTX 4060"]):
            gpu = bootstrap.nvidia_gpu(refresh=True)
        self.assertEqual(gpu["name"], "NVIDIA GeForce RTX 4060")
        self.assertFalse(gpu["driver"])

    def test_the_index_follows_the_hardware(self):
        with mock.patch.object(bootstrap, "nvidia_gpu",
                               return_value={"name": "RTX 4060", "driver": True}):
            self.assertEqual(bootstrap.torch_index({}), bootstrap.CUDA_INDEX)
        with mock.patch.object(bootstrap, "nvidia_gpu",
                               return_value={"name": "", "driver": False}):
            self.assertEqual(bootstrap.torch_index({}), bootstrap.CPU_INDEX)

    def test_a_chosen_index_is_never_second_guessed(self):
        picked = "https://download.pytorch.org/whl/cu121"
        with mock.patch.object(bootstrap, "nvidia_gpu",
                               return_value={"name": "", "driver": False}):
            self.assertEqual(bootstrap.torch_index({"torch_index": picked}),
                             picked)

    def test_the_build_tag_comes_out_of_the_index(self):
        for index, want in ((bootstrap.CUDA_INDEX, "cu128"),
                            (bootstrap.CPU_INDEX, "cpu"),
                            ("https://download.pytorch.org/whl/rocm6.2", "rocm6.2"),
                            ("", ""), ("https://pypi.org/simple", "")):
            with self.subTest(index=index):
                self.assertEqual(bootstrap.torch_build(index), want)

    def test_no_gpu_message_names_the_card_it_can_see(self):
        with mock.patch.object(bootstrap, "nvidia_gpu",
                               return_value={"name": "NVIDIA GeForce RTX 4060",
                                             "driver": True}):
            text = manager.no_cuda_reason("2.14.0+cpu")
        self.assertIn("RTX 4060", text)
        self.assertIn("Reinstall", text)
        self.assertNotIn("no GPU found", text)

    def test_no_gpu_message_says_so_when_there_really_is_none(self):
        with mock.patch.object(bootstrap, "nvidia_gpu",
                               return_value={"name": "", "driver": False}):
            self.assertIn("no NVIDIA GPU", manager.no_cuda_reason("2.14.0+cpu"))


class TorchReinstall(unittest.TestCase):
    """pip counts torch 2.14.0+cpu as satisfying `torch`, so Reinstall against
    the CUDA index changed nothing at all."""

    def _attempt(self, installed, index):
        calls = []
        with mock.patch.object(bootstrap, "installed_torch",
                               return_value=installed), \
             mock.patch.object(bootstrap, "_run",
                               side_effect=lambda cmd, **kw: calls.append(cmd)
                               or subprocess.CompletedProcess(cmd, 0, "", "")):
            dropped = bootstrap.drop_mismatched_torch("py", index, lambda _m: None)
        return dropped, calls

    def test_a_cpu_build_is_removed_before_the_cuda_one_lands(self):
        dropped, calls = self._attempt("2.14.0+cpu", bootstrap.CUDA_INDEX)
        self.assertTrue(dropped)
        self.assertTrue(any("uninstall" in c for c in calls[0]))

    def test_a_matching_build_is_left_alone(self):
        dropped, calls = self._attempt("2.14.0+cu128", bootstrap.CUDA_INDEX)
        self.assertFalse(dropped)
        self.assertEqual(calls, [])

    def test_an_untagged_wheel_is_not_reinstalled_on_a_guess(self):
        # Plain PyPI wheels carry no +tag; which build they are depends on the
        # platform, so there is nothing to compare and nothing to do.
        dropped, calls = self._attempt("2.14.0", bootstrap.CUDA_INDEX)
        self.assertFalse(dropped)
        self.assertEqual(calls, [])

    def test_nothing_is_removed_when_no_torch_is_there(self):
        dropped, calls = self._attempt("", bootstrap.CUDA_INDEX)
        self.assertFalse(dropped)
        self.assertEqual(calls, [])


class PipReadiness(unittest.TestCase):
    """`--progress-bar raw` arrived in pip 24.1 and a fresh venv ships 24.0,
    so the longest step of the install had no number at all."""

    OLD = ("  --progress-bar <progress_bar>\n"
           "                              Specify whether the progress bar "
           "should be used [on, off] (default: on)\n")
    NEW = ("  --progress-bar <progress_bar>\n"
           "                              Specify whether the progress bar "
           "should be used. [auto, on,\n"
           "                              off, raw] (default: auto)\n")

    def setUp(self):
        bootstrap._PIP_RAW.clear()
        bootstrap._PIP_READY.clear()

    tearDown = setUp

    def test_raw_is_asked_for_not_guessed_from_a_version(self):
        with mock.patch.object(bootstrap, "_run", return_value=
                               subprocess.CompletedProcess([], 0, self.OLD, "")):
            self.assertEqual(bootstrap.pip_raw_progress("py"), [])
        bootstrap._PIP_RAW.clear()
        with mock.patch.object(bootstrap, "_run", return_value=
                               subprocess.CompletedProcess([], 0, self.NEW, "")):
            self.assertEqual(bootstrap.pip_raw_progress("py"),
                             ["--progress-bar", "raw"])

    def test_the_answer_is_remembered_per_interpreter(self):
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 0,
                                                                 self.NEW, ""))
        with mock.patch.object(bootstrap, "_run", run):
            bootstrap.pip_raw_progress("py")
            bootstrap.pip_raw_progress("py")
        self.assertEqual(run.call_count, 1)

    def test_an_old_pip_is_upgraded_and_asked_again(self):
        answers = [subprocess.CompletedProcess([], 0, self.OLD, ""),   # probe
                   subprocess.CompletedProcess([], 0, "ok", ""),       # upgrade
                   subprocess.CompletedProcess([], 0, self.NEW, "")]   # re-probe
        seen = []
        with mock.patch.object(bootstrap, "_run",
                               side_effect=lambda cmd, **kw: seen.append(cmd)
                               or answers[len(seen) - 1]):
            bootstrap.pip_ready("py", lambda _m: None)
        self.assertIn("--upgrade", seen[1])
        self.assertEqual(bootstrap.pip_raw_progress("py"),
                         ["--progress-bar", "raw"])

    def test_a_pip_that_already_reports_is_left_alone(self):
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 0,
                                                                 self.NEW, ""))
        with mock.patch.object(bootstrap, "_run", run):
            bootstrap.pip_ready("py", lambda _m: None)
        self.assertEqual(run.call_count, 1)      # the probe, and no upgrade

    def test_a_failed_upgrade_is_not_a_failed_install(self):
        with mock.patch.object(bootstrap, "_run", side_effect=OSError("boom")):
            bootstrap.pip_ready("py", lambda _m: None)   # must not raise


class QuietPipPhases(unittest.TestCase):
    """pip prints nothing at all while it unpacks a 2.7 GB wheel."""

    def test_a_download_line_records_its_size(self):
        state = {}
        text, pct = bootstrap.pip_progress(
            "Downloading torch-2.5.1+cu128-cp312-cp312-win_amd64.whl (2.7 GB)",
            state)
        self.assertIsNone(pct)
        self.assertIn("2.70 GB", text)
        self.assertEqual(state["phase"], "download")

    def test_a_size_in_any_unit_reads_as_that_size(self):
        for line, want in (
                ("Downloading torch-2.5.1.whl (2.7 GB)", "2.70 GB"),
                ("Downloading torchaudio-2.5.1.whl (1.8 MB)", "1.8 MB"),
                ("Downloading six-1.17.0.whl (11 kB)", "11 kB")):
            with self.subTest(line=line):
                self.assertIn(want, bootstrap.pip_progress(line, {})[0])

    def test_the_unpack_step_counts_its_packages(self):
        state = {}
        text, pct = bootstrap.pip_progress(
            "Installing collected packages: torch, sympy, filelock", state)
        self.assertIsNone(pct)
        self.assertEqual(state["packages"], 3)
        self.assertEqual(state["phase"], "install")
        self.assertIn("3 packages", text)

    def test_the_quiet_line_shows_bytes_and_never_a_fake_percentage(self):
        state = {"what": "Unpacking 11 packages"}
        line = bootstrap.quiet_detail(state, 1_400_000_000, 123)
        self.assertIn("1.40 GB written", line)
        self.assertIn("2m 03s", line)
        self.assertNotIn("%", line)
        # Never "1.40 GB of 2.70 GB downloaded": a wheel unpacks to more than
        # it downloads, so weighing one against the other prints "199 MB of
        # 88 MB", which reads as a fault rather than as progress.
        self.assertNotIn(" of ", line)

    def test_the_clock_runs_before_anything_has_been_written(self):
        line = bootstrap.quiet_detail({"what": "Collecting torch"}, 0, 65)
        self.assertIn("Collecting torch", line)
        self.assertIn("1m 05s", line)
        self.assertNotIn("written", line)

    def test_a_small_wheel_does_not_round_to_zero(self):
        # "0 of 1 MB" reads as a stuck download rather than a small one.
        state = {"what": "pygments", "since": 0}
        text, _pct = bootstrap.pip_progress("Progress 130000 of 1300000", state)
        self.assertIn("0.1 of 1.3 MB", text)

    def test_dir_size_adds_up_and_gives_up_when_told_to(self):
        root = Path(tempfile.mkdtemp(prefix="sb-dirsize-"))
        try:
            (root / "sub").mkdir()
            (root / "a.bin").write_bytes(b"x" * 1000)
            (root / "sub" / "b.bin").write_bytes(b"y" * 2000)
            self.assertEqual(bootstrap.dir_size(root), 3000)
            self.assertLess(bootstrap.dir_size(root, budget=1), 3000)
            self.assertEqual(bootstrap.dir_size(root / "nowhere"), 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class NodeImportDiagnosis(unittest.TestCase):
    """"Installed but ComfyUI has not loaded them" is a dead end for anyone
    running from a launcher with no console to read."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="sb-node-"))
        self.node = self.root / "custom_nodes" / bootstrap.NODE_DIR_NAME
        self.node.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_the_real_exception_comes_back(self):
        (self.node / "__init__.py").write_text(
            "import a_package_that_is_not_installed\n")
        why = bootstrap.node_import_error(sys.executable, self.root)
        self.assertIn("ModuleNotFoundError", why)
        self.assertIn("a_package_that_is_not_installed", why)

    def test_an_import_that_works_says_nothing(self):
        (self.node / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n")
        self.assertEqual(bootstrap.node_import_error(sys.executable, self.root), "")

    def test_nodes_that_are_not_there_say_that_instead(self):
        shutil.rmtree(self.node)
        self.assertIn("not installed",
                      bootstrap.node_import_error(sys.executable, self.root))


class NodesNotLoaded(unittest.TestCase):
    """ComfyUI reads custom_nodes once, at startup, so installing them into a
    running engine leaves it running without them."""

    class Engine:
        def __init__(self, loaded):
            self.loaded = loaded

        def has(self, _cls):
            return self.loaded

        def engine_ready(self, _engine):
            return self.loaded

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="sb-deps-"))
        self.cfg = split_cfg(self.root)
        # Each engine has its own ComfyUI with its own node pack in it.
        for eid, eng in bootstrap.ENGINES.items():
            d = Path(self.cfg["engines"][eid]["comfy_dir"])
            (d / "main.py").write_text("")
            node = d / "custom_nodes" / eng["node_dir"]
            node.mkdir(parents=True)
            (node / eng["node_marker"]).write_text("")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _node_item(self, loaded):
        clients = {e: self.Engine(loaded) for e in bootstrap.ENGINES}
        items = manager.dependencies(self.cfg, clients)
        return next(i for i in items if i["id"] == "node_qwen")

    def test_an_engine_without_the_classes_offers_a_restart(self):
        item = self._node_item(False)
        self.assertEqual(item["state"], "warn")
        # The old text said "check the ComfyUI console", which is not a thing
        # anyone running from a launcher can do.
        self.assertEqual(item["action"], "restart")
        self.assertIn("Restart", item["detail"])

    def test_an_engine_that_loaded_them_offers_an_update(self):
        item = self._node_item(True)
        self.assertEqual(item["state"], "ok")
        self.assertEqual(item["action"], "update")

    def test_each_engine_gets_its_own_row_for_everything(self):
        clients = {e: self.Engine(True) for e in bootstrap.ENGINES}
        rows = {i["id"] for i in manager.dependencies(self.cfg, clients)}
        # Nothing below ComfyUI is shared any more, so nothing below ComfyUI
        # gets one row for both.
        for base in ("comfyui", "node", "torch", "node_reqs", "models",
                     "engine"):
            for eid in ("qwen", "moss"):
                with self.subTest(row=f"{base}_{eid}"):
                    self.assertIn(f"{base}_{eid}", rows)
        self.assertEqual({"python", "git"} & rows, {"python", "git"})

    def test_a_comfyui_we_do_not_own_is_not_reported_as_missing(self):
        # "Connect to a ComfyUI I start myself" never records a comfy_dir, so
        # the folder check finds nothing — and both node rows read "missing",
        # with an Install button, in front of someone whose engine was working
        # perfectly and whose pill said Engine ready. The running schema is the
        # better witness: if the classes are loaded, they are installed.
        bare = dict(bootstrap.DEFAULT_CONFIG)
        items = manager.dependencies(bare, {e: self.Engine(True)
                                            for e in bootstrap.ENGINES})
        rows = {i["id"]: i for i in items
                if i["id"] in ("node_qwen", "node_moss")}
        self.assertEqual({r["state"] for r in rows.values()}, {"ok"})
        self.assertTrue(all(r["action"] is None for r in rows.values()))
        self.assertIn("ComfyUI you are running", rows["node_qwen"]["detail"])

    def test_with_no_folder_and_no_engine_they_really_are_missing(self):
        items = manager.dependencies(dict(bootstrap.DEFAULT_CONFIG), None)
        rows = {i["id"]: i for i in items
                if i["id"] in ("node_qwen", "node_moss")}
        self.assertEqual({r["state"] for r in rows.values()}, {"missing"})

    def test_an_engine_turned_off_is_not_reported_as_missing(self):
        items = manager.dependencies(dict(self.cfg, want_moss=False),
                                     {e: self.Engine(True)
                                      for e in bootstrap.ENGINES})
        moss = [i for i in items if i["id"] == "comfyui_moss"][0]
        self.assertEqual(moss["state"], "off")
        self.assertIsNone(moss["action"])
        self.assertFalse([i for i in items if i["id"] == "torch_moss"])


# --------------------------------------------------------------------------- #
# MOSS-TTS, the second engine
# --------------------------------------------------------------------------- #
MOSS_VARIANTS = ["MOSS-TTS (Delay 8B)", "MOSS-TTS (Local 1.7B)",
                 "MOSS-TTSD v1.0", "MOSS-VoiceGenerator", "MOSS-SoundEffect"]

MOSS_SCHEMA = {
    "MossTTSModelLoader": {"input": {"required": {
        "model_variant": [MOSS_VARIANTS, {"default": MOSS_VARIANTS[0]}],
        "local_model_path": ["STRING", {"default": ""}],
        "codec_local_path": ["STRING", {"default": ""}]}}},
    "MossTTSGenerate": {"input": {
        "required": {"moss_pipe": ["MOSS_TTS_PIPE"],
                     "language": [["auto", "zh", "en"], {"default": "auto"}],
                     "text": ["STRING", {"default": ""}],
                     "seed": ["INT", {"default": 0}],
                     "temperature": ["FLOAT", {"default": 1.7}],
                     "top_p": ["FLOAT", {"default": 0.8}]},
        "optional": {"reference_audio": ["AUDIO"]}}},
    "MossTTSVoiceDesign": {"input": {"required": {
        "moss_pipe": ["MOSS_TTS_PIPE"],
        "language": [["auto", "zh", "en"], {"default": "auto"}],
        "text": ["STRING", {"default": ""}],
        "instruction": ["STRING", {"default": ""}],
        "seed": ["INT", {"default": 0}]}}},
    "LoadAudio": {"input": {"required": {"audio": [["ref.wav"], {}]}}},
    **SAVE,
}

MOSS_DIRS = {
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer": "/m/moss-tts/Local",
    "OpenMOSS-Team/MOSS-Audio-Tokenizer": "/m/moss-tts/Codec",
    "OpenMOSS-Team/MOSS-VoiceGenerator": "/m/moss-tts/VG",
}
MOSS_OPTS = {"engine": "moss", "moss_dirs": MOSS_DIRS, "temperature": 1.0,
             "top_p": 0.9, "prefer_wav": True}


class ModelFolderLayout(unittest.TestCase):
    """The two engines do not agree on where a model folder goes, and neither
    layout is a preference — each is where that node looks."""

    ROOT = Path("/models")

    def test_qwen_nests_by_org(self):
        self.assertEqual(
            bootstrap.model_dir(self.ROOT, "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
            self.ROOT / "qwen-tts" / "Qwen" / "Qwen3-TTS-12Hz-0.6B-Base")

    def test_moss_flattens_the_slash(self):
        # The MOSS loader builds its cache path as repo_id.replace("/", "--").
        # A folder in the Qwen shape is invisible to it, and it downloads a
        # second copy of a model that is already on disk.
        self.assertEqual(
            bootstrap.model_dir(self.ROOT,
                                "OpenMOSS-Team/MOSS-TTS-Local-Transformer"),
            self.ROOT / "moss-tts"
            / "OpenMOSS-Team--MOSS-TTS-Local-Transformer")

    def test_a_repo_knows_its_own_engine(self):
        self.assertEqual(bootstrap.engine_of("Qwen/Qwen3-TTS-Tokenizer-12Hz"),
                         "qwen")
        self.assertEqual(bootstrap.engine_of("OpenMOSS-Team/MOSS-TTS"), "moss")
        # Something neither table knows falls back rather than raising.
        self.assertEqual(bootstrap.engine_of("Someone/Else"), "qwen")

    def test_the_node_marker_differs_too(self):
        root = Path(tempfile.mkdtemp(prefix="sb-nodes-"))
        try:
            (root / "custom_nodes" / "ComfyUI-Qwen-TTS").mkdir(parents=True)
            (root / "custom_nodes" / "ComfyUI-Qwen-TTS" / "nodes.py").write_text("")
            (root / "custom_nodes" / "comfyui-moss-tts").mkdir(parents=True)
            self.assertTrue(bootstrap.node_installed(root, "qwen"))
            # The MOSS repo has no nodes.py at all — its classes live under
            # nodes/, so checking for that file would call it never installed.
            self.assertFalse(bootstrap.node_installed(root, "moss"))
            (root / "custom_nodes" / "comfyui-moss-tts" / "__init__.py").write_text("")
            self.assertTrue(bootstrap.node_installed(root, "moss"))
        finally:
            shutil.rmtree(root, ignore_errors=True)


class WhichModelsAreWanted(unittest.TestCase):
    def test_both_engines_are_asked_for_by_default(self):
        got = bootstrap.wanted_models(dict(bootstrap.DEFAULT_CONFIG))
        self.assertEqual({m["engine"] for m in got}, {"qwen", "moss"})
        self.assertIn("OpenMOSS-Team/MOSS-TTS-Local-Transformer",
                      [m["repo"] for m in got])

    def test_only_the_real_8b_is_held_back(self):
        # ~18 GB through this node, which loads bf16 weights. Downloading tens
        # of gigabytes someone cannot run is worse than not having them.
        cfg = dict(bootstrap.DEFAULT_CONFIG)
        repos = [m["repo"] for m in bootstrap.wanted_models(cfg)]
        self.assertNotIn("OpenMOSS-Team/MOSS-TTS", repos)
        cfg["want_moss_8b"] = True
        self.assertIn("OpenMOSS-Team/MOSS-TTS",
                      [m["repo"] for m in bootstrap.wanted_models(cfg)])

    def test_voice_design_is_not_mistaken_for_an_8b(self):
        # The ComfyUI node's README calls MOSS-VoiceGenerator "Delay 8B,
        # ~18 GB", conflating the architecture with the size. OpenMOSS
        # publishes it at 1.7B, and believing the node README hid MOSS voice
        # design behind a warning that it would not run on an 8 GB card.
        entry = next(m for m in bootstrap.MOSS_MODEL_REPOS
                     if m["repo"] == "OpenMOSS-Team/MOSS-VoiceGenerator")
        self.assertEqual(entry["params"], "1.7B")
        self.assertNotIn("18 GB", entry["note"])
        # …and because it fits, it is fetched by default: MOSS has no preset
        # speakers, so a description is one of only two ways to pin a voice.
        self.assertIn("OpenMOSS-Team/MOSS-VoiceGenerator",
                      [m["repo"] for m in
                       bootstrap.wanted_models(dict(bootstrap.DEFAULT_CONFIG))])

    def test_everything_fetched_by_default_fits_an_8gb_card(self):
        for m in bootstrap.wanted_models(dict(bootstrap.DEFAULT_CONFIG)):
            with self.subTest(repo=m["repo"]):
                self.assertNotIn("8B", m["params"])

    def test_turning_moss_off_leaves_only_qwen(self):
        cfg = dict(bootstrap.DEFAULT_CONFIG, want_moss=False)
        self.assertEqual({m["engine"] for m in bootstrap.wanted_models(cfg)},
                         {"qwen"})

    def test_one_engine_is_not_the_other_engine_s_problem(self):
        # With MOSS selected, a missing Qwen folder is not what stands between
        # this script and a take.
        root = Path(tempfile.mkdtemp(prefix="sb-models-"))
        try:
            cfg = dict(bootstrap.DEFAULT_CONFIG)
            only_moss = bootstrap.missing_models(root, cfg, "moss")
            self.assertTrue(only_moss)
            self.assertEqual({m["engine"] for m in only_moss}, {"moss"})
        finally:
            shutil.rmtree(root, ignore_errors=True)


class MossGraphs(unittest.TestCase):
    """MOSS splits into a loader and a generator, and every choice about which
    checkpoint that is comes off the node's own enum."""

    def setUp(self):
        self.c = client_for(MOSS_SCHEMA)

    def test_the_variant_is_read_off_the_node_not_typed_in(self):
        for repo, want in (
                ("OpenMOSS-Team/MOSS-TTS-Local-Transformer",
                 "MOSS-TTS (Local 1.7B)"),
                ("OpenMOSS-Team/MOSS-TTS", "MOSS-TTS (Delay 8B)"),
                ("OpenMOSS-Team/MOSS-VoiceGenerator", "MOSS-VoiceGenerator")):
            with self.subTest(repo=repo):
                self.assertEqual(self.c.moss_variant_for(repo), want)

    def test_a_renamed_variant_list_gives_nothing_rather_than_a_wrong_one(self):
        c = client_for({**MOSS_SCHEMA, "MossTTSModelLoader": {"input": {
            "required": {"model_variant": [["something else"], {}],
                         "local_model_path": ["STRING", {"default": ""}],
                         "codec_local_path": ["STRING", {"default": ""}]}}}})
        self.assertEqual(c.moss_variant_for("OpenMOSS-Team/MOSS-TTS"), "")
        # …and the graph then leaves model_variant on the node's own default
        # rather than sending a value it would reject.
        g = c.build_line({"text": "Hi."}, {"kind": "preset"}, MOSS_OPTS)
        self.assertEqual(g["prompt"]["1"]["inputs"]["model_variant"],
                         "something else")

    def test_the_models_own_voice_needs_neither_clip_nor_description(self):
        g = self.c.build_line({"text": "Hi."}, {"kind": "preset"}, MOSS_OPTS)
        self.assertEqual(sorted(g["prompt"]), ["1", "3", "4"])
        self.assertEqual(g["prompt"]["4"]["class_type"], "MossTTSGenerate")
        self.assertNotIn("reference_audio", g["prompt"]["4"]["inputs"])
        self.assertEqual(g["prompt"]["4"]["inputs"]["moss_pipe"], ["1", 0])

    def test_a_cloned_voice_loads_its_clip(self):
        g = self.c.build_line({"text": "Hi."},
                              {"kind": "clone", "ref_audio": "ref.wav"},
                              MOSS_OPTS)
        self.assertEqual(g["prompt"]["2"]["class_type"], "LoadAudio")
        self.assertEqual(g["prompt"]["4"]["inputs"]["reference_audio"],
                         ["2", 0])

    def test_cloning_with_no_clip_is_a_sentence(self):
        with self.assertRaises(comfy.ComfyError):
            self.c.build_line({"text": "Hi."}, {"kind": "clone"}, MOSS_OPTS)

    def test_a_designed_voice_loads_the_only_model_that_can_do_it(self):
        # MossTTSVoiceDesign prints a warning and misbehaves on anything but
        # MOSS-VoiceGenerator, so the loader is pointed at it whatever the
        # model picker says — the same reasoning as Qwen forcing 1.7B.
        g = self.c.build_line({"text": "Hi."},
                              {"kind": "design", "instruct": "A low narrator"},
                              dict(MOSS_OPTS,
                                   moss_model="OpenMOSS-Team/MOSS-TTS"))
        self.assertEqual(g["prompt"]["1"]["inputs"]["model_variant"],
                         "MOSS-VoiceGenerator")
        self.assertEqual(g["prompt"]["1"]["inputs"]["local_model_path"],
                         "/m/moss-tts/VG")
        self.assertEqual(g["prompt"]["4"]["class_type"], "MossTTSVoiceDesign")
        self.assertEqual(g["prompt"]["4"]["inputs"]["instruction"],
                         "A low narrator")

    def test_a_folder_that_is_not_there_is_sent_as_empty(self):
        # The loader treats local_model_path as a path only when it can stat
        # it, and as a HuggingFace repo id otherwise — so a folder that has not
        # been downloaded would become snapshot_download("D:\\...\\MOSS-TTS"),
        # which is not a repo id and fails. "" lets the node fetch it instead.
        g = self.c.build_line({"text": "Hi."}, {"kind": "preset"},
                              dict(MOSS_OPTS, moss_dirs={}))
        self.assertEqual(g["prompt"]["1"]["inputs"]["local_model_path"], "")
        self.assertEqual(g["prompt"]["1"]["inputs"]["codec_local_path"], "")

    def test_every_line_gets_its_own_seed(self):
        seeds = {self.c.build_line({"text": "Hi."}, {"kind": "preset"},
                                   MOSS_OPTS)["prompt"]["4"]["inputs"]["seed"]
                 for _ in range(8)}
        self.assertGreater(len(seeds), 1)

    def test_an_empty_line_is_refused_before_anything_is_queued(self):
        with self.assertRaises(comfy.ComfyError):
            self.c.build_line({"text": "   "}, {"kind": "preset"}, MOSS_OPTS)

    def test_moss_nodes_that_are_not_loaded_say_so_about_moss(self):
        c = client_for({**SAVE})
        self.assertFalse(c.engine_ready("moss"))
        with self.assertRaises(comfy.ComfyError) as caught:
            c.build_line({"text": "Hi."}, {"kind": "preset"}, MOSS_OPTS)
        self.assertIn("MOSS-TTS", str(caught.exception))

    def test_moss_reports_no_preset_speakers_rather_than_an_empty_list(self):
        caps = self.c.capabilities("moss")
        self.assertFalse(caps["preset"])
        self.assertTrue(caps["own_voice"])
        self.assertTrue(caps["clone"])

    def test_the_engines_do_not_answer_for_each_other(self):
        qwen_only = client_for({"CustomVoiceNode": custom_voice(), **SAVE})
        self.assertTrue(qwen_only.engine_ready("qwen"))
        self.assertFalse(qwen_only.engine_ready("moss"))
        self.assertTrue(self.c.engine_ready("moss"))
        self.assertFalse(self.c.engine_ready("qwen"))


class BothEnginesOnDisk(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="sb-disk-"))
        self.cfg = split_cfg(self.root)
        for repo in ("Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                     "OpenMOSS-Team/MOSS-TTS-Local-Transformer"):
            eid = bootstrap.engine_of(repo)
            d = bootstrap.model_dir(
                bootstrap.engine_models_dir(self.cfg, eid), repo, eid)
            d.mkdir(parents=True)
            (d / "config.json").write_text("{}")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_both_layouts_are_listed(self):
        rows = {r["repo"]: r["engine"] for r in manager.local_models(self.cfg)}
        # Walking the nested shape over a flat folder lists nothing, which is
        # how a downloaded MOSS model reads as never downloaded.
        self.assertEqual(rows.get("OpenMOSS-Team/MOSS-TTS-Local-Transformer"),
                         "moss")
        self.assertEqual(rows.get("Qwen/Qwen3-TTS-12Hz-0.6B-Base"), "qwen")

    def test_installed_is_checked_in_the_right_place(self):
        moss = bootstrap.engine_models_dir(self.cfg, "moss")
        self.assertTrue(bootstrap.model_installed(
            moss, "OpenMOSS-Team/MOSS-TTS-Local-Transformer"))
        self.assertFalse(bootstrap.model_installed(
            moss, "OpenMOSS-Team/MOSS-VoiceGenerator"))

    def test_each_engine_keeps_its_models_in_its_own_install(self):
        qwen = bootstrap.engine_models_dir(self.cfg, "qwen")
        moss = bootstrap.engine_models_dir(self.cfg, "moss")
        self.assertNotEqual(qwen, moss)
        # Neither can see the other's: separate ComfyUIs, separate folders.
        self.assertFalse(bootstrap.model_installed(
            qwen, "OpenMOSS-Team/MOSS-TTS-Local-Transformer"))

    def test_deleting_a_moss_folder_stays_inside_its_own_root(self):
        manager.delete_model(self.cfg, "OpenMOSS-Team/MOSS-TTS-Local-Transformer")
        self.assertFalse(bootstrap.model_dir(
            bootstrap.engine_models_dir(self.cfg, "moss"),
            "OpenMOSS-Team/MOSS-TTS-Local-Transformer", "moss").exists())
        for bad in ("../../etc", "OpenMOSS-Team/../../../etc", "nope"):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                manager.delete_model(self.cfg, bad)

    def test_the_models_page_lists_both_engines(self):
        rows = manager.curated(self.cfg)
        self.assertEqual({r["engine"] for r in rows}, {"qwen", "moss"})
        moss = [r for r in rows
                if r["repo"] == "OpenMOSS-Team/MOSS-TTS-Local-Transformer"][0]
        self.assertEqual(moss["role"], "required")
        self.assertTrue(moss["installed"])
        self.assertEqual(moss["engine_label"], "MOSS-TTS")


class VramGuard(unittest.TestCase):
    """A model that cannot be held by the card is worth saying so about before
    the download, not after ComfyUI runs out of memory mid-take."""

    def test_the_card_reports_its_memory(self):
        done = subprocess.CompletedProcess(
            [], 0, "NVIDIA GeForce RTX 4060, 8188\n", "")
        with mock.patch.object(bootstrap, "_run", return_value=done):
            gpu = bootstrap.nvidia_gpu(refresh=True)
        self.assertEqual(gpu["name"], "NVIDIA GeForce RTX 4060")
        self.assertEqual(gpu["vram_mb"], 8188)
        bootstrap._GPU.clear()

    def test_a_card_that_answers_without_a_number_is_still_a_card(self):
        done = subprocess.CompletedProcess([], 0, "NVIDIA GeForce RTX 4060\n", "")
        with mock.patch.object(bootstrap, "_run", return_value=done):
            gpu = bootstrap.nvidia_gpu(refresh=True)
        self.assertEqual(gpu["name"], "NVIDIA GeForce RTX 4060")
        self.assertEqual(gpu["vram_mb"], 0)
        bootstrap._GPU.clear()

    def test_an_8gb_card_holds_the_1_7b_models_and_not_the_8b(self):
        eight = 8188
        for repo, want in (("OpenMOSS-Team/MOSS-TTS-Local-Transformer", True),
                           ("OpenMOSS-Team/MOSS-VoiceGenerator", True),
                           ("OpenMOSS-Team/MOSS-TTS", False)):
            entry = next(m for m in bootstrap.MOSS_MODEL_REPOS
                         if m["repo"] == repo)
            with self.subTest(repo=repo):
                self.assertIs(bootstrap.fits_vram(entry["vram_gb"], eight), want)

    def test_an_unknown_card_is_never_treated_as_a_small_one(self):
        # nvidia-smi missing is rule 5b's failure, and answering "will not fit"
        # there would hide every model from someone who has the memory.
        self.assertIsNone(bootstrap.fits_vram(18, 0))
        rows = manager.curated({"models_dir": ""}, vram_mb=0)
        self.assertTrue(all(r["fits"] is None for r in rows))

    def test_the_models_page_marks_what_will_not_load(self):
        rows = {r["repo"]: r for r in manager.curated({"models_dir": ""},
                                                      vram_mb=8188)}
        self.assertFalse(rows["OpenMOSS-Team/MOSS-TTS"]["fits"])
        self.assertTrue(rows["OpenMOSS-Team/MOSS-VoiceGenerator"]["fits"])
        self.assertTrue(rows["Qwen/Qwen3-TTS-12Hz-0.6B-Base"]["fits"])

    def test_every_model_carries_a_figure_to_judge_it_by(self):
        for eng in bootstrap.ENGINES.values():
            for m in eng["models"]:
                with self.subTest(repo=m["repo"]):
                    self.assertIsInstance(m.get("vram_gb"), int)
                    self.assertGreater(m["vram_gb"], 0)

    def test_comfyui_answers_for_the_card_when_nvidia_smi_cannot(self):
        # A portable ComfyUI carries its own CUDA and knows the card on a
        # machine where nvidia-smi is not on PATH.
        c = client_for({})
        payload = {"devices": [{"name": "cuda:0", "vram_total": 8588886016}]}

        class Reply:
            status_code = 200

            @staticmethod
            def raise_for_status():
                pass

            @staticmethod
            def json():
                return payload

        with mock.patch.object(comfy.requests, "get", return_value=Reply):
            self.assertEqual(c.vram_mb(), 8191)

    def test_an_engine_that_will_not_answer_reports_no_memory(self):
        c = client_for({})
        with mock.patch.object(comfy.requests, "get",
                               side_effect=comfy.requests.ConnectionError()):
            self.assertEqual(c.vram_mb(), 0)



class Moss8bPrerequisites(unittest.TestCase):
    """The quantized 8B is never assumed to be sitting on HuggingFace waiting,
    and two of its prerequisites cannot be downloaded at all."""

    def test_two_steps_are_not_downloadable_and_say_so(self):
        steps = {s["id"]: s for s in bootstrap.GGUF_STEPS}
        # llama.cpp is compiled from source; the TensorRT engines are built
        # against the card in front of you. A first run that promised to fetch
        # its way to a working 8B would be lying.
        self.assertFalse(steps["toolchain"]["obtainable"])
        self.assertFalse(steps["engines"]["obtainable"])
        self.assertTrue(steps["weights"]["obtainable"])

    def test_the_repos_are_looked_up_not_taken_on_trust(self):
        seen = []

        def tree(_cfg, repo, *a, **kw):
            seen.append(repo)
            return [{"path": "MOSS_TTS_Q4_K_M.gguf", "size": 1}]

        with mock.patch.object(bootstrap, "hf_tree", tree):
            out = bootstrap.gguf_available({})
        self.assertEqual(sorted(seen), sorted([bootstrap.GGUF_REPO,
                                               bootstrap.GGUF_TOKENIZER_REPO]))
        self.assertTrue(out["ready"])

    def test_a_repo_that_is_not_there_is_reported_not_guessed(self):
        with mock.patch.object(bootstrap, "hf_tree",
                               side_effect=RuntimeError("404 not found")):
            out = bootstrap.gguf_available({})
        self.assertFalse(out["ready"])
        for repo in out["repos"].values():
            self.assertFalse(repo["found"])
            self.assertIn("404", repo["why"])

    def test_the_8b_is_not_in_the_default_download_set(self):
        # Whatever the llama.cpp path can do, first launch fetches nothing it
        # cannot then load through the engine it actually ships with.
        repos = [m["repo"] for m in
                 bootstrap.wanted_models(dict(bootstrap.DEFAULT_CONFIG))]
        self.assertNotIn("OpenMOSS-Team/MOSS-TTS", repos)
        self.assertNotIn(bootstrap.GGUF_REPO, repos)



class SelfTestSteps(unittest.TestCase):
    """A self-test that only ever reports success is worth nothing, so each
    way it can stop has its own check."""

    class Engine:
        def __init__(self, ready=True, speakers=("Eric",)):
            self.ready, self._speakers = ready, list(speakers)

        def schema(self, force=False):
            return {}

        def engine_ready(self, _engine):
            return self.ready

        def speakers(self):
            return self._speakers

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="sb-self-"))
        self.cfg = split_cfg(self.root, want_clone=False,
                             want_moss_design=False)
        self.models = bootstrap.engine_models_dir(self.cfg, "moss")
        self.task = manager.Task("selftest", "Test")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _seed(self, engine, weights=True):
        root = bootstrap.engine_models_dir(self.cfg, engine)
        for m in bootstrap.wanted_models(self.cfg, engine):
            d = bootstrap.model_dir(root, m["repo"], engine)
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text("{}")
            if weights:
                (d / "model.safetensors").write_bytes(b"\0" * 32)

    def _run(self, engine="moss", online=True, client=None):
        with mock.patch.object(bootstrap, "comfy_online", return_value=online):
            try:
                manager.selftest(self.cfg, client or self.Engine(), engine,
                                 self.task)
            except Exception:  # noqa: BLE001
                pass
        return {s["id"]: s for s in self.task.meta.get("steps", [])}

    def test_a_dead_engine_stops_at_the_first_step(self):
        steps = self._run(online=False)
        self.assertEqual(steps["engine"]["state"], "fail")
        self.assertNotIn("nodes", steps)

    def test_nodes_that_are_not_loaded_stop_before_the_models(self):
        steps = self._run(client=self.Engine(ready=False))
        self.assertEqual(steps["engine"]["state"], "ok")
        self.assertEqual(steps["nodes"]["state"], "fail")
        self.assertNotIn("models", steps)

    def test_the_node_list_is_read_fresh_not_from_the_cache(self):
        # The schema is cached for two minutes, and the reason anyone presses
        # Test is usually that something just changed. Reading the cache once
        # reported "nodes are loaded" about a ComfyUI that had just been shown
        # not to have them.
        forced = []

        class Watcher(self.Engine):
            def schema(self, force=False):
                forced.append(force)
                return {}

        self._seed("moss")
        self._run(client=Watcher())
        self.assertIn(True, forced)

    def test_a_folder_with_no_weights_in_it_is_caught(self):
        self._seed("moss", weights=False)
        steps = self._run()
        self.assertEqual(steps["models"]["state"], "fail")
        self.assertIn("no weights", steps["models"]["detail"])
        self.assertNotIn("graph", steps)

    def test_a_missing_folder_is_named(self):
        steps = self._run()
        self.assertEqual(steps["models"]["state"], "fail")
        self.assertIn("MOSS-TTS-Local-Transformer", steps["models"]["detail"])

    def test_silence_counts_as_a_failure(self):
        # The right number of frames, all zeros: it decodes perfectly and
        # plays nothing, which is what a model that generated nothing sounds
        # like.
        self.assertEqual(manager._peak(b"\x00\x00" * 200, 2), 0.0)
        loud = struct.pack("<h", 20000) * 200
        self.assertGreater(manager._peak(loud, 2), 0.5)

    def test_a_format_it_cannot_measure_is_not_called_silent(self):
        # 24-bit or float audio is not a failure, it is simply not something
        # this check can weigh — and -1 keeps it out of the silence branch.
        self.assertEqual(manager._peak(b"\x00\x00\x00" * 60, 3), -1.0)
        self.assertEqual(manager._peak(b"", 2), -1.0)

    def test_weight_suffixes_cover_both_engines_shapes(self):
        for suffix in (".safetensors", ".bin", ".gguf", ".onnx", ".npy"):
            with self.subTest(suffix=suffix):
                self.assertIn(suffix, manager.WEIGHT_SUFFIXES)



class SeparateInstalls(unittest.TestCase):
    """Each engine gets its own ComfyUI, environment, models folder and port,
    because that is the only way one engine's packages cannot break the
    other's."""

    def test_nothing_below_comfyui_is_shared(self):
        seen = {}
        for eid, eng in bootstrap.ENGINES.items():
            comfy = Path("/app") / eng["dir_name"]
            seen[eid] = {
                "comfy": comfy,
                "venv": bootstrap.venv_python(comfy).parents[1],
                "nodes": comfy / "custom_nodes" / eng["node_dir"],
                "models": comfy / "models" / eng["subdir"],
                "port": eng["port"],
            }
        a, b = seen["qwen"], seen["moss"]
        for key in a:
            with self.subTest(part=key):
                self.assertNotEqual(a[key], b[key])

    def test_the_two_environments_do_not_collide(self):
        # They sit under the same parent, so a bare "comfy-venv" beside both
        # would be one environment shared by both — the thing the layout is
        # for. The name has to carry the install.
        q = bootstrap.venv_python(Path("/app/ComfyUI-Qwen3-TTS"))
        m = bootstrap.venv_python(Path("/app/ComfyUI-MOSS-TTS"))
        self.assertNotEqual(q.parents[1], m.parents[1])
        self.assertIn("Qwen", str(q))
        self.assertIn("MOSS", str(m))

    def test_a_single_install_config_becomes_qwen_s(self):
        # Every install before the split had one ComfyUI with both node packs
        # in it, and its settings at the top level. Those are Qwen's now; MOSS
        # starts from defaults, which means its own install to fetch.
        old = dict(bootstrap.DEFAULT_CONFIG,
                   comfy_url="http://127.0.0.1:9000",
                   comfy_dir="/somewhere/ComfyUI",
                   models_dir="/somewhere/ComfyUI/models",
                   python="/somewhere/py", setup_complete=True)
        bootstrap._migrate(old)
        self.assertEqual(old["engines"]["qwen"]["comfy_dir"],
                         "/somewhere/ComfyUI")
        self.assertEqual(old["engines"]["qwen"]["comfy_url"],
                         "http://127.0.0.1:9000")
        self.assertEqual(old["engines"]["moss"]["comfy_dir"], "")
        self.assertEqual(old["engines"]["moss"]["comfy_url"],
                         "http://127.0.0.1:8189")

    def test_a_fresh_install_claims_neither(self):
        fresh = dict(bootstrap.DEFAULT_CONFIG)
        bootstrap._migrate(fresh)
        self.assertEqual(fresh["engines"]["qwen"]["comfy_dir"], "")
        self.assertEqual(fresh["engines"]["moss"]["comfy_dir"], "")

    def test_migrating_twice_changes_nothing(self):
        cfg = dict(bootstrap.DEFAULT_CONFIG, comfy_dir="/a/ComfyUI",
                   setup_complete=True)
        bootstrap._migrate(cfg)
        first = json.dumps(cfg["engines"], sort_keys=True)
        bootstrap._migrate(cfg)
        self.assertEqual(json.dumps(cfg["engines"], sort_keys=True), first)

    def test_a_missing_port_is_healed_not_inherited(self):
        cfg = dict(bootstrap.DEFAULT_CONFIG)
        cfg["engines"] = {"qwen": {}, "moss": {}}
        self.assertEqual(bootstrap.engine_cfg(cfg, "moss")["comfy_url"],
                         "http://127.0.0.1:8189")

    def test_each_engine_looks_for_models_in_its_own_comfyui(self):
        cfg = dict(bootstrap.DEFAULT_CONFIG)
        cfg["engines"] = {
            "qwen": dict(bootstrap.engine_defaults("qwen"),
                         comfy_dir="/a/ComfyUI-Qwen3-TTS"),
            "moss": dict(bootstrap.engine_defaults("moss"),
                         comfy_dir="/a/ComfyUI-MOSS-TTS")}
        self.assertEqual(bootstrap.engine_models_dir(cfg, "qwen"),
                         Path("/a/ComfyUI-Qwen3-TTS/models"))
        self.assertEqual(bootstrap.engine_models_dir(cfg, "moss"),
                         Path("/a/ComfyUI-MOSS-TTS/models"))

    def test_running_both_at_once_is_off_by_default(self):
        # Two ComfyUIs that have both generated each hold their models in
        # their own VRAM, and unload_all_models only reaches inside one
        # process. On 8 GB the second engine is the one that fails to
        # allocate, so the default is one at a time.
        self.assertFalse(bootstrap.DEFAULT_CONFIG["run_both_engines"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
