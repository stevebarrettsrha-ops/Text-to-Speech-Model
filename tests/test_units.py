"""Unit tests for the parts that are easy to get quietly wrong.

Standard library only — `python -m unittest discover tests` needs nothing that
requirements.txt does not already install.

Every test here stands for a fault that actually shipped at some point, so the
names say what would break rather than what the function is called.
"""
from __future__ import annotations

import copy
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import subprocess
import threading
import time
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
        (self.models / "qwen-tts" / "Real").mkdir(parents=True)
        (self.models / "qwen-tts" / "Real" / "w.safetensors").write_text("x")
        (self.models / "qwen-tts" / "voices").mkdir()
        (self.models / "qwen-tts" / "voices" / "mine.wav").write_text("x")
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
        link = self.models / "qwen-tts" / "Escape"
        link.symlink_to(self.outside, target_is_directory=True)
        with self.assertRaises(Exception):
            manager.delete_model(self.cfg, "Qwen/Escape")
        self.assertTrue((self.outside / "f.txt").exists())

    def test_a_real_delete_still_works(self):
        manager.delete_model(self.cfg, "Qwen/Real")
        self.assertFalse((self.models / "qwen-tts" / "Real").exists())
        self.assertTrue(self.precious.exists())

    def test_the_root_and_the_nodes_saved_voices_are_not_models(self):
        # With no org folder in between, "Qwen/" names models/qwen-tts itself
        # and "Qwen/voices" the node's saved voices. Neither is a model, and
        # a delete of either would take every model or every voice with it.
        for repo in ("Qwen/", "Qwen/voices", "Anyone/voices"):
            with self.subTest(repo=repo):
                with self.assertRaises(Exception):
                    manager.delete_model(self.cfg, repo)
        self.assertTrue((self.models / "qwen-tts" / "voices" / "mine.wav")
                        .exists())
        self.assertTrue((self.models / "qwen-tts" / "Real").exists())


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

    def test_a_config_whose_weights_never_came_is_not_installed(self):
        # The weights request failed before its .part was opened — a 503, a
        # DNS blip — and the config alone counted as installed, so setup
        # never fetched the model again.
        d = self._folder("Qwen/C")
        (d / "config.json").write_text("{}")
        self.assertFalse(bootstrap.model_installed(self.models, "Qwen/C"))


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

    CLONE_SCHEMA = {"VoiceCloneNode": {"input": {
        "required": {"target_text": ["STRING", {"default": ""}],
                     "model_choice": [["0.6B", "1.7B"], {"default": "0.6B"}]},
        "optional": {"ref_audio": ["AUDIO"],
                     "ref_text": ["STRING", {"default": ""}],
                     "x_vector_only": ["BOOLEAN", {"default": False}]}}},
        "LoadAudio": {"input": {"required": {
            "audio": [["ref.wav"], {"audio_upload": True}]}}}, **SAVE}

    def test_a_clone_with_no_transcript_clones_from_the_sound_alone(self):
        # The node's default mode refuses a line outright without the words
        # spoken in the clip — "ref_text is required when
        # x_vector_only_mode=False" — and the page never said the box was
        # required, so every clone left blank failed.
        for blank in ("", "   ", None):
            with self.subTest(ref_text=blank):
                ins = client_for(self.CLONE_SCHEMA).build_line(
                    {"text": "hi"},
                    {"kind": "clone", "ref_audio": "ref.wav",
                     "ref_text": blank}, OPTS)["prompt"]["2"]["inputs"]
                self.assertIs(ins["x_vector_only"], True)

    def test_a_clone_with_a_transcript_keeps_the_closer_copy(self):
        ins = client_for(self.CLONE_SCHEMA).build_line(
            {"text": "hi"},
            {"kind": "clone", "ref_audio": "ref.wav", "ref_text": " spoken "},
            OPTS)["prompt"]["2"]["inputs"]
        self.assertIs(ins["x_vector_only"], False)
        self.assertEqual(ins["ref_text"], "spoken")

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


class OneModelLoadPerTake(unittest.TestCase):
    """Both node packs hold one checkpoint at a time, so on an 8 GB card the
    order lines are spoken in decides how many times a model is read from
    disk, and the unload switch decides whether it happens on every line."""

    class Recorder:
        """Stands in for a ComfyClient: builds nothing, remembers everything."""

        def __init__(self, root: Path):
            self.root, self.calls, self.n = root, [], 0

        def build_line(self, line, voice, opts):
            self.calls.append({"text": line["text"],
                               "weights": comfy.ComfyClient.line_weights(
                                   voice, opts),
                               "unload": opts["unload"]})
            return {"prompt": {}}

        def queue(self, prompt):
            self.n += 1
            return f"p{self.n}"

        def result(self, prompt_id):
            clip = make_clip(self.root / f"{prompt_id}.wav")
            return [{"filename": clip.name}], None

        def view(self, item):
            resp = mock.MagicMock()
            resp.__enter__.return_value = resp
            resp.iter_content.return_value = [
                (self.root / item["filename"]).read_bytes()]
            return resp

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.client = self.Recorder(self.dir)
        for target, value in (("for_engine", lambda *_: self.client),
                              ("TAKES_DIR", self.dir / "takes"),
                              ("TAKES_PATH", self.dir / "takes.json")):
            patcher = mock.patch.object(server, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_take(self, speakers, lines, **extra):
        job = f"job{len(server.jobs)}"
        server.jobs[job] = {"status": "running"}
        self.addCleanup(server.jobs.pop, job, None)
        server.run_job(job, dict({"engine": "qwen", "model": "0.6B",
                                  "pause": 0.1, "speakers": speakers,
                                  "lines": [{"speaker": k, "text": t}
                                            for k, t in lines]}, **extra))
        return server.jobs[job]

    DIALOGUE = [("1", "a1"), ("2", "b1"), ("1", "a2"), ("2", "b2"),
                ("1", "a3")]
    MIXED = {"1": {"name": "Ann", "kind": "preset", "speaker": "Aiden"},
             "2": {"name": "Bo", "kind": "clone", "ref_audio": "bo.wav"}}

    def test_a_preset_answering_a_clone_loads_each_model_once(self):
        # Spoken in script order this swapped CustomVoice for Base on every
        # line: five lines, five loads from disk.
        job = self.run_take(self.MIXED, self.DIALOGUE)
        self.assertEqual(job["status"], "done", job.get("error"))
        weights = [c["weights"] for c in self.client.calls]
        swaps = sum(1 for a, b in zip(weights, weights[1:]) if a != b)
        self.assertEqual(swaps, 1)
        self.assertEqual([c["text"] for c in self.client.calls],
                         ["a1", "a2", "a3", "b1", "b2"])

    def test_the_take_is_still_joined_in_script_order(self):
        job = self.run_take(self.MIXED, self.DIALOGUE)
        lines = job["take"]["lines"]
        self.assertEqual([ln["text"] for ln in lines],
                         [t for _, t in self.DIALOGUE])
        self.assertEqual([ln["index"] for ln in lines], list(range(5)))
        self.assertEqual(lines[1]["file"], "line_001.wav")

    def test_free_memory_is_asked_for_once_after_the_last_line(self):
        # Sent with every line it unloaded the model after each one, and the
        # next line read it back from disk.
        self.run_take(self.MIXED, self.DIALOGUE, unload=True)
        self.assertEqual([c["unload"] for c in self.client.calls],
                         [False] * 4 + [True])

    def test_with_the_switch_off_nothing_is_unloaded(self):
        self.run_take(self.MIXED, self.DIALOGUE, unload=False)
        self.assertFalse(any(c["unload"] for c in self.client.calls))

    def test_one_voice_keeps_script_order(self):
        both = {"1": self.MIXED["1"], "2": dict(self.MIXED["1"], name="Cy")}
        self.run_take(both, self.DIALOGUE)
        self.assertEqual([c["text"] for c in self.client.calls],
                         [t for _, t in self.DIALOGUE])

    def test_the_weights_follow_what_each_builder_loads(self):
        w = comfy.ComfyClient.line_weights
        qwen = {"engine": "qwen", "model": "0.6B"}
        self.assertNotEqual(w({"kind": "preset"}, qwen),
                            w({"kind": "clone"}, qwen))
        # A designed voice is always the 1.7B, whatever the picker says.
        self.assertEqual(w({"kind": "design"}, qwen),
                         w({"kind": "design"}, dict(qwen, model="1.7B")))
        moss = {"engine": "moss", "moss_model": comfy.MOSS_DEFAULT_MODEL}
        # MOSS clones and speaks in its own voice on one loader.
        self.assertEqual(w({"kind": "preset"}, moss),
                         w({"kind": "clone"}, moss))
        self.assertEqual(w({"kind": "design"}, moss),
                         ("moss", comfy.MOSS_VOICE_GENERATOR))


class TheNamesComfyUIKnowsTheNodesBy(unittest.TestCase):
    """ComfyUI registers a node under its NODE_CLASS_MAPPINGS key, and the
    Qwen pack keys them FB_Qwen3TTS*. Asking for the Python class names found
    no Qwen node on any real install, so the engine never read as ready."""

    REAL = {"FB_Qwen3TTSCustomVoice": custom_voice(), **SAVE}

    def test_the_registered_name_is_found(self):
        c = client_for(self.REAL)
        self.assertTrue(c.engine_ready("qwen"))
        self.assertEqual(c.speakers(), ["Aiden", "Serena"])
        self.assertTrue(c.capabilities("qwen")["preset"])

    def test_the_graph_names_the_node_as_ComfyUI_registered_it(self):
        g = client_for(self.REAL).build_line(
            {"text": "hi"}, {"kind": "preset", "speaker": "Aiden"}, OPTS)
        self.assertEqual(g["prompt"]["2"]["class_type"],
                         "FB_Qwen3TTSCustomVoice")

    def test_an_older_pack_s_names_still_work(self):
        for name in ("Qwen3TTSCustomVoice", "CustomVoiceNode"):
            c = client_for({name: custom_voice(), **SAVE})
            self.assertTrue(c.engine_ready("qwen"), name)
            g = c.build_line({"text": "hi"}, {"kind": "preset"}, OPTS)
            self.assertEqual(g["prompt"]["2"]["class_type"], name)

    def test_the_stand_in_uses_the_real_names(self):
        # The suite passed for months against a mock that used the class
        # names, which is how the fault shipped.
        src = (REPO / "tests" / "mock_comfy.py").read_text(encoding="utf-8")
        self.assertIn('"FB_Qwen3TTSCustomVoice": _node(', src)
        self.assertNotIn('"CustomVoiceNode": _node(', src)


class PipesAreUtf8(unittest.TestCase):
    """Windows hands a piped Python the ANSI code page with strict errors, and
    the Qwen pack prints an emoji as it imports: IMPORT FAILED, but only when
    this app started ComfyUI."""

    def test_a_child_on_a_cp1252_console_can_still_print_an_emoji(self):
        env = dict(os.environ, PYTHONIOENCODING="cp1252")
        env.pop("PYTHONUTF8", None)
        out = bootstrap._run([sys.executable, "-c",
                              "print('\u2705 ComfyUI-Qwen-TTS loaded')"],
                             env=env, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("\u2705", out.stdout)

    def test_the_engine_is_launched_with_utf8(self):
        env = bootstrap.py_env({"PYTHONIOENCODING": "cp1252"})
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(env["PYTHONUTF8"], "1")


class TakesJoinFromFlac(unittest.TestCase):
    """Current ComfyUI's SaveAudioAdvanced offers flac, mp3 and opus as a
    DynamicCombo and never wav, so every take used to arrive as a zip."""

    V3_SAVE = {"SaveAudioAdvanced": {"input": {"required": {
        "audio": ["AUDIO"],
        "filename_prefix": ["STRING", {"default": "audio/ComfyUI"}],
        "format": ["COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "flac", "inputs": {}},
            {"key": "mp3", "inputs": {"required": {"quality": [
                ["V0", "128k", "320k"], {"default": "V0"}]}}},
            {"key": "opus", "inputs": {}}]}]}}}}

    def test_a_dynamic_combo_is_read_as_its_keys(self):
        c = client_for(self.V3_SAVE)
        self.assertEqual(c._enum("SaveAudioAdvanced", "format"),
                         ["flac", "mp3", "opus"])
        self.assertEqual(c.save_node(), ("SaveAudioAdvanced", "flac"))

    def test_lossless_is_picked_over_whatever_comes_first(self):
        schema = copy.deepcopy(self.V3_SAVE)
        opts = schema["SaveAudioAdvanced"]["input"]["required"]["format"][1]
        opts["options"].reverse()
        self.assertEqual(client_for(schema).save_node()[1], "flac")

    def test_no_interpreter_leaves_the_clips_for_the_zip(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        clip = d / "line_000.flac"
        clip.write_bytes(b"fLaC")
        with mock.patch.object(bootstrap, "comfy_python", lambda *_: ""):
            self.assertEqual(server.to_wav([clip], "qwen"), [clip])
        self.assertTrue(clip.exists())

    def test_a_failed_conversion_leaves_nothing_half_done(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        clip = d / "line_000.flac"
        clip.write_bytes(b"not audio")
        with mock.patch.object(bootstrap, "comfy_python",
                               lambda *_: sys.executable):
            self.assertEqual(server.to_wav([clip], "qwen"), [clip])
        self.assertEqual(sorted(p.name for p in d.iterdir()),
                         ["line_000.flac"])

    def test_flac_as_ComfyUI_writes_it_comes_back_frame_exact(self):
        try:
            import av  # noqa: F401
            import numpy as np
        except ImportError:
            self.skipTest("PyAV is ComfyUI's, not this app's")
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        clips = []
        for i, (rate, chans) in enumerate([(24000, 1), (24000, 1)]):
            # AudioSaveHelper.save_audio, as ComfyUI does it: float frames in.
            wav = np.sin(np.arange(rate) * 0.05)[None].repeat(chans, 0) * 0.5
            buf = io.BytesIO()
            with av.open(buf, mode="w", format="flac") as out:
                stream = out.add_stream("flac", rate=rate, layout="mono")
                frame = av.AudioFrame.from_ndarray(
                    wav.T.reshape(1, -1).astype(np.float32), format="flt",
                    layout="mono")
                frame.sample_rate, frame.pts = rate, 0
                out.mux(stream.encode(frame))
                out.mux(stream.encode(None))
            clip = d / f"line_{i:03d}.flac"
            clip.write_bytes(buf.getvalue())
            clips.append(clip)
        with mock.patch.object(bootstrap, "comfy_python",
                               lambda *_: sys.executable):
            wavs = server.to_wav(clips, "qwen")
        self.assertEqual([w.suffix for w in wavs], [".wav", ".wav"])
        self.assertFalse(any(c.exists() for c in clips))
        with wave.open(str(wavs[0]), "rb") as w:
            self.assertEqual((w.getnchannels(), w.getframerate(),
                              w.getnframes()), (1, 24000, 24000))
        self.assertTrue(server.stitch_wavs(wavs, d / "take.wav", 0.5))


class JobsKeepToThemselves(unittest.TestCase):
    """A take, a Stop and an engine switch each touched more than their own."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.client = mock.MagicMock()
        for target, value in (("for_engine", lambda *_: self.client),
                              ("REFS_DIR", self.dir / "refs"),
                              ("engine_online", lambda *_: True)):
            patcher = mock.patch.object(server, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.saved_jobs = dict(server.jobs)
        server.jobs.clear()
        self.addCleanup(lambda: (server.jobs.clear(),
                                 server.jobs.update(self.saved_jobs)))
        self.app = server.app.test_client()

    def test_stop_after_a_take_has_finished_interrupts_nothing(self):
        # The player's Stop sends the last job's id, and this used to stop
        # whatever the engine was doing — a self-test, a preview — regardless.
        server.jobs["old"] = {"id": "old", "status": "done", "engine": "qwen",
                              "prompt_id": "p1"}
        r = self.app.post("/api/jobs/old/cancel")
        self.assertFalse(r.get_json()["running"])
        self.client.interrupt.assert_not_called()

    def test_stop_on_a_running_take_interrupts_its_own_prompt(self):
        server.jobs["now"] = {"id": "now", "status": "running",
                              "engine": "moss", "prompt_id": "p9"}
        self.app.post("/api/jobs/now/cancel")
        self.client.interrupt.assert_called_once_with("p9")
        self.assertTrue(server.jobs["now"]["cancelled"])

    def test_switching_engines_mid_take_leaves_the_take_s_engine_running(self):
        server.jobs["t"] = {"id": "t", "status": "running", "engine": "qwen",
                            "title": "Chapter one"}
        with mock.patch.dict(server.cfg, {"run_both_engines": False}), \
                mock.patch.object(server.PROCS["qwen"], "stop") as stop:
            why = server.activate("moss")
        self.assertIn("Chapter one", why)
        stop.assert_not_called()

    def test_with_room_for_both_a_take_does_not_block_a_switch(self):
        server.jobs["t"] = {"id": "t", "status": "running", "engine": "qwen"}
        with mock.patch.dict(server.cfg, {"run_both_engines": True}):
            self.assertEqual(server.busy_elsewhere("moss"), "")

    def test_a_reference_is_kept_by_its_contents(self):
        # Uploaded by file name with overwrite on, two speakers' own
        # "recording.wav" became one voice.
        names = []
        for body in (b"RIFF-one", b"RIFF-two"):
            r = self.app.post("/api/upload-reference", data={
                "file": (io.BytesIO(body), "recording.wav")},
                content_type="multipart/form-data")
            names.append(r.get_json()["name"])
        self.assertNotEqual(names[0], names[1])
        self.assertTrue(all(n.endswith(".wav") for n in names))
        self.assertTrue((self.dir / "refs" / names[0]).is_file())

    def test_the_engine_that_speaks_the_line_is_sent_the_clip(self):
        # Each engine is its own ComfyUI with its own input folder: a clip
        # uploaded while Qwen was showing did not exist for MOSS.
        r = self.app.post("/api/upload-reference", data={
            "file": (io.BytesIO(b"RIFF-voice"), "me.wav")},
            content_type="multipart/form-data")
        name = r.get_json()["name"]
        self.client.reset_mock()
        server.ensure_reference("moss", name)
        self.client.upload_bytes.assert_called_once()
        self.assertEqual(self.client.upload_bytes.call_args[0][:2],
                         (name, b"RIFF-voice"))

    def test_a_pause_of_nothing_is_nothing(self):
        clip = self.dir / "c.wav"
        make_clip(clip)
        self.client.build_line.return_value = {"prompt": {}}
        self.client.queue.return_value = "p"
        self.client.result.return_value = ([{"filename": "c.wav"}], None)
        resp = mock.MagicMock()
        resp.__enter__.return_value = resp
        resp.iter_content.return_value = [clip.read_bytes()]
        self.client.view.return_value = resp
        server.jobs["z"] = {"id": "z", "status": "running"}
        with mock.patch.object(server, "TAKES_DIR", self.dir / "takes"), \
                mock.patch.object(server, "TAKES_PATH",
                                  self.dir / "takes.json"):
            server.run_job("z", {"engine": "qwen", "pause": 0,
                                 "speakers": {"1": {"kind": "preset"}},
                                 "lines": [{"speaker": 1, "text": "a"},
                                           {"speaker": 1, "text": "b"}]})
        take = server.jobs["z"]["take"]
        self.assertEqual(take["pause"], 0.0)
        with wave.open(str(self.dir / "takes" / take["id"] / "take.wav")) as w:
            self.assertEqual(w.getnframes(), 4000)


class EngineHousekeeping(unittest.TestCase):
    """Small readings of the engine that were each wrong on a real machine."""

    def test_a_relative_main_py_names_no_folder(self):
        # Started as "python main.py" from its own folder, or "ComfyUI\\main.py"
        # by a portable .bat: resolved against this app's folder, either named
        # a ComfyUI that does not exist, and our own engine read as foreign.
        self.assertEqual(comfy.root_from_argv(["main.py", "--port", "8188"]),
                         "")
        self.assertEqual(comfy.root_from_argv(["ComfyUI\\main.py"]), "")
        self.assertEqual(comfy.root_from_argv(["/opt/ComfyUI/main.py"]),
                         "/opt/ComfyUI")
        self.assertEqual(comfy.root_from_argv(["D:\\AI\\ComfyUI\\main.py"]),
                         "D:\\AI\\ComfyUI")

    def test_this_app_s_own_environment_is_never_an_engine_s(self):
        # A managed install sits beside the launcher, so <parent>/.venv is
        # Script Builder's Flask venv — which won whenever the engine's own
        # environment had no torch in it.
        comfy_dir = bootstrap.APP_DIR / "ComfyUI-Qwen3-TTS"
        cands = [bootstrap._env_root(c)
                 for c in bootstrap._interpreters(comfy_dir)]
        self.assertNotIn((bootstrap.APP_DIR / ".venv").resolve(), cands)
        self.assertIn(bootstrap._env_root(bootstrap.venv_python(comfy_dir)),
                      cands)

    def test_windows_reads_a_command_line_without_wmic(self):
        # WMIC is gone from Windows 11 25H2; an empty answer waved the
        # not-a-ComfyUI guard through.
        calls = []

        def run(cmd, **_):
            calls.append(cmd[0])
            return subprocess.CompletedProcess(
                cmd, 0, "C:\\py\\python.exe main.py --port 8188\r\n", "")
        with mock.patch.object(bootstrap.platform, "system",
                               return_value="Windows"), \
                mock.patch.object(bootstrap, "_run", side_effect=run):
            self.assertIn("main.py", bootstrap.pid_cmdline(42))
        self.assertEqual(calls, ["powershell"])

    def test_an_unreadable_command_line_is_not_closed(self):
        no_manager = mock.Mock(status_code=404)
        with mock.patch.object(server.requests, "post",
                               return_value=no_manager), \
                mock.patch.object(server, "comfy_online", return_value=True), \
                mock.patch.object(bootstrap, "port_pids", return_value=[77]), \
                mock.patch.object(bootstrap, "pid_cmdline", return_value=""), \
                mock.patch.object(bootstrap, "kill_pid") as kill, \
                mock.patch.object(server.time, "sleep"):
            how, advice = server.take_over_port("http://127.0.0.1:1", 1,
                                                "qwen")
        self.assertIsNone(how)
        self.assertIn("cannot be read", advice)
        kill.assert_not_called()

    def test_versions_compare_as_numbers(self):
        v = manager.version_tuple
        self.assertLess(v("4.9.0"), (4, 40))       # "4.9" >= "4.40" as text
        self.assertGreaterEqual(v("4.57.3"), (4, 40))
        self.assertEqual(v("5.0.0rc1"), (5, 0, 0))
        self.assertEqual(v("4.57.3.dev0")[:3], (4, 57, 3))

    def test_the_console_answers_since_a_mark_after_it_trims_itself(self):
        # The self-test sliced the buffer by its old length; once it trimmed,
        # nothing was "new", and a run that downloaded passed as offline.
        proc = bootstrap.ComfyProcess()
        for i in range(1500):
            proc.note(f"old {i}")
        mark = proc.written
        for i in range(599):
            proc.note(f"new {i}")
        proc.note("Downloading model.safetensors from huggingface")
        fresh = proc.since(mark)
        self.assertEqual(len(fresh), 600)
        self.assertIn("huggingface", fresh[-1])
        self.assertTrue(all("new" in l or "huggingface" in l for l in fresh))


class ALineThatSavedNothing(unittest.TestCase):
    """A prompt ComfyUI finished with no audio was waited on for the whole
    fifteen-minute timeout, because nothing was ever going to arrive."""

    def client_with(self, hist):
        c = comfy.ComfyClient()
        c.history = lambda _pid: hist
        return c

    def test_finished_without_audio_is_an_error(self):
        outs, err = self.client_with({
            "status": {"status_str": "success", "completed": True},
            "outputs": {}}).result("p")
        self.assertEqual(outs, [])
        self.assertIn("saved no audio", err)

    def test_still_running_is_not(self):
        self.assertEqual(self.client_with({
            "status": {"status_str": "running", "completed": False},
            "outputs": {}}).result("p"), ([], None))

    def test_the_node_s_own_error_comes_first(self):
        _, err = self.client_with({"status": {
            "status_str": "error", "completed": False,
            "messages": [["execution_error", {
                "node_type": "VoiceCloneNode",
                "exception_message": "CUDA out of memory"}]]}}).result("p")
        self.assertEqual(err, "VoiceCloneNode: CUDA out of memory")


class WhyTheNodesDidNotLoad(unittest.TestCase):
    """The probe exists to get the node pack's own exception out of ComfyUI's
    console, where nobody running from a launcher can read it. Reporting its
    own scaffolding instead is the one failure it must not have — and it did:
    a real install answered "ModuleNotFoundError: No module named
    'qwen_tts_probe'", which names nothing the person can act on."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="sb-probe-"))
        self.nodes = (self.root / "custom_nodes"
                      / bootstrap.ENGINES["qwen"]["node_dir"])
        self.nodes.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, init_body: str, extra: dict | None = None):
        (self.nodes / "__init__.py").write_text(init_body)
        for name, body in (extra or {}).items():
            (self.nodes / name).write_text(body)

    def test_a_pack_with_relative_imports_loads(self):
        # Every real node pack opens like this. Executed under a name that was
        # never registered in sys.modules, the relative import has no parent to
        # resolve through and dies naming the probe.
        self._write("from .nodes import NODE_CLASS_MAPPINGS\n",
                    {"nodes.py": "NODE_CLASS_MAPPINGS = {}\n"})
        self.assertEqual(
            bootstrap.node_import_error(sys.executable, self.root, "qwen"), "")

    def test_the_real_missing_package_is_what_comes_back(self):
        self._write("import transformers_that_is_not_here\n")
        why = bootstrap.node_import_error(sys.executable, self.root, "qwen")
        self.assertIn("transformers_that_is_not_here", why)
        self.assertNotIn("probe", why)

    def test_a_pack_that_raises_reports_its_own_exception(self):
        self._write('raise RuntimeError("torch is the CPU build")\n')
        why = bootstrap.node_import_error(sys.executable, self.root, "qwen")
        self.assertIn("torch is the CPU build", why)
        self.assertTrue(why.startswith("RuntimeError"), why)

    def test_nodes_that_are_not_there_say_so(self):
        shutil.rmtree(self.nodes)
        self.assertIn("not installed",
                      bootstrap.node_import_error(sys.executable, self.root,
                                                  "qwen"))


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

    def test_it_does_not_send_them_hunting_for_a_setting_already_set(self):
        # Reported from a panel whose picker read "Automatic — NVIDIA GeForce
        # RTX 4060 (CUDA build)" beside a row telling them to pick the NVIDIA
        # build. With a card present Automatic already is that build, so the
        # only move left is the button.
        with mock.patch.object(bootstrap, "nvidia_gpu",
                               return_value={"name": "NVIDIA GeForce RTX 4060",
                                             "driver": True}):
            text = manager.no_cuda_reason("2.14.0+cpu")
        self.assertIn("Press Reinstall", text)
        self.assertNotIn("above", text)

    def test_it_does_name_the_picker_when_automatic_would_not_do_it(self):
        # Where Automatic would not resolve to CUDA — no card visible to
        # nvidia-smi — the picker is the way out and has to be named.
        with mock.patch.object(bootstrap, "nvidia_gpu",
                               return_value={"name": "NVIDIA GeForce RTX 4060",
                                             "driver": True}), \
             mock.patch.object(bootstrap, "torch_index", return_value=""):
            text = manager.no_cuda_reason("2.14.0+cpu")
        self.assertIn("Pick the NVIDIA build above", text)


def torch_info(version: str, cuda: str | None = None) -> dict:
    """What bootstrap.installed_torch reports for a wheel."""
    return {"version": version, "cuda": cuda, "hip": None, "xpu": None}


RTX_4060 = {"name": "NVIDIA GeForce RTX 4060", "driver": True, "vram_mb": 8188}
NO_GPU = {"name": "", "driver": False, "vram_mb": 0}


class TorchReinstall(unittest.TestCase):
    """pip counts torch 2.14.0+cpu as satisfying `torch`, so Reinstall against
    the CUDA index changed nothing at all."""

    def _attempt(self, installed, index, code=0, said="", offered=True):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            if "index" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 0 if offered else 1, "",
                    "" if offered else "ERROR: No matching distribution "
                                       "found for torch")
            return subprocess.CompletedProcess(cmd, code, "", said)

        with mock.patch.object(bootstrap, "installed_torch",
                               return_value=installed), \
             mock.patch.object(bootstrap, "_run", side_effect=run):
            dropped = bootstrap.drop_mismatched_torch("py", index, lambda _m: None)
        return dropped, [c for c in calls if "index" not in c]

    def test_a_cpu_build_is_removed_before_the_cuda_one_lands(self):
        dropped, calls = self._attempt(torch_info("2.14.0+cpu"),
                                       bootstrap.CUDA_INDEX)
        self.assertTrue(dropped)
        self.assertTrue(any("uninstall" in c for c in calls))

    def test_a_matching_build_is_left_alone(self):
        dropped, calls = self._attempt(torch_info("2.14.0+cu128", "12.8"),
                                       bootstrap.CUDA_INDEX)
        self.assertFalse(dropped)
        self.assertEqual(calls, [])

    def test_pypis_untagged_cpu_wheel_is_removed_too(self):
        # The Windows wheel from PyPI is the CPU build and says so nowhere in
        # its version. "No tag, nothing to compare" left it in place through
        # every Reinstall on a machine with an RTX 4060, and ComfyUI died on
        # "Torch not compiled with CUDA enabled" at every start.
        dropped, calls = self._attempt(torch_info("2.14.0"),
                                       bootstrap.CUDA_INDEX)
        self.assertTrue(dropped)
        self.assertTrue(any("uninstall" in c for c in calls))

    def test_an_untagged_cuda_wheel_is_not_reinstalled_over_a_minor_version(self):
        # PyPI's Linux wheel is a CUDA build with no tag. It drives the card;
        # 3 GB is not worth trading cu126 for cu128.
        dropped, calls = self._attempt(torch_info("2.14.0", "12.6"),
                                       bootstrap.CUDA_INDEX)
        self.assertFalse(dropped)
        self.assertEqual(calls, [])

    def test_the_silent_uninstall_says_what_it_is(self):
        # pip prints nothing while it deletes thousands of files; on Windows
        # that is a minute or more of a button that looks stuck.
        said = []
        with mock.patch.object(bootstrap, "installed_torch",
                               return_value=torch_info("2.14.0+cpu")), \
             mock.patch.object(bootstrap, "_run", return_value=
                               subprocess.CompletedProcess([], 0, "", "")):
            bootstrap.drop_mismatched_torch(
                "py", bootstrap.CUDA_INDEX, lambda _m: None,
                lambda text, pct: said.append((text, pct)))
        self.assertEqual(said[-1], ("Removing torch 2.14.0+cpu (the cpu "
                                    "build) first…", None))

    def test_nothing_is_removed_until_the_index_has_a_replacement(self):
        # Uninstall first, find out at the download: an environment with no
        # torch at all, worse than the CPU build it replaced.
        calls = []
        with mock.patch.object(bootstrap, "installed_torch",
                               return_value=torch_info("2.14.0+cpu")), \
             mock.patch.object(bootstrap, "_run", side_effect=lambda cmd, **kw:
                               calls.append(cmd) or subprocess.CompletedProcess(
                                   cmd, 1, "", "ERROR: No matching "
                                               "distribution found for torch")):
            with self.assertRaises(RuntimeError) as caught:
                bootstrap.drop_mismatched_torch("py", bootstrap.CUDA_INDEX,
                                                lambda _m: None)
        self.assertFalse(any("uninstall" in c for c in calls),
                         "it removed torch with nothing to replace it")
        self.assertIn("left in place", str(caught.exception))
        self.assertIn("No matching distribution", str(caught.exception))

    def test_an_index_that_cannot_be_asked_does_not_block_the_install(self):
        # pip too old for `pip index`: that is not an answer, so no refusal.
        with mock.patch.object(bootstrap, "_run", return_value=
                               subprocess.CompletedProcess(
                                   [], 1, "", 'ERROR: unknown command "index"')):
            self.assertEqual(bootstrap.index_lacks_torch(
                "py", bootstrap.CUDA_INDEX), "")

    def test_a_cuda_build_is_removed_when_the_cpu_one_is_asked_for(self):
        dropped, _ = self._attempt(torch_info("2.14.0", "12.8"),
                                   bootstrap.CPU_INDEX)
        self.assertTrue(dropped)

    def test_nothing_is_removed_when_no_torch_is_there(self):
        dropped, calls = self._attempt({}, bootstrap.CUDA_INDEX)
        self.assertFalse(dropped)
        self.assertEqual(calls, [])

    def test_an_uninstall_that_fails_stops_there_and_says_why(self):
        # It used to log and carry on: pip then found the old build still in
        # place, called the request satisfied, and the task said "PyTorch
        # installed" over the build it had failed to remove.
        with self.assertRaises(RuntimeError) as caught:
            self._attempt(torch_info("2.14.0+cpu"), bootstrap.CUDA_INDEX,
                          code=1, said="ERROR: [WinError 5] Access is denied: "
                                       "'torch\\lib\\c10.dll'")
        self.assertIn("Access is denied", str(caught.exception))
        self.assertIn("Close it", str(caught.exception))

    def _install(self, after, index=bootstrap.CUDA_INDEX):
        calls = []
        seen = iter([after])
        with mock.patch.object(bootstrap, "torch_index", return_value=index), \
             mock.patch.object(bootstrap, "nvidia_gpu", return_value=RTX_4060), \
             mock.patch.object(bootstrap, "drop_mismatched_torch",
                               side_effect=lambda *a: calls.append("drop")), \
             mock.patch.object(bootstrap, "installed_torch",
                               side_effect=lambda _py: next(seen)), \
             mock.patch.object(bootstrap, "pip_install",
                               side_effect=lambda _py, args, *_a:
                               calls.append(args)):
            bootstrap.install_requested_torch("py", {}, lambda _m: None)
        return calls

    def test_the_selected_build_is_the_last_dependency_installed(self):
        calls = self._install(torch_info("2.10.0+cu128", "12.8"))
        # torchvision rides along: the drop takes it out, ComfyUI needs it,
        # and it has to match the torch it was built against.
        self.assertEqual(calls, ["drop", [
            "torch", "torchvision", "torchaudio",
            "--index-url", bootstrap.CUDA_INDEX]])

    def test_a_build_pip_left_in_place_fails_the_install(self):
        # "PyTorch installed" over a CPU build is the report that sent someone
        # to restart an engine that could only ever stop as it started.
        with self.assertRaises(RuntimeError) as caught:
            self._install(torch_info("2.14.0"))
        self.assertIn("still", str(caught.exception))
        self.assertIn("cu128", str(caught.exception))


class WhenAnInstallFails(unittest.TestCase):
    """Reinstall looked as if it did nothing: its progress and its failure
    both went to a panel a screen below the button, and the failure said
    only "see the log"."""

    @unittest.skipIf(sys.platform == "win32", "uses a shell script as python")
    def test_pip_failing_says_what_pip_said(self):
        root = Path(tempfile.mkdtemp(prefix="sb-pip-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        fake = root / "python"
        fake.write_text(
            "#!/bin/sh\n"
            "echo 'Collecting torch'\n"
            "echo 'ERROR: Could not find a version that satisfies the "
            "requirement torch (from versions: none)'\n"
            "echo 'ERROR: No matching distribution found for torch'\n"
            "exit 1\n")
        fake.chmod(0o755)
        with mock.patch.object(bootstrap, "pip_ready"), \
             mock.patch.object(bootstrap, "pip_raw_progress", return_value=[]):
            with self.assertRaises(RuntimeError) as caught:
                bootstrap.pip_install(str(fake), ["torch"], lambda _m: None)
        self.assertIn("No matching distribution found for torch",
                      str(caught.exception))

    def test_a_comfyui_requirement_that_fails_does_not_keep_the_cpu_build(self):
        # Stopping at ComfyUI's requirements left the row reading exactly as
        # it had before the button was pressed. The build of PyTorch is what
        # the button is for; the requirement is reported, afterwards.
        root = Path(tempfile.mkdtemp(prefix="sb-reinstall-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        comfy = root / "ComfyUI-Qwen3-TTS"
        comfy.mkdir()
        (comfy / "main.py").write_text("")
        (comfy / "requirements.txt").write_text("av>=99\n")
        vpy = bootstrap.venv_python(comfy)
        vpy.parent.mkdir(parents=True)
        vpy.write_text("")
        cfg = copy.deepcopy(bootstrap.DEFAULT_CONFIG)
        bootstrap.engine_cfg(cfg, "qwen").update(comfy_dir=str(comfy),
                                                 managed=True)
        calls = []

        def pip(_py, args, *_a):
            calls.append(args)
            if args[:1] == ["-r"]:
                raise RuntimeError("pip install failed: ERROR: No matching "
                                   "distribution found for av>=99")

        with mock.patch.object(bootstrap, "pip_install", side_effect=pip), \
             mock.patch.object(bootstrap, "install_requested_torch",
                               side_effect=lambda *_a: calls.append("torch")), \
             mock.patch.object(bootstrap, "save_config"):
            with self.assertRaises(RuntimeError) as caught:
                manager._install_torch(
                    manager.Task("dependency", "Install PyTorch"), cfg,
                    {"engine": "qwen"})
        self.assertEqual(calls[-1], "torch")
        self.assertIn("PyTorch is in place", str(caught.exception))
        self.assertIn("av>=99", str(caught.exception))


class OnePipPerEnvironment(unittest.TestCase):
    """A button that looked as if it had done nothing got pressed again, and
    the one below it — two pips writing one site-packages, one of them
    uninstalling torch, break each other."""

    def _finish(self, task, timeout=10):
        deadline = time.time() + timeout
        while task.state == "running" and time.time() < deadline:
            time.sleep(0.05)
        return task.state

    def test_a_second_install_into_the_same_environment_is_refused(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        with mock.patch.object(manager, "_install_torch",
                               side_effect=lambda *_a: gate.wait(10)), \
             mock.patch.object(manager, "_install_node_reqs"), \
             mock.patch.object(manager, "_install_git"):
            first = manager.install_dependency("torch_qwen", {}, {})
            with self.assertRaises(manager.InstallBusy) as caught:
                manager.install_dependency("node_reqs_qwen", {}, {})
            self.assertIn("Install PyTorch for Qwen3-TTS",
                          str(caught.exception))
            # And the page is told so, as a refusal rather than a fault.
            with server.app.test_client() as web:
                r = web.post("/api/deps/torch_qwen/install", json={})
            self.assertEqual(r.status_code, 409)
            self.assertIn("still running", r.get_json()["error"])
            # MOSS's environment is not Qwen's, and Git is not pip at all.
            for other in ("node_reqs_moss", "git"):
                self.assertEqual(self._finish(manager.install_dependency(
                    other, {}, {})), "done")
            gate.set()
            self.assertEqual(self._finish(first), "done")
            self.assertEqual(self._finish(manager.install_dependency(
                "node_reqs_qwen", {}, {})), "done")


def fake_torch_site(site: Path, version: str = "2.11.0+cu128",
                    files: dict | None = None, record: bool = True) -> Path:
    """A torch installed the way pip leaves one: files, dist-info, RECORD.

    Point PYTHONPATH at `site` and the environment's own interpreter finds it
    exactly as it would a real one — importlib.metadata reads the RECORD, and
    find_spec finds the package.
    """
    import base64, hashlib
    files = files or {
        "torch/__init__.py": "from .version import __version__\n",
        "torch/version.py": f"__version__ = {version!r}\ncuda = '12.8'\n",
        "torch/utils/__init__.py": "",
        "torch/utils/_debug_mode.py": "MODE = 1\n",
        "torch/_subclasses/fake_tensor.py": "def _is_plain_tensor(t):\n    "
                                            "return True\n",
        "torchgen/__init__.py": "",
    }
    info = site / f"torch-{version}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    rows = []
    for rel, text in files.items():
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        (site / rel).write_text(text)
        digest = base64.urlsafe_b64encode(
            hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()
        rows.append(f"{rel},sha256={digest},{len(text.encode())}")
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: torch\nVersion: {version}\n")
    if record:
        rows.append(f"torch-{version}.dist-info/METADATA,,")
        rows.append(f"torch-{version}.dist-info/RECORD,,")
        (info / "RECORD").write_text("\n".join(rows) + "\n")
    return site


class ADamagedTorch(unittest.TestCase):
    """The Qwen engine died with "cannot import name 'is_fake_tensor'" from
    inside torch: files of two versions mixed by an install that was cut off.
    Its version.py read as the right build, so the row said ok, Start
    launched it, and Reinstall — builds agreeing — did nothing at all."""

    def setUp(self):
        self.site = Path(tempfile.mkdtemp(prefix="sb-site-"))
        self.addCleanup(shutil.rmtree, self.site, ignore_errors=True)
        patch = mock.patch.dict(os.environ, {"PYTHONPATH": str(self.site)})
        patch.start()
        self.addCleanup(patch.stop)

    def damage(self) -> str:
        return bootstrap.torch_damage_summary(
            bootstrap.torch_damage(sys.executable))

    def test_a_whole_install_is_not_damaged(self):
        fake_torch_site(self.site)
        self.assertEqual(self.damage(), "")

    def test_files_left_by_another_version_are_found(self):
        # The shape of the real one: a package directory from the newer
        # torch, left behind beside the older torch's module of that name.
        fake_torch_site(self.site)
        stale = self.site / "torch" / "utils" / "_debug_mode"
        stale.mkdir()
        (stale / "__init__.py").write_text("from ._calls import *\n")
        (stale / "_calls.py").write_text("from torch._subclasses.fake_tensor "
                                         "import is_fake_tensor\n")
        said = self.damage()
        self.assertIn("2 files from another version are mixed in", said)
        self.assertIn("torch/utils/_debug_mode/", said)

    def test_a_file_that_never_arrived_is_found(self):
        fake_torch_site(self.site)
        (self.site / "torch" / "_subclasses" / "fake_tensor.py").unlink()
        self.assertIn("1 of its files is missing", self.damage())

    def test_a_file_from_another_version_in_its_place_is_found(self):
        fake_torch_site(self.site)
        (self.site / "torch" / "utils" / "_debug_mode.py").write_text("NEW\n")
        self.assertIn("1 of its files is not the one that was installed "
                      "(torch/utils/_debug_mode.py)", self.damage())

    def test_two_versions_installed_over_each_other_are_found(self):
        fake_torch_site(self.site)
        fake_torch_site(self.site, "2.14.0+cpu")
        self.assertIn("two versions of torch", self.damage())

    def test_torch_with_no_record_of_installing_it_is_found(self):
        fake_torch_site(self.site)
        shutil.rmtree(self.site / "torch-2.11.0+cu128.dist-info")
        self.assertIn("pip has no record", self.damage())

    def test_an_install_with_no_record_file_is_not_judged(self):
        # conda and some system packages ship no RECORD: every file would
        # read as a stranger, and a working torch would be called damaged.
        fake_torch_site(self.site, record=False)
        (self.site / "torch" / "extra.py").write_text("")
        self.assertEqual(self.damage(), "")

    def test_reinstall_takes_out_what_pip_does_not_know_about(self):
        fake_torch_site(self.site)
        stale = self.site / "torch" / "utils" / "_debug_mode"
        stale.mkdir()
        (stale / "__init__.py").write_text("")
        real_run = bootstrap._run

        def run(cmd, **kw):
            if "uninstall" in cmd:
                # What pip does: exactly the files its RECORD lists.
                info = next(self.site.glob("torch-*.dist-info"))
                for row in (info / "RECORD").read_text().splitlines():
                    target = self.site / row.split(",")[0]
                    if target.is_file():
                        target.unlink()
                shutil.rmtree(info)
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return real_run(cmd, **kw)

        with mock.patch.object(bootstrap, "_run", side_effect=run):
            bootstrap.remove_torch(sys.executable, "the damaged torch",
                                   lambda _m: None)
        self.assertFalse((self.site / "torch").exists(),
                         "the leftovers pip did not know about are still there")
        self.assertFalse((self.site / "torchgen").exists())
        self.assertEqual(bootstrap.torch_damage(sys.executable)["dists"], {})

    def test_reinstall_repairs_a_damaged_torch_of_the_right_build(self):
        # Builds agreeing used to mean pip called it satisfied and nothing
        # happened. A damaged torch now goes out before it goes back in.
        calls = []
        damaged = {"dists": {"torch": ["2.11.0+cu128"]}, "stray_count": 2,
                   "stray": ["torch/utils/_debug_mode/__init__.py"]}
        states = iter([damaged, {}])
        with mock.patch.object(bootstrap, "torch_index",
                               return_value=bootstrap.CUDA_INDEX), \
             mock.patch.object(bootstrap, "nvidia_gpu", return_value=RTX_4060), \
             mock.patch.object(bootstrap, "drop_mismatched_torch",
                               return_value=False), \
             mock.patch.object(bootstrap, "torch_damage",
                               side_effect=lambda _py: next(states)), \
             mock.patch.object(bootstrap, "index_lacks_torch", return_value=""), \
             mock.patch.object(bootstrap, "installed_torch",
                               return_value=torch_info("2.11.0+cu128", "12.8")), \
             mock.patch.object(bootstrap, "remove_torch",
                               side_effect=lambda *a, **k: calls.append("remove")), \
             mock.patch.object(bootstrap, "pip_install",
                               side_effect=lambda _py, args, *_a:
                               calls.append("install")):
            bootstrap.install_requested_torch("py", {}, lambda _m: None)
        self.assertEqual(calls, ["remove", "install"])

    def test_a_damaged_torch_is_not_removed_with_nothing_to_replace_it(self):
        damaged = {"dists": {"torch": ["2.11.0+cu128"]}, "missing_count": 1,
                   "missing": ["torch/x.py"]}
        with mock.patch.object(bootstrap, "torch_index",
                               return_value=bootstrap.CUDA_INDEX), \
             mock.patch.object(bootstrap, "nvidia_gpu", return_value=RTX_4060), \
             mock.patch.object(bootstrap, "drop_mismatched_torch",
                               return_value=False), \
             mock.patch.object(bootstrap, "torch_damage", return_value=damaged), \
             mock.patch.object(bootstrap, "installed_torch",
                               return_value=torch_info("2.11.0+cu128", "12.8")), \
             mock.patch.object(bootstrap, "index_lacks_torch",
                               return_value="ERROR: No matching distribution"), \
             mock.patch.object(bootstrap, "remove_torch") as removed:
            with self.assertRaises(RuntimeError) as caught:
                bootstrap.install_requested_torch("py", {}, lambda _m: None)
        removed.assert_not_called()
        self.assertIn("left in place", str(caught.exception))

    def test_start_refuses_a_damaged_torch_and_names_the_fix(self):
        with mock.patch.object(bootstrap, "torch_damage", return_value={
                "dists": {"torch": ["2.11.0+cu128"]}, "stray_count": 1,
                "stray": ["torch/utils/_debug_mode/__init__.py"]}):
            flags, refusal = bootstrap.torch_launch("py", {}, "qwen")
        self.assertEqual(flags, [])
        self.assertIn("damaged", refusal)
        self.assertIn("Reinstall on PyTorch · Qwen3-TTS", refusal)

    def test_the_row_marks_a_damaged_torch_for_repair(self):
        with mock.patch.object(manager, "_probe", return_value=(0, json.dumps(
                {"v": "2.11.0+cu128", "cuda": True, "built": "12.8",
                 "hip": None, "dev": "NVIDIA GeForce RTX 4060"}))), \
             mock.patch.object(bootstrap, "torch_damage", return_value={
                 "dists": {"torch": ["2.11.0+cu128"]}, "stray_count": 1,
                 "stray": ["torch/utils/_debug_mode/__init__.py"]}), \
             mock.patch.object(bootstrap, "installed_torch",
                               return_value=torch_info("2.11.0+cu128", "12.8")):
            row = manager._torch_row("py", "_qwen", "Qwen3-TTS", {})
        self.assertTrue(row.get("repair"))
        self.assertIn("damaged", row["detail"])

    def test_a_torch_that_will_not_import_is_not_called_missing(self):
        with mock.patch.object(manager, "_probe", return_value=(
                1, "Traceback …\nImportError: DLL load failed")), \
             mock.patch.object(bootstrap, "torch_damage", return_value={}), \
             mock.patch.object(bootstrap, "installed_torch",
                               return_value=torch_info("2.11.0+cu128", "12.8")):
            row = manager._torch_row("py", "_qwen", "Qwen3-TTS", {})
        self.assertTrue(row.get("repair"))
        self.assertIn("will not import: ImportError: DLL load failed",
                      row["detail"])


# The last words of the Qwen engine that prompted all this, as ComfyUI's
# console had them — Windows paths, and the colour-coded lines it went on
# printing after the traceback.
DAMAGED_TORCH_CRASH = [
    "Traceback (most recent call last):",
    '  File "D:\\AI\\Text-to-Speech-Model-main\\ComfyUI-Qwen3-TTS\\main.py", '
    "line 145, in <module>",
    '  File "D:\\AI\\Text-to-Speech-Model-main\\comfy-venv-ComfyUI-Qwen3-TTS\\'
    'Lib\\site-packages\\torch\\utils\\_debug_mode\\_utils.py", line 14, '
    "in <module>",
    "    from torch._subclasses.fake_tensor import is_fake_tensor",
    "ImportError: cannot import name 'is_fake_tensor' from "
    "'torch._subclasses.fake_tensor' (D:\\AI\\Text-to-Speech-Model-main\\"
    "comfy-venv-ComfyUI-Qwen3-TTS\\Lib\\site-packages\\torch\\_subclasses\\"
    "fake_tensor.py). Did you mean: '_is_plain_tensor'?",
    "\x1b[32m[INFO]\x1b[0m FakeTensor cache stats:",
    "\x1b[32m[INFO]\x1b[0m   cache_hits: 0",
]


class WhyItStoppedWhileStarting(unittest.TestCase):
    """"Stopped while starting — its last words are below" over a traceback
    is rule 15's stack trace with extra steps."""

    def test_a_damaged_torch_is_named_and_the_fix_given(self):
        said = bootstrap.crash_reason(DAMAGED_TORCH_CRASH, "qwen")
        self.assertIn("Qwen3-TTS's PyTorch is damaged", said)
        self.assertIn("is_fake_tensor", said)
        self.assertIn("Reinstall on PyTorch · Qwen3-TTS", said)

    def test_the_cpu_build_is_named(self):
        said = bootstrap.crash_reason(
            ['  File "x\\site-packages\\torch\\cuda\\__init__.py", line 1',
             "AssertionError: Torch not compiled with CUDA enabled"], "moss")
        self.assertIn("MOSS-TTS's PyTorch is the CPU-only build", said)

    def test_an_import_error_outside_torch_is_not_blamed_on_torch(self):
        said = bootstrap.crash_reason(
            ['  File "x\\site-packages\\transformers\\__init__.py", line 1',
             "ImportError: cannot import name 'thing'"], "qwen")
        self.assertNotIn("PyTorch", said)
        self.assertIn("ImportError: cannot import name 'thing'", said)

    def test_a_taken_port_is_named(self):
        said = bootstrap.crash_reason(
            ["OSError: [WinError 10048] Only one usage of each socket address "
             "(protocol/network address/port) is normally permitted"], "qwen")
        self.assertIn("port is taken", said)

    def test_a_console_with_no_exception_still_says_something(self):
        self.assertIn("stopped while starting",
                      bootstrap.crash_reason(["Starting server"], "qwen"))


class ReadingTheTorchBuild(unittest.TestCase):
    """What a torch was built for comes out of the wheel, through the
    environment's own interpreter, and without importing torch — that costs
    seconds on Windows and is asked before every engine start."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="sb-torch-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _wheel(self, version: str, cuda) -> None:
        pkg = self.root / "torch"
        pkg.mkdir()
        # Importing it would fail the test: the probe must not.
        (pkg / "__init__.py").write_text("raise RuntimeError('imported')\n")
        (pkg / "version.py").write_text(
            "from typing import Optional\n"
            f"__version__ = {version!r}\n"
            f"cuda: Optional[str] = {cuda!r}\n"
            "hip: Optional[str] = None\n")

    def _read(self) -> dict:
        with mock.patch.dict(os.environ, {"PYTHONPATH": str(self.root)}):
            return bootstrap.installed_torch(sys.executable)

    def test_pypis_windows_wheel_reads_as_the_cpu_build(self):
        self._wheel("2.14.0", None)
        info = self._read()
        self.assertEqual(info["version"], "2.14.0")
        self.assertEqual(bootstrap.torch_kind(info), "cpu")

    def test_a_cuda_wheel_reads_as_cuda(self):
        self._wheel("2.10.0+cu128", "12.8")
        info = self._read()
        self.assertEqual(info["cuda"], "12.8")
        self.assertEqual(bootstrap.torch_kind(info), "cuda")

    def test_no_torch_is_an_empty_answer(self):
        self.assertEqual(self._read(), {})
        self.assertEqual(bootstrap.torch_kind({}), "")


class ACpuOnlyTorch(unittest.TestCase):
    """ComfyUI asks CUDA for a device while it imports, so a torch with no GPU
    support in it stops as it starts unless it was told --cpu. Which of the
    two that means depends on the machine."""

    def _launch(self, installed, gpu=RTX_4060, cfg=None):
        with mock.patch.object(bootstrap.platform, "system",
                               return_value="Windows"), \
             mock.patch.object(bootstrap, "installed_torch",
                               return_value=installed), \
             mock.patch.object(bootstrap, "nvidia_gpu", return_value=gpu):
            return bootstrap.torch_launch("py", cfg or {}, "qwen")

    def test_beside_an_nvidia_card_it_is_refused_and_the_fix_named(self):
        flags, refusal = self._launch(torch_info("2.14.0"))
        self.assertEqual(flags, [])
        self.assertIn("CPU-only build", refusal)
        self.assertIn("RTX 4060", refusal)
        self.assertIn("Reinstall on PyTorch · Qwen3-TTS", refusal)

    def test_with_no_nvidia_card_it_runs_on_the_cpu(self):
        self.assertEqual(self._launch(torch_info("2.14.0+cpu"), NO_GPU),
                         (["--cpu"], ""))

    def test_the_cpu_build_chosen_on_purpose_runs_on_the_cpu(self):
        # A card too old for cu128 is a reason to pick the CPU build, and
        # refusing to run it would leave that machine nothing at all.
        self.assertEqual(
            self._launch(torch_info("2.14.0+cpu"),
                         cfg={"torch_index": bootstrap.CPU_INDEX}),
            (["--cpu"], ""))

    def test_a_cuda_build_is_started_as_it_is(self):
        # Never --cpu for a CUDA build, even when nvidia-smi cannot be found:
        # a portable ComfyUI carries its own CUDA, and silently running it on
        # the CPU would be rule 5b in a new coat.
        for gpu in (RTX_4060, NO_GPU):
            with self.subTest(gpu=gpu["name"]):
                self.assertEqual(
                    self._launch(torch_info("2.10.0+cu128", "12.8"), gpu),
                    ([], ""))

    def test_a_torch_that_cannot_be_read_is_left_to_comfyui(self):
        self.assertEqual(self._launch({}), ([], ""))

    def test_a_mac_is_never_given_the_flag(self):
        with mock.patch.object(bootstrap.platform, "system",
                               return_value="Darwin"), \
             mock.patch.object(bootstrap, "installed_torch",
                               return_value=torch_info("2.14.0")):
            self.assertEqual(bootstrap.torch_launch("py", {}, "qwen"), ([], ""))


class TheTorchRow(unittest.TestCase):
    """The row has to see a CPU build as CPU when the tag does not say so, and
    mark it as the repair it is when it is what stops the engine starting."""

    def _row(self, probe: dict, gpu=RTX_4060, cfg=None) -> dict:
        with mock.patch.object(manager, "_probe",
                               return_value=(0, json.dumps(probe))), \
             mock.patch.object(bootstrap, "nvidia_gpu", return_value=gpu):
            return manager._torch_row("py", "_qwen", "Qwen3-TTS", cfg or {})

    def test_an_untagged_cpu_build_beside_a_card_is_a_repair(self):
        row = self._row({"v": "2.14.0", "cuda": False, "built": None,
                         "hip": None, "dev": ""})
        self.assertTrue(row.get("repair"))
        self.assertIn("CPU-only build", row["detail"])
        self.assertIn("RTX 4060", row["detail"])

    def test_a_cpu_build_chosen_on_purpose_is_not_a_repair(self):
        row = self._row({"v": "2.14.0+cpu", "cuda": False, "built": None,
                         "hip": None, "dev": ""},
                        cfg={"torch_index": bootstrap.CPU_INDEX})
        self.assertFalse(row.get("repair"))

    def test_a_cuda_build_waiting_on_a_driver_is_not_a_repair(self):
        # Reinstalling torch cannot install a driver.
        row = self._row({"v": "2.10.0+cu128", "cuda": False, "built": "12.8",
                         "hip": None, "dev": ""},
                        gpu={"name": "NVIDIA GeForce RTX 4060",
                             "driver": False, "vram_mb": 0})
        self.assertFalse(row.get("repair"))
        self.assertIn("driver", row["detail"])

    def test_a_working_gpu_is_ok(self):
        row = self._row({"v": "2.10.0+cu128", "cuda": True, "built": "12.8",
                         "hip": None, "dev": "NVIDIA GeForce RTX 4060"})
        self.assertEqual(row["state"], "ok")
        self.assertFalse(row.get("repair"))


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
                     # The Delay 8B's numbers, whatever the loader holds.
                     "temperature": ["FLOAT", {"default": 1.7}],
                     "top_p": ["FLOAT", {"default": 0.8}],
                     "top_k": ["INT", {"default": 25}],
                     "repetition_penalty": ["FLOAT", {"default": 1.0}]},
        "optional": {"reference_audio": ["AUDIO"]}}},
    "MossTTSVoiceDesign": {"input": {"required": {
        "moss_pipe": ["MOSS_TTS_PIPE"],
        "language": [["auto", "zh", "en"], {"default": "auto"}],
        "text": ["STRING", {"default": ""}],
        "instruction": ["STRING", {"default": ""}],
        "seed": ["INT", {"default": 0}],
        "temperature": ["FLOAT", {"default": 1.5}],
        "top_p": ["FLOAT", {"default": 0.6}],
        "top_k": ["INT", {"default": 50}],
        "repetition_penalty": ["FLOAT", {"default": 1.1}]}}},
    "LoadAudio": {"input": {"required": {"audio": [["ref.wav"], {}]}}},
    **SAVE,
}

MOSS_DIRS = {
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer": "/m/moss-tts/Local",
    "OpenMOSS-Team/MOSS-Audio-Tokenizer": "/m/moss-tts/Codec",
    "OpenMOSS-Team/MOSS-VoiceGenerator": "/m/moss-tts/VG",
}
MOSS_OPTS = {"engine": "moss", "moss_dirs": MOSS_DIRS, "temperature": 0.9,
             "prefer_wav": True}


class ModelFolderLayout(unittest.TestCase):
    """The two engines do not agree on where a model folder goes, and neither
    layout is a preference — each is where that node looks."""

    ROOT = Path("/models")

    def test_qwen_drops_the_org(self):
        # The node's README draws models/qwen-tts/Qwen/<Name>; its code lists
        # models/qwen-tts one level deep and downloads to <Name>. Following
        # the README put every folder where the node never looked.
        self.assertEqual(
            bootstrap.model_dir(self.ROOT, "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
            self.ROOT / "qwen-tts" / "Qwen3-TTS-12Hz-0.6B-Base")

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


def qwen_node_finds(models_dir: Path, model_type: str, choice: str):
    """Where ComfyUI-Qwen-TTS loads a model from, or None where it would go
    to HuggingFace for a second copy — transcribed from its nodes.py.

    load_qwen_model lists models/qwen-tts one level deep for a folder whose
    name holds both the size and the kind; failing that,
    download_model_if_needed looks at <qwen_root>/<repo.split("/")[-1]> and
    downloads there when it is absent.
    """
    hf = {("Base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
          ("Base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
          ("VoiceDesign", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
          ("CustomVoice", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
          ("CustomVoice", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}
    base = models_dir / "qwen-tts"
    for d in os.listdir(base):
        cand = base / d
        if cand.is_dir() and choice in d and model_type.lower() in d.lower():
            return cand
    target = base / hf[(model_type, choice)].split("/")[-1]
    return target if target.is_dir() else None


class TheQwenNodeFindsWhatWeDownload(unittest.TestCase):
    """Every folder setup fetches has to be the one the node loads. The app
    followed the node's README, which draws models/qwen-tts/Qwen/<Name>; the
    node's code has only ever looked one level down. Nothing failed that the
    app could see — the node quietly fetched every model a second time on the
    first take, and offline the take failed."""

    def setUp(self):
        self.models = Path(tempfile.mkdtemp(prefix="sb-qwen-"))
        self.addCleanup(shutil.rmtree, self.models, ignore_errors=True)
        for m in bootstrap.MODEL_REPOS:
            d = bootstrap.model_dir(self.models, m["repo"], "qwen")
            d.mkdir(parents=True)
            (d / "model.safetensors").write_text("w")
        (self.models / "qwen-tts" / "voices").mkdir()

    def test_every_model_a_line_can_ask_for_is_found_on_disk(self):
        for m in bootstrap.MODEL_REPOS:
            name = m["repo"].split("/")[-1]
            if "Tokenizer" in name:
                continue
            kind = next(k for k in ("CustomVoice", "VoiceDesign", "Base")
                        if k in name)
            size = "0.6B" if "0.6B" in name else "1.7B"
            with self.subTest(repo=m["repo"]):
                self.assertEqual(
                    qwen_node_finds(self.models, kind, size),
                    bootstrap.model_dir(self.models, m["repo"], "qwen"))

    def test_the_tokenizer_is_where_the_node_checks_for_it(self):
        # check_and_download_tokenizer runs before every first load and
        # downloads to <qwen_root>/Qwen3-TTS-Tokenizer-12Hz when that is absent.
        self.assertEqual(
            bootstrap.model_dir(self.models, "Qwen/Qwen3-TTS-Tokenizer-12Hz"),
            self.models / "qwen-tts" / "Qwen3-TTS-Tokenizer-12Hz")

    def test_the_old_shape_is_invisible_to_the_node(self):
        # Proof the transcription is honest: the folders this app used to
        # write are not found, which is the fault being fixed.
        old = Path(tempfile.mkdtemp(prefix="sb-qwen-old-"))
        self.addCleanup(shutil.rmtree, old, ignore_errors=True)
        d = old / "qwen-tts" / "Qwen" / "Qwen3-TTS-12Hz-0.6B-CustomVoice"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_text("w")
        self.assertIsNone(qwen_node_finds(old, "CustomVoice", "0.6B"))

    def test_the_models_page_lists_them_under_their_own_repo_ids(self):
        cfg = dict(bootstrap.DEFAULT_CONFIG, want_moss=False)
        cfg["engines"] = {eid: dict(bootstrap.engine_defaults(eid),
                                    models_dir=str(self.models))
                          for eid in bootstrap.ENGINES}
        rows = {r["repo"] for r in manager.local_models(cfg)
                if r["engine"] == "qwen"}
        # voices/ is the node's saved voices, not a model.
        self.assertEqual(rows, {m["repo"] for m in bootstrap.MODEL_REPOS})


class OldQwenFoldersMoveIntoPlace(unittest.TestCase):
    """Folders an earlier version left in models/qwen-tts/Qwen/<Name> are
    moved to where the node looks, rather than downloaded a third time."""

    NAME = "Qwen3-TTS-12Hz-0.6B-CustomVoice"

    def setUp(self):
        self.models = Path(tempfile.mkdtemp(prefix="sb-migrate-"))
        self.addCleanup(shutil.rmtree, self.models, ignore_errors=True)
        self.root = self.models / "qwen-tts"
        self.old = self.root / "Qwen" / self.NAME
        self.new = self.root / self.NAME
        self.said = []

    def _whole(self, d, tag):
        (d / "speech_tokenizer").mkdir(parents=True)
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_text(tag)
        (d / "speech_tokenizer" / "model.safetensors").write_text(tag)

    def _half(self, d):
        # What huggingface_hub leaves when the node's own download is cut
        # off: config first, weights still arriving under .cache.
        (d / ".cache" / "huggingface" / "download").mkdir(parents=True)
        (d / "config.json").write_text("{}")
        (d / ".cache" / "huggingface" / "download"
         / "model.safetensors.incomplete").write_text("half")

    def _migrate(self):
        return bootstrap.migrate_qwen_layout(self.models, self.said.append)

    def test_a_folder_in_the_old_shape_is_moved(self):
        self._whole(self.old, "ours")
        self.assertEqual(self._migrate(), 1)
        self.assertEqual((self.new / "model.safetensors").read_text(), "ours")
        self.assertFalse((self.root / "Qwen").exists())
        self.assertTrue(bootstrap.model_installed(
            self.models, "Qwen/" + self.NAME))
        self.assertTrue(self.said)

    def test_a_second_copy_behind_a_whole_one_is_removed(self):
        # The node already fetched its own; ours is gigabytes nobody can
        # reach, and the Models page no longer lists it to delete.
        self._whole(self.old, "ours")
        self._whole(self.new, "node's")
        self._migrate()
        self.assertFalse(self.old.exists())
        self.assertEqual((self.new / "model.safetensors").read_text(),
                         "node's")

    def test_an_unfinished_node_download_is_replaced_by_a_whole_copy(self):
        # The node loads from any folder that exists, whole or not — so a
        # download it was cut off in the middle of fails every line after.
        self._whole(self.old, "ours")
        self._half(self.new)
        self.assertFalse(bootstrap.model_installed(
            self.models, "Qwen/" + self.NAME))
        self._migrate()
        self.assertEqual((self.new / "model.safetensors").read_text(), "ours")
        self.assertFalse((self.new / ".cache").exists())
        self.assertTrue(bootstrap.model_installed(
            self.models, "Qwen/" + self.NAME))

    def test_two_unfinished_copies_are_both_left_for_a_download(self):
        (self.old).mkdir(parents=True)
        (self.old / "model.safetensors.part").write_text("half")
        self._half(self.new)
        self.assertEqual(self._migrate(), 0)
        self.assertTrue(self.old.exists())
        self.assertTrue(self.new.exists())

    def test_nothing_but_our_own_org_folders_is_touched(self):
        self._whole(self.root / "voices" / "x", "voice")
        self._whole(self.root / "SomeoneElse" / "Model", "theirs")
        self._migrate()
        self.assertTrue((self.root / "voices" / "x").exists())
        self.assertTrue((self.root / "SomeoneElse" / "Model").exists())

    def test_it_runs_on_a_missing_or_unset_folder(self):
        self.assertEqual(bootstrap.migrate_qwen_layout(None), 0)
        self.assertEqual(bootstrap.migrate_qwen_layout(
            self.models / "nowhere"), 0)
        self._whole(self.old, "ours")
        self._migrate()
        self.assertEqual(self._migrate(), 0)  # a second run changes nothing


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

    def _sampling(self, voice, **opts):
        ins = self.c.build_line({"text": "Hi."}, voice,
                                dict(MOSS_OPTS, **opts))["prompt"]["4"]["inputs"]
        return {k: ins[k] for k in ("temperature", "top_p", "top_k",
                                    "repetition_penalty")}

    def test_each_checkpoint_samples_the_way_openmoss_tuned_it(self):
        # MossTTSGenerate's defaults are the Delay 8B's whatever the loader
        # holds. Left to them, the Local 1.7B — the default model — ran with
        # no repetition penalty and half its top_k.
        self.assertEqual(self._sampling({"kind": "preset"}), {
            "temperature": 1.0, "top_p": 0.95, "top_k": 50,
            "repetition_penalty": 1.1})
        self.assertEqual(
            self._sampling({"kind": "preset"},
                           moss_model="OpenMOSS-Team/MOSS-TTS"),
            {"temperature": 1.7, "top_p": 0.8, "top_k": 25,
             "repetition_penalty": 1.0})

    def test_a_designed_voice_samples_as_voicegenerator_whatever_is_picked(self):
        self.assertEqual(
            self._sampling({"kind": "design", "instruct": "A low narrator"},
                           moss_model="OpenMOSS-Team/MOSS-TTS"),
            {"temperature": 1.5, "top_p": 0.6, "top_k": 50,
             "repetition_penalty": 1.1})

    def test_expressiveness_scales_the_models_temperature(self):
        # The slider rests at 0.9, Qwen's own temperature. Passed through as
        # it was, it cooled the 8B from 1.7 and VoiceGenerator from 1.5.
        hot = self._sampling({"kind": "preset"}, temperature=1.8)
        self.assertAlmostEqual(hot["temperature"], 2.0)
        cool = self._sampling({"kind": "design", "instruct": "x"},
                              temperature=0.45)
        self.assertAlmostEqual(cool["temperature"], 0.75)
        # Only the temperature moves; the rest is the checkpoint's own.
        self.assertEqual(hot["top_k"], 50)
        self.assertEqual(hot["repetition_penalty"], 1.1)

    def test_the_tuning_table_names_every_model_a_line_can_load(self):
        for m in bootstrap.MOSS_MODEL_REPOS:
            if "Tokenizer" in m["repo"]:
                continue
            with self.subTest(repo=m["repo"]):
                self.assertIn(m["repo"], comfy.MOSS_SAMPLING)

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
            (d / "model.safetensors").write_bytes(b"\0" * 32)

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


# --------------------------------------------------------------------------- #
# the engine kit: real processes, real ports
# --------------------------------------------------------------------------- #
# Everything below runs actual processes on actual ports. Mocking the takeover
# would only prove that the mock returns what it was told to: the whole point
# is that a port is really held, a pid is really found, and a process really
# does or does not close.
import requests  # noqa: E402  (in requirements.txt; the app itself uses it)

MOCK_COMFY = Path(__file__).resolve().parent / "mock_comfy.py"
# The suite talks to servers on this machine, and a proxy in the environment
# would swallow every one of those requests.
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def online(url: str, timeout: float = 20) -> bool:
    import time as _t
    deadline = _t.time() + timeout
    while _t.time() < deadline:
        if bootstrap.comfy_online(url):
            return True
        _t.sleep(0.2)
    return False


def offline(url: str, timeout: float = 20) -> bool:
    import time as _t
    deadline = _t.time() + timeout
    while _t.time() < deadline:
        if not bootstrap.comfy_online(url):
            return True
        _t.sleep(0.2)
    return False


def spawn_mock(port: int, root: Path) -> subprocess.Popen:
    """A stand-in ComfyUI on a port of its own, as its own process — which is
    what makes it something the app has to find and close rather than drop."""
    root.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [sys.executable, str(MOCK_COMFY), str(root)],
        env={**os.environ, "MOCK_COMFY_PORT": str(port)},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)


def fake_install(root: Path, engine: str = "qwen",
                 with_nodes: bool = False) -> Path:
    """A pretend ComfyUI checkout whose main.py serves the stand-in engine.

    This is what lets the app truly own, stop and restart a process in a test:
    ComfyProcess.start runs `<python> main.py --port N` in this folder, so the
    engine it ends up managing is a real child of the app.

    `with_nodes` puts the node pack's marker file on disk without putting its
    classes in the engine — an install that is complete and an engine that
    started before it was, which is the state Restart exists for.
    """
    install = root / bootstrap.ENGINES[engine]["dir_name"]
    (install / "models").mkdir(parents=True, exist_ok=True)
    (install / "main.py").write_text(
        "import argparse, os, pathlib, runpy, sys\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--listen'); p.add_argument('--port')\n"
        "p.add_argument('--disable-auto-launch', action='store_true')\n"
        "p.add_argument('--cpu', action='store_true')\n"
        "a = p.parse_args()\n"
        "os.environ['MOCK_COMFY_PORT'] = a.port\n"
        "print('Device: cpu' if a.cpu else 'Device: cuda:0', flush=True)\n"
        "print('Starting server', flush=True)\n"
        "here = pathlib.Path(__file__).parent\n"
        "sys.argv = ['mock_comfy.py', str(here / 'mockroot')]\n"
        f"runpy.run_path({str(MOCK_COMFY)!r}, run_name='__main__')\n")
    if with_nodes:
        eng = bootstrap.ENGINES[engine]
        pack = install / "custom_nodes" / eng["node_dir"]
        pack.mkdir(parents=True, exist_ok=True)
        (pack / eng["node_marker"]).write_text("# pretend node pack\n")
    return install


def drop_weights(cfg: dict, engine: str) -> None:
    """Every model this config asks of that engine, on disk and whole."""
    models = bootstrap.engine_models_dir(cfg, engine)
    for m in bootstrap.wanted_models(cfg, engine):
        folder = bootstrap.model_dir(models, m["repo"], engine)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text("{}")
        (folder / "model.safetensors").write_bytes(b"\x00" * 16)


class EngineFixture(unittest.TestCase):
    """Shared setup: a throwaway config, and nothing left running after."""

    engine = "qwen"

    def setUp(self):
        self.saved = copy.deepcopy(server.cfg)
        self.root = Path(tempfile.mkdtemp(prefix="sb-engine-"))
        self.strays: list[subprocess.Popen] = []
        server.cfg["setup_complete"] = True

    def tearDown(self):
        for proc in server.PROCS.values():
            proc.stop()
            proc.lines.clear()
        for p in self.strays:
            try:
                p.kill()
                p.wait(timeout=5)
            except Exception:
                pass
        server.cfg.clear()
        server.cfg.update(self.saved)
        for client in server.CLIENTS.values():
            client._schema = None
        shutil.rmtree(self.root, ignore_errors=True)

    def stray(self, proc: subprocess.Popen) -> subprocess.Popen:
        self.strays.append(proc)
        return proc

    def slot(self, **kw) -> dict:
        slot = bootstrap.engine_cfg(server.cfg, self.engine)
        slot.update(kw)
        return slot

    def finish_task(self, view: dict, timeout: float = 60) -> str:
        import time as _t
        deadline = _t.time() + timeout
        task = manager.TASKS.get(view["id"])
        while _t.time() < deadline and task and task.state == "running":
            _t.sleep(0.25)
        return task.state if task else "gone"


class TheEngineConsole(EngineFixture):
    """The engine's own output, and what the app did to it, in one window.

    "Check the ComfyUI console" is not an instruction anyone running from a
    launcher can follow — there is no console. This endpoint is the console,
    and note() is how the app's own half of the story gets into it.
    """

    def test_the_tail_reports_shape_and_state(self):
        self.slot(comfy_url=f"http://127.0.0.1:{free_port()}")
        with server.app.test_client() as web:
            body = web.get("/api/comfy/log?engine=qwen").get_json()
        self.assertEqual(body["engine"], "qwen")
        self.assertEqual(body["lines"], [])
        self.assertFalse(body["running"])
        self.assertFalse(body["online"])

    def test_what_the_app_did_to_the_engine_is_in_it(self):
        server.PROCS["qwen"].note("Stopping pid 1234 — stopped")
        with server.app.test_client() as web:
            lines = web.get("/api/comfy/log?engine=qwen").get_json()["lines"]
        self.assertIn("[Script Builder] Stopping pid 1234 — stopped", lines)

    def test_the_count_is_clamped_and_never_a_500(self):
        for i in range(500):
            server.PROCS["qwen"].note(f"line {i}")
        with server.app.test_client() as web:
            self.assertEqual(
                len(web.get("/api/comfy/log?n=9999").get_json()["lines"]), 400)
            self.assertEqual(
                len(web.get("/api/comfy/log?n=0").get_json()["lines"]), 1)
            # A value typed into a URL is not a reason for a stack trace.
            junk = web.get("/api/comfy/log?n=lots")
            self.assertEqual(junk.status_code, 200)
            self.assertEqual(len(junk.get_json()["lines"]), 80)

    def test_an_engine_that_does_not_exist_is_a_sentence(self):
        with server.app.test_client() as web:
            r = web.get("/api/comfy/log?engine=nope")
        self.assertEqual(r.status_code, 400)
        self.assertIn("nope", r.get_json()["error"])


class RestartTakesTheFourRoutes(EngineFixture):
    """Start said "already running", Restart said "not started by this app",
    and the only advice left was to hunt a windowless python in Task Manager.

    Each route now says which one it took, because "Restarting ComfyUI" over a
    takeover hides the part that matters — something else was on that port and
    has just been closed.
    """

    def test_nothing_running_is_a_plain_start(self):
        port = free_port()
        install = fake_install(self.root)
        self.slot(comfy_url=f"http://127.0.0.1:{port}",
                  comfy_dir=str(install), python=sys.executable)
        with server.app.test_client() as web:
            body = web.post("/api/comfy/restart?engine=qwen").get_json()
        self.assertEqual(body["how"], "started")
        self.assertEqual(self.finish_task(body["task"]), "done")
        self.assertTrue(online(f"http://127.0.0.1:{port}"))

    def test_one_we_own_is_stopped_and_started_under_a_new_pid(self):
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        install = fake_install(self.root)
        self.slot(comfy_url=url, comfy_dir=str(install),
                  python=sys.executable)
        server.PROCS["qwen"].start(sys.executable, install, port,
                                   server.progress)
        self.assertTrue(online(url))
        before = server.PROCS["qwen"].proc.pid
        with server.app.test_client() as web:
            body = web.post("/api/comfy/restart?engine=qwen").get_json()
        self.assertEqual(body["how"], "managed")
        self.assertEqual(self.finish_task(body["task"]), "done")
        self.assertNotEqual(server.PROCS["qwen"].proc.pid, before)
        self.assertTrue(online(url))

    def test_somebody_elses_is_closed_and_replaced(self):
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        orphan = self.stray(spawn_mock(port, self.root / "orphan"))
        self.assertTrue(online(url))
        install = fake_install(self.root)
        self.slot(comfy_url=url, comfy_dir=str(install),
                  python=sys.executable)
        with server.app.test_client() as web:
            body = web.post("/api/comfy/restart?engine=qwen").get_json()
        self.assertEqual(body["how"], "takeover")
        self.assertIsNotNone(orphan.poll(), "the orphan was left running")
        self.assertEqual(self.finish_task(body["task"]), "done")
        self.assertTrue(server.PROCS["qwen"].alive())
        # And the console says what was done to it, not just that it happened.
        said = "\n".join(server.PROCS["qwen"].tail(200))
        self.assertIn("was not started here", said)
        self.assertIn("Stopping pid", said)

    def test_an_engine_with_no_install_is_refused_before_anything_is_killed(self):
        # Taking a port from someone and having nothing to start in its place
        # is not a restart, it is a hole — so this one is answered before the
        # takeover, and the ComfyUI on the port is left alone.
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        theirs = self.stray(spawn_mock(port, self.root / "theirs"))
        self.assertTrue(online(url))
        self.slot(comfy_url=url, comfy_dir="", python="", managed=True)
        with server.app.test_client() as web:
            r = web.post("/api/comfy/restart?engine=qwen")
        self.assertEqual(r.status_code, 400)
        self.assertIn("setup", r.get_json()["error"].lower())
        self.assertIsNone(theirs.poll(), "it closed a ComfyUI it could not replace")


class StartingOnACpuOnlyTorch(EngineFixture):
    """What this is for: an RTX 4060, PyPI's CPU torch in Qwen's environment,
    and every Start and Restart ending in a stack trace that closed with
    "AssertionError: Torch not compiled with CUDA enabled"."""

    def setUp(self):
        super().setUp()
        server.cfg["torch_index"] = ""          # Automatic
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.install = fake_install(self.root)
        self.slot(comfy_url=self.url, comfy_dir=str(self.install),
                  python=sys.executable)

    def cpu_torch(self, gpu: dict) -> None:
        for patch in (mock.patch.object(bootstrap, "installed_torch",
                                        return_value=torch_info("2.14.0")),
                      mock.patch.object(bootstrap, "nvidia_gpu",
                                        return_value=gpu)):
            patch.start()
            self.addCleanup(patch.stop)

    def said(self) -> str:
        return "\n".join(server.PROCS["qwen"].tail(200))

    @unittest.skipIf(sys.platform == "darwin", "a Mac never gets --cpu")
    def test_start_beside_a_card_is_a_sentence_and_nothing_is_launched(self):
        self.cpu_torch(RTX_4060)
        with server.app.test_client() as web:
            r = web.post("/api/comfy/start?engine=qwen")
        self.assertEqual(r.status_code, 400)
        self.assertIn("Reinstall on PyTorch · Qwen3-TTS", r.get_json()["error"])
        self.assertIsNone(server.PROCS["qwen"].proc,
                          "it launched an engine that could only die")
        # And the engine console says it, not just the toast.
        self.assertIn("CPU-only build", self.said())

    @unittest.skipIf(sys.platform == "darwin", "a Mac never gets --cpu")
    def test_start_with_no_card_runs_it_on_the_cpu(self):
        self.cpu_torch(NO_GPU)
        with server.app.test_client() as web:
            r = web.post("/api/comfy/start?engine=qwen")
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertTrue(online(self.url))
        self.assertIn("--cpu", server.PROCS["qwen"].proc.args)
        deadline = time.time() + 10
        while "Device: cpu" not in self.said() and time.time() < deadline:
            time.sleep(0.1)
        self.assertIn("Device: cpu", self.said(),
                      "the flag never reached ComfyUI's own command line")
        self.assertIn("--cpu", self.said())

    @unittest.skipIf(sys.platform == "darwin", "a Mac never gets --cpu")
    def test_restart_refuses_before_it_takes_anyone_elses_port(self):
        # Rule 33a: no port is taken that cannot be filled.
        theirs = self.stray(spawn_mock(self.port, self.root / "theirs"))
        self.assertTrue(online(self.url))
        self.cpu_torch(RTX_4060)
        with server.app.test_client() as web:
            r = web.post("/api/comfy/restart?engine=qwen")
        self.assertEqual(r.status_code, 409)
        self.assertIn("CPU-only build", r.get_json()["error"])
        self.assertIsNone(theirs.poll(),
                          "it closed a ComfyUI it could not replace")

    def test_a_torch_reinstall_stops_the_engine_it_is_about_to_replace(self):
        # Windows will not let pip replace a DLL a running ComfyUI has loaded,
        # and torch is nothing but DLLs.
        server.PROCS["qwen"].start(sys.executable, self.install, self.port,
                                   server.progress)
        self.assertTrue(online(self.url))
        running = []
        with mock.patch.object(manager, "_install_torch",
                               side_effect=lambda *_a: running.append(
                                   server.PROCS["qwen"].alive())):
            with server.app.test_client() as web:
                body = web.post("/api/deps/torch_qwen/install",
                                json={}).get_json()
            self.assertEqual(self.finish_task(body["task"]), "done")
        self.assertEqual(running, [False])
        self.assertIn("Stopping this engine while its packages change",
                      self.said())

    def test_installing_git_leaves_the_engine_alone(self):
        stops = []
        with mock.patch.object(manager, "_install_git"):
            view = manager.install_dependency(
                "git", server.cfg, {},
                stop_engine=lambda e: stops.append(e) or True).view()
            self.assertEqual(self.finish_task(view), "done")
        self.assertEqual(stops, [])


class ChoosingAnEngineStartsIt(EngineFixture):
    """Choosing an engine brings it up and takes the other one down — one
    engine on the card (rule 31) — and choosing one that cannot start says
    why and leaves the one that was running alone. Both engines, real
    processes, real ports."""

    def setUp(self):
        super().setUp()
        server.cfg.update(run_both_engines=False, want_moss=True)
        self.urls = {}
        for eid in ("qwen", "moss"):
            self.urls[eid] = f"http://127.0.0.1:{free_port()}"
            bootstrap.engine_cfg(server.cfg, eid).update(
                comfy_url=self.urls[eid], python=sys.executable,
                comfy_dir=str(fake_install(self.root / eid, eid)),
                managed=True, auto_start=True)

    def choose(self, eid: str):
        with server.app.test_client() as web:
            return web.post(f"/api/comfy/start?engine={eid}")

    def test_each_engine_starts_when_chosen_and_the_other_stops(self):
        for eid, other in (("qwen", "moss"), ("moss", "qwen"),
                           ("qwen", "moss")):
            with self.subTest(chose=eid):
                r = self.choose(eid)
                self.assertEqual(r.status_code, 200, r.get_json())
                self.assertTrue(online(self.urls[eid]), f"{eid} never came up")
                self.assertTrue(offline(self.urls[other]),
                                f"{other} was left on the card")
                self.assertFalse(server.PROCS[other].alive())

    def test_an_engine_that_cannot_start_leaves_the_running_one_alone(self):
        self.assertEqual(self.choose("qwen").status_code, 200)
        self.assertTrue(online(self.urls["qwen"]))
        refuse = lambda _py, _cfg, eid: ([], "MOSS-TTS's PyTorch is damaged.") \
            if eid == "moss" else ([], "")
        with mock.patch.object(bootstrap, "torch_launch", side_effect=refuse):
            r = self.choose("moss")
        self.assertEqual(r.status_code, 400)
        self.assertIn("damaged", r.get_json()["error"])
        self.assertTrue(server.PROCS["qwen"].alive(),
                        "switching to a broken engine took the working one down")
        self.assertTrue(bootstrap.comfy_online(self.urls["qwen"]))


class AnEngineThatDiesWhileStarting(EngineFixture):
    """Restart sat on "Restarting…" for fifteen minutes over an engine that
    had died in its first seconds, and Start said "its last words are below"
    over a traceback. Both notice the exit at once and say why."""

    def setUp(self):
        super().setUp()
        self.url = f"http://127.0.0.1:{free_port()}"
        install = fake_install(self.root)
        (install / "main.py").write_text(
            "import sys\nprint(%r, flush=True)\nsys.exit(1)\n"
            % "\n".join(DAMAGED_TORCH_CRASH))
        self.slot(comfy_url=self.url, comfy_dir=str(install),
                  python=sys.executable)

    def test_restart_says_why_at_once(self):
        began = time.time()
        with server.app.test_client() as web:
            body = web.post("/api/comfy/restart?engine=qwen").get_json()
        self.assertEqual(self.finish_task(body["task"], timeout=30), "error")
        self.assertLess(time.time() - began, 30)
        detail = manager.TASKS.get(body["task"]["id"]).detail
        self.assertIn("PyTorch is damaged", detail)

    def test_start_reports_why_it_stopped_in_a_sentence(self):
        with server.app.test_client() as web:
            self.assertEqual(web.post("/api/comfy/start?engine=qwen")
                             .status_code, 200)
            deadline = time.time() + 20
            while not server.PROCS["qwen"].crashed() and time.time() < deadline:
                time.sleep(0.1)
            time.sleep(0.3)             # let the reader drain the pipe
            slot = web.get("/api/status?engine=qwen").get_json()["installs"]["qwen"]
            log = web.get("/api/comfy/log?engine=qwen").get_json()
        self.assertFalse(slot["running"])
        self.assertIn("PyTorch is damaged", slot["stopped"])
        self.assertIn("PyTorch is damaged", log["stopped"])

    def test_a_take_is_told_why_instead_of_waiting(self):
        began = time.time()
        why = server.activate("qwen")
        self.assertLess(time.time() - began, 30)
        self.assertIn("Reinstall on PyTorch · Qwen3-TTS", why)


class WhenThePortWillNotBeGivenUp(EngineFixture):
    """A refusal has to name the obstacle it actually hit.

    "It would not close" covers a process owned by an administrator, a
    supervisor respawning it, and a database that was never ComfyUI — and all
    three need a different sentence from the person reading it.
    """

    def test_something_supervising_it_is_diagnosed_not_shrugged_at(self):
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        supervisor = self.root / "supervisor.py"
        # ComfyUI Desktop and every launcher script behave exactly like this:
        # kill the engine and a second one is up before the port stops
        # answering. Quiet is only free once it stays quiet.
        supervisor.write_text(
            "import os, subprocess, sys, time\n"
            "while True:\n"
            f"    p = subprocess.Popen([sys.executable, {str(MOCK_COMFY)!r},\n"
            f"                          {str(self.root / 'sup')!r}],\n"
            "                         env=dict(os.environ,\n"
            f"                                  MOCK_COMFY_PORT='{port}'),\n"
            "                         stdout=subprocess.DEVNULL,\n"
            "                         stderr=subprocess.DEVNULL)\n"
            "    p.wait()\n"
            "    time.sleep(0.2)\n")
        self.stray(subprocess.Popen(
            [sys.executable, str(supervisor)], start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        self.assertTrue(online(url, 30))
        install = fake_install(self.root)
        self.slot(comfy_url=url, comfy_dir=str(install),
                  python=sys.executable)
        with server.app.test_client() as web:
            r = web.post("/api/comfy/restart?engine=qwen")
        self.assertEqual(r.status_code, 409)
        self.assertIn("supervising", r.get_json()["error"])

    def test_a_process_that_is_not_comfyui_is_named_and_left_alone(self):
        # The port is this engine's only by convention. Another app's dev
        # server on it answers every health check exactly like a ComfyUI, and
        # closing it would be this app doing real damage on a guess.
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        # Launched through a link whose name says nothing about python, so the
        # command line is the one the guard has to read in the wild.
        pretender = self.root / "acme-ledger-daemon"
        os.symlink(sys.executable, pretender)
        squatter = self.stray(subprocess.Popen(
            [str(pretender), "-c",
             "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
             "class H(BaseHTTPRequestHandler):\n"
             "    def do_GET(self):\n"
             "        self.send_response(200)\n"
             "        self.send_header('Content-Type', 'application/json')\n"
             "        self.end_headers()\n"
             "        self.wfile.write(b'{\"system\": {}}')\n"
             "    def log_message(self, *a): pass\n"
             f"HTTPServer(('127.0.0.1', {port}), H).serve_forever()\n"],
            start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        self.assertTrue(online(url))
        install = fake_install(self.root)
        self.slot(comfy_url=url, comfy_dir=str(install),
                  python=sys.executable)
        with server.app.test_client() as web:
            r = web.post("/api/comfy/restart?engine=qwen")
        self.assertEqual(r.status_code, 409)
        self.assertIn("acme-ledger-daemon", r.get_json()["error"])
        self.assertIsNone(squatter.poll(), "it killed something it should not")

    def test_kill_pid_reports_what_the_system_said(self):
        # A refusal that is guessed at reads the same as a process that was
        # never there, and those need different sentences.
        victim = self.stray(subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)"],
            start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        self.assertIn("time.sleep", bootstrap.pid_cmdline(victim.pid))
        # Reaped on another thread: a killed child nobody waits on stays in
        # the process table as a zombie, kill(pid, 0) keeps succeeding on it,
        # and a polite stop then reads as one that had to be forced. Nothing
        # the app closes is a child of its own, so that is a shape of this
        # test rather than of the function.
        threading.Thread(target=victim.wait, daemon=True).start()
        self.assertEqual(bootstrap.kill_pid(victim.pid), "stopped")
        self.assertEqual(bootstrap.kill_pid(victim.pid), "already gone")

    def test_a_fresh_process_cmdline_survives_the_exec_race(self):
        path = mock.Mock()
        path.exists.return_value = True
        path.read_bytes.side_effect = [b"", b"", b"python\0main.py\0"]
        with mock.patch.object(bootstrap, "Path", return_value=path), \
                mock.patch.object(bootstrap.time, "sleep") as sleep:
            self.assertEqual(bootstrap.pid_cmdline(123), "python main.py")
        self.assertEqual(path.read_bytes.call_count, 3)
        self.assertEqual(sleep.call_count, 2)


class WeightsTheEngineCannotReach(EngineFixture):
    """Script Builder's version of the stale-model-scan warning.

    Both node packs resolve their checkpoints per call, so weights that land
    behind a running engine are found without a restart — that half does not
    apply here. The half that does is rule 17: ComfyUI reads custom_nodes
    once, at startup, so an engine started before the pack landed is a
    complete install with no classes in it. Every folder present, every
    download finished, and nothing that can speak.
    """

    def _status(self) -> dict:
        with server.app.test_client() as web:
            return web.get("/api/status?engine=qwen").get_json()

    def test_a_complete_install_the_engine_cannot_use_is_named(self):
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        install = fake_install(self.root, with_nodes=True)
        self.slot(comfy_url=url, comfy_dir=str(install),
                  models_dir=str(install / "models"), python=sys.executable)
        drop_weights(server.cfg, "qwen")
        self.stray(spawn_mock(port, self.root / "orphan"))
        self.assertTrue(online(url))
        requests.post(f"{url}/mock/hide/qwen", timeout=5)
        server.for_engine("qwen").schema(force=True)

        st = self._status()
        self.assertTrue(st["comfy_online"])
        self.assertEqual(st["missing_models"], [])
        self.assertTrue(st["stale_models"])
        self.assertIn("started before", st["stale_reason"])

        # And it stops being stale the moment the classes are there — the
        # flag is about this engine, not about the download.
        requests.post(f"{url}/mock/hide/none", timeout=5)
        server.for_engine("qwen").schema(force=True)
        st = self._status()
        self.assertFalse(st["stale_models"])
        self.assertEqual(st["stale_reason"], "")

    def test_models_still_arriving_are_not_called_stale(self):
        # Nothing is on disk yet, so "the weights are here and unreachable" is
        # simply untrue — and a warning that fires during a first download is
        # one nobody reads the second time.
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        install = fake_install(self.root, with_nodes=True)
        self.slot(comfy_url=url, comfy_dir=str(install),
                  models_dir=str(install / "models"), python=sys.executable)
        self.stray(spawn_mock(port, self.root / "orphan"))
        self.assertTrue(online(url))
        requests.post(f"{url}/mock/hide/qwen", timeout=5)
        server.for_engine("qwen").schema(force=True)
        st = self._status()
        self.assertTrue(st["missing_models"])
        self.assertFalse(st["stale_models"])

    def test_moss_is_judged_on_the_one_list_that_names_checkpoints(self):
        # MossTTSModelLoader.model_variant is the only enum either pack
        # publishes that names models. Qwen's name sizes and speakers, which
        # is why it declares no marker at all.
        self.assertEqual(bootstrap.ENGINES["moss"]["model_marker"], "moss")
        self.assertEqual(bootstrap.ENGINES["qwen"]["model_marker"], "")
        loaded = client_for({
            "MossTTSModelLoader": {"input": {"required": {
                "model_variant": [["MOSS-TTS (Local 1.7B)"], {}]}}},
            "MossTTSGenerate": {"input": {"required": {}}}})
        self.assertEqual(loaded.model_list("moss"), ["MOSS-TTS (Local 1.7B)"])
        self.assertEqual(loaded.model_list("qwen"), [])


class WhichComfyUIIsAnswering(EngineFixture):
    """8188 is the port every ComfyUI picks, so the one holding it is often
    somebody else's — and status has to say so in a flag the console can read,
    not only in a sentence buried in the dependency report."""

    def test_a_different_install_on_the_address_is_a_mismatch(self):
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        self.stray(spawn_mock(port, self.root / "theirs"))
        self.assertTrue(online(url))
        requests.post(f"{url}/mock/argv", json={"root": "/somebody/elses/ComfyUI"},
                timeout=5)
        self.slot(comfy_url=url, comfy_dir="/opt/mine/ComfyUI")
        with server.app.test_client() as web:
            st = web.get("/api/status?engine=qwen").get_json()
        self.assertTrue(st["engine_mismatch"])
        self.assertIn("somebody/elses", st["engine_argv"])
        self.assertFalse(st["engine_managed"])

        requests.post(f"{url}/mock/argv", json={"root": "/opt/mine/ComfyUI"},
                timeout=5)
        with server.app.test_client() as web:
            st = web.get("/api/status?engine=qwen").get_json()
        self.assertFalse(st["engine_mismatch"])

    def test_a_build_that_will_not_say_is_not_accused(self):
        # Older ComfyUI reports no argv. Crying wolf about the usual case
        # teaches people to ignore the warning that matters.
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        self.stray(spawn_mock(port, self.root / "quiet"))
        self.assertTrue(online(url))
        self.slot(comfy_url=url, comfy_dir="/opt/mine/ComfyUI")
        with server.app.test_client() as web:
            st = web.get("/api/status?engine=qwen").get_json()
        self.assertFalse(st["engine_mismatch"])
        self.assertEqual(st["engine_argv"], "")


class ALaunchEndsWithAWorkingEngine(EngineFixture):
    """No button pressed. Offline: start it. Healthy: adopt it, and say so — a
    ComfyUI somebody left running is not a problem to be solved. Useless:
    replace it, through the same guard Restart uses."""

    def test_a_quiet_port_gets_an_engine_of_our_own(self):
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        install = fake_install(self.root)
        self.slot(comfy_url=url, comfy_dir=str(install),
                  models_dir=str(install / "models"), python=sys.executable)
        server.ensure_engine_at_boot()
        self.assertTrue(online(url, 30))
        self.assertTrue(server.PROCS["qwen"].alive())

    def test_a_healthy_engine_is_adopted_rather_than_killed(self):
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        install = fake_install(self.root, with_nodes=True)
        self.slot(comfy_url=url, comfy_dir=str(install),
                  models_dir=str(install / "models"), python=sys.executable)
        drop_weights(server.cfg, "qwen")
        healthy = self.stray(spawn_mock(port, self.root / "healthy"))
        self.assertTrue(online(url))
        server.ensure_engine_at_boot()
        self.assertIsNone(healthy.poll(), "it killed a working engine")
        self.assertFalse(server.PROCS["qwen"].alive())
        self.assertIn("Adopting", "\n".join(server.PROCS["qwen"].tail(50)))

    def test_an_engine_that_cannot_reach_the_weights_is_replaced(self):
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        install = fake_install(self.root, with_nodes=True)
        self.slot(comfy_url=url, comfy_dir=str(install),
                  models_dir=str(install / "models"), python=sys.executable)
        drop_weights(server.cfg, "qwen")
        orphan = self.stray(spawn_mock(port, self.root / "orphan"))
        self.assertTrue(online(url))
        requests.post(f"{url}/mock/hide/qwen", timeout=5)

        server.ensure_engine_at_boot()
        self.assertIsNotNone(orphan.poll(), "the useless engine was left up")
        self.assertTrue(server.PROCS["qwen"].alive())
        said = "\n".join(server.PROCS["qwen"].tail(200))
        self.assertIn("Replacing it", said)
        self.assertIn("Stopping pid", said)

    def test_an_engine_somebody_else_runs_is_never_touched(self):
        # External mode: managed False with no folder of its own. There is
        # nothing here to put back, so closing it would leave them with
        # nothing at all.
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        theirs = self.stray(spawn_mock(port, self.root / "theirs"))
        self.assertTrue(online(url))
        self.slot(comfy_url=url, comfy_dir="", python="", managed=False)
        server.ensure_engine_at_boot()
        self.assertIsNone(theirs.poll())
        self.assertIn("yours, not this app's",
                      "\n".join(server.PROCS["qwen"].tail(50)))

    def test_an_engine_set_not_to_start_is_left_alone(self):
        port = free_port()
        install = fake_install(self.root)
        self.slot(comfy_url=f"http://127.0.0.1:{port}",
                  comfy_dir=str(install), python=sys.executable,
                  auto_start=False)
        server.ensure_engine_at_boot()
        self.assertFalse(server.PROCS["qwen"].alive())
        self.assertFalse(bootstrap.comfy_online(f"http://127.0.0.1:{port}"))


class ProductionLaunch(unittest.TestCase):
    """Exercise the real entry point, not only its startup helper.

    A unit call to ``ensure_engine_at_boot`` can pass even if ``main`` stops
    invoking it, invokes it before loading the saved configuration, or fails
    to bring up the web application alongside it.  This is the launch shape a
    packaged user actually runs: a fresh server process and an offline,
    managed ComfyUI install.
    """

    def test_server_launch_starts_its_managed_engine(self):
        root = Path(tempfile.mkdtemp(prefix="sb-production-launch-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        data = root / "data"
        data.mkdir()
        install = fake_install(root)
        app_port, engine_port = free_port(), free_port()
        config = copy.deepcopy(bootstrap.DEFAULT_CONFIG)
        config.update({"setup_complete": True, "engine": "moss"})
        config["engines"] = {
            "qwen": dict(bootstrap.engine_defaults("qwen"),
                         comfy_url=f"http://127.0.0.1:{engine_port}",
                         comfy_dir=str(install),
                         models_dir=str(install / "models"),
                         python=sys.executable),
            "moss": bootstrap.engine_defaults("moss"),
        }
        (data / "config.json").write_text(json.dumps(config))
        env = {**os.environ,
               "SCRIPT_BUILDER_DATA": str(data),
               "SCRIPT_BUILDER_PORT": str(app_port),
               "SCRIPT_BUILDER_NO_BROWSER": "1"}
        app_proc = subprocess.Popen(
            [sys.executable, str(REPO / "server.py")], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        engine_url = f"http://127.0.0.1:{engine_port}"
        try:
            deadline = time.time() + 30
            app_url = f"http://127.0.0.1:{app_port}"
            while time.time() < deadline:
                try:
                    if requests.get(app_url, timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.1)
            else:
                self.fail("server.py did not make the application reachable")

            self.assertTrue(online(engine_url, 30),
                            "launch did not start the managed ComfyUI")
            log = requests.get(f"{app_url}/api/comfy/log?engine=qwen",
                               timeout=5).json()
            self.assertTrue(log["running"])
            self.assertTrue(log["online"])
            self.assertIn("Starting the engine",
                          "\n".join(log["lines"]))

            # Read through the same public API as the browser and inspect the
            # bytes it returns.  "Engine online" is not production-ready if
            # the first prompt cannot travel through ComfyUI and come back as
            # decodable, non-silent audio.
            speak = requests.post(f"{app_url}/api/speak", json={
                "mode": "multi", "style": "Clear and natural",
                "title": "Production launch audio",
                "lines": [{"speaker": 1,
                           "text": "The production launch can speak."},
                          {"speaker": 2,
                           "text": "It can design a second voice too."}],
                "speakers": {"1": {"name": "Narrator",
                                    "kind": "preset", "speaker": "Ryan"},
                             "2": {"name": "Designed voice",
                                   "kind": "design",
                                   "instruct": "A warm, confident voice"}},
                "model": "0.6B", "attention": "auto", "pause": 0.2,
            }, timeout=5)
            self.assertEqual(speak.status_code, 200, speak.text)
            job_id = speak.json()["job"]
            deadline = time.time() + 30
            job = None
            while time.time() < deadline:
                jobs = requests.get(f"{app_url}/api/jobs", timeout=5).json()
                job = next((item for item in jobs if item["id"] == job_id), None)
                if job and job["status"] != "running":
                    break
                time.sleep(0.1)
            self.assertIsNotNone(job, "audio job disappeared")
            self.assertEqual(job["status"], "done", job)
            take = job["take"]
            audio = requests.get(f"{app_url}/api/take/{take['id']}", timeout=5)
            self.assertEqual(audio.status_code, 200)
            with wave.open(io.BytesIO(audio.content), "rb") as wav:
                self.assertGreater(wav.getnframes(), 0)
                self.assertGreater(wav.getframerate(), 0)
                raw = wav.readframes(wav.getnframes())
            samples = struct.unpack(f"<{len(raw) // 2}h", raw)
            self.assertGreater(max(map(abs, samples)), 0,
                               "generated WAV contains only silence")

            # Voice cloning is the third advertised Qwen workflow. Feed the
            # take back through the upload endpoint, then require a second
            # generation to complete with that server-side reference name.
            upload = requests.post(
                f"{app_url}/api/upload-reference",
                files={"file": ("reference.wav", audio.content, "audio/wav")},
                timeout=5)
            self.assertEqual(upload.status_code, 200, upload.text)
            clone = requests.post(f"{app_url}/api/speak", json={
                "mode": "single", "title": "Production clone audio",
                "lines": [{"speaker": 1,
                           "text": "The cloned voice path works."}],
                "speakers": {"1": {"name": "Clone", "kind": "clone",
                                    "ref_audio": upload.json()["name"],
                                    "ref_text": "The production launch can speak."}},
                "model": "0.6B", "attention": "auto",
            }, timeout=5)
            self.assertEqual(clone.status_code, 200, clone.text)
            clone_id = clone.json()["job"]
            deadline = time.time() + 30
            clone_job = None
            while time.time() < deadline:
                jobs = requests.get(f"{app_url}/api/jobs", timeout=5).json()
                clone_job = next((item for item in jobs
                                  if item["id"] == clone_id), None)
                if clone_job and clone_job["status"] != "running":
                    break
                time.sleep(0.1)
            self.assertIsNotNone(clone_job, "clone job disappeared")
            self.assertEqual(clone_job["status"], "done", clone_job)

            # The page calls the transcript optional, and the node refuses a
            # clone without one unless it is told to copy the sound alone.
            bare = requests.post(f"{app_url}/api/speak", json={
                "mode": "single", "title": "Clone with no transcript",
                "lines": [{"speaker": 1, "text": "Nobody typed the words."}],
                "speakers": {"1": {"name": "Clone", "kind": "clone",
                                    "ref_audio": upload.json()["name"],
                                    "ref_text": ""}},
                "model": "0.6B", "attention": "auto",
            }, timeout=5)
            self.assertEqual(bare.status_code, 200, bare.text)
            bare_id = bare.json()["job"]
            deadline = time.time() + 30
            bare_job = None
            while time.time() < deadline:
                jobs = requests.get(f"{app_url}/api/jobs", timeout=5).json()
                bare_job = next((item for item in jobs
                                 if item["id"] == bare_id), None)
                if bare_job and bare_job["status"] != "running":
                    break
                time.sleep(0.1)
            self.assertIsNotNone(bare_job, "clone job disappeared")
            self.assertEqual(bare_job["status"], "done", bare_job)

            # Launch always returns to the primary engine, rather than
            # silently restoring the secondary engine from the last session.
            saved = json.loads((data / "config.json").read_text())
            self.assertEqual(saved["engine"], "qwen")
        finally:
            app_proc.terminate()
            try:
                app_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                app_proc.kill()
                app_proc.wait(timeout=5)
            # SIGTERM can bypass Flask's finally block on some interpreters;
            # never let the test's stand-in engine escape into the next test.
            for pid in bootstrap.port_pids(engine_port):
                bootstrap.kill_pid(pid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
