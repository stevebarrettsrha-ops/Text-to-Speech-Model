"""Unit tests for the parts that are easy to get quietly wrong.

Standard library only — `python -m unittest discover tests` needs nothing that
requirements.txt does not already install.

Every test here stands for a fault that actually shipped at some point, so the
names say what would break rather than what the function is called.
"""
from __future__ import annotations

import os
import shutil
import struct
import sys
import tempfile
import threading
import unittest
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
        self.cfg = {"models_dir": str(self.models)}

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
        prog.detail("deps", "unpacking", None)       # keeps the last number
        self.assertEqual(prog.snapshot()["steps"][3]["pct"], 42.0)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
