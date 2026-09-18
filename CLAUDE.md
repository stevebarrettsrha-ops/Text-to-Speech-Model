# Script Builder — invariants

## Hard rules (do not revisit)

1. **`web/index.html` stays one file with no build step.** It is the YuE Studio
   shell — rail, five pages, workspace column, player bar — with the script
   builder as the Create page. Same tokens and class names as YuE Studio so the
   two stay in step.
1b. **Textareas in the Script card are auto-sized, and `scrollHeight` reads 0
   while a page is hidden.** Any code path that renders blocks off-screen must
   call `resizeAll()` once the page is shown, or every line renders clipped.
1c. **The narrow breakpoint needs `min-width:0` and has to out-weigh
   `body.solo`.** A grid item is `min-width:auto`, so the rail refused to shrink
   below its content and `overflow-x:auto` never engaged — it widened the page
   instead of scrolling its own chips. And a media query adds no specificity, so
   plain `.app` inside `@media` loses to `body.solo .app`, which left Models and
   Engine on the desktop rail. Both are named explicitly in the 760px block.
1d. **`playTake` takes a start index, never a sliced take, and owns the player
   by token.** `stepLine` used to hand it `lines.slice(next)`, and that copy
   became `S.take`: "line 2 of 3" renumbered itself to "line 1 of 2", and the
   lines it dropped were the ones ⏮ needed, so one press of ⏭ left the other
   button with nowhere to go. The token matters because `stepLine` starts the
   next playback 120ms after telling the current one to stop — without it the
   older run reaches its cleanup last and switches the player off underneath
   its replacement.
2. **Never hard-code a ComfyUI workflow.** `comfy.py` builds each graph from
   `/object_info` and matches inputs through candidate-name lists. The
   Qwen-TTS node renames inputs between releases; a schema read turns that into
   a clear message instead of a wrong value.
3. **Preset voices, models and attention modes are read from the node, never
   typed into the source.** `CustomVoiceNode.speaker` is the only truth about
   which voices exist.
4. **Node requirements install into the interpreter ComfyUI runs on.** Portable
   `python_embeded\python.exe` first; then, for an install we did not make, the
   environment it already has (`existing_python` probes `venv/`, `.venv/`,
   `python_standalone`, by execution); then our own managed `comfy-venv`. Never
   the system Python. Building a second environment beside someone else's
   ComfyUI costs gigabytes and puts the requirements where ComfyUI never looks,
   so the nodes still fail to import.
5. **Python detection is by execution, never PATH lookup.** Windows Store stubs
   resolve on PATH and fail to run.
5b. **So is GPU detection.** `torch_index` gated on
   `shutil.which("nvidia-smi")` and, when that came back empty, installed the
   CPU wheel index on a machine with an RTX 4060 in it — so torch arrived as
   `2.14.0+cpu` and the Engine panel reported "no GPU found" to someone holding
   a GPU. `nvidia_gpu()` runs nvidia-smi from PATH *and* from the places the
   driver puts it, and falls back to the display-adapter list, which tells a
   missing driver apart from a missing card. The two get different sentences.
5c. **A build already installed satisfies pip, so Reinstall must uninstall
   first.** pip counts torch 2.14.0+cpu as satisfying `torch`; pointing it at
   the CUDA index and asking again changes nothing, which is why pressing
   Reinstall on the CPU build left the CPU build in place.
   `drop_mismatched_torch` compares the `+tag` against the index and removes
   the old one — and does nothing when they agree, or when the wheel carries
   no tag and there is nothing to compare.
6. **Downloads are resumable, and nothing is renamed until it is whole.** Stream
   to `<name>.part`, `Range` on retry, atomic `replace()` — but only once what
   arrived accounts for the size the listing gave. A dropped connection ends the
   chunk loop exactly like a clean finish, so promoting the short file made it
   look complete for good: the `.part` to resume from was gone, and the folder
   counted as installed with truncated weights in it. `model_installed()` also
   returns False while any `.part` remains, or a repo whose config.json landed
   first reports installed while its weights are still arriving. Whole-repo
   downloads skip `.bin` duplicates of safetensors and repo furniture.
7. **One line of dialogue is one graph.** Do not switch to
   `DialogueInferenceNode` — see below.
8. **Model deletes are path-checked**: repo must contain `/`, no `..`, and the
   resolved path must sit under `models_dir/qwen-tts`.
9. **`comfy_url` goes through `clean_url()`, and its port through
   `comfy_port()`.** Never `int(url.rsplit(":")[-1])`: a trailing slash or a
   port-less address makes that a `ValueError` at import time, and the whole
   server stops booting over a value typed into Settings. `load_config` heals
   an address saved before it was normalised.
10. **Setup steps cross the wire as a list, in run order.** Flask sorts the keys
   of every dict it sends, which listed *Check Python* last — after the step
   that starts the engine — on the one screen where order is the point.
11. **A launcher never installs into the Python it found.** `run.sh`/`run.bat`
   build a `.venv` beside themselves. Debian, Ubuntu and Homebrew mark their
   Python externally managed and pip refuses it (PEP 668), which took the
   launcher down with `set -e` before it ever reached `server.py`.
12. **`takes.json` is read, changed and written inside one hold of
   `takes_lock`, and moved into place from a `.tmp`.** `add_take` used to take
   the lock twice with a gap: two jobs finishing together each wrote the list
   they had read before the other's take was in it, and the loser vanished
   from the library while its audio stayed on disk for the sweep below to
   delete. Forty concurrent adds lost twenty-nine. The private `_read_takes`
   and `_write_takes` assume the caller holds the lock; the public ones take
   it.
12b. **A folder missing from `takes.json` is rubbish.** A run
   that fails or is cancelled records no take, so the clips it already fetched
   are unreachable — no card lists them, no Delete removes them. `run_job`
   clears the folder on every exit that is not a recorded take, and
   `sweep_orphan_takes()` clears what an earlier crash left.
13. **The page carries its own favicon, inline.** The server has no static
   route, so without it every load asks for `/favicon.ico` and logs a 404.
13b. **An unexpected fault reaches the person using it.** `error` and
   `unhandledrejection` are hooked at the very top of the inline script,
   because the faults worth catching are the ones during boot, and they report
   through the toast. Without it the page simply stops — the silent death the
   gate exists to catch before it ships. The reporter is defensive on purpose:
   `toast()` is declared further down and may not exist yet, so it falls back
   to writing the element directly, and it gives up after three so one fault
   cannot bury the page.
14. **Anything that grows is capped.** `progress.lines`, `Task.lines`,
   `ComfyProcess.lines`, `TASKS` and `takes.json` all have a limit; `jobs` was
   the one that did not, and a finished job holds the whole take while
   `/api/jobs` walks the lot once a second during a run.
15. **A dead engine is a sentence, not a stack trace.** `comfy.py` routes its
   requests through `_reach`, so a ComfyUI that crashes mid-take says so and
   says what to do, instead of surfacing "ConnectionError: HTTPConnectionPool
   (host='127.0.0.1', port=8188): Max retries exceeded" as the take's error.
16. **Long work reports a percentage, and pip is asked what it supports.**
   `pip_install` reads its pipe a character at a time, because pip redraws
   progress with `\r` and iterating by line waits for a `\n` that only lands
   once the download is over — which is why a 2.7 GB PyTorch showed
   "Collecting torch" and then nothing for minutes. The numbers come from
   `--progress-bar raw`, and whether to pass it is read out of `pip install
   --help`, never inferred from a version: pip 24.0 takes only on/off and
   exits with "invalid choice: 'raw'", so guessing there fails the install
   rather than merely losing the bar. Where pip is too old, the file name,
   its size and a running clock stand in. `Progress` steps carry a numeric
   `pct` that is None until there is a real number — a bar sitting at 0% for
   fifteen minutes reads as broken — and `download_repo`'s percentage is
   spread across the folders so the bar crosses the step once.
16b. **pip 24.1 is the floor for a percentage, and a fresh venv is below it.**
   `--progress-bar raw` arrived in pip 24.1; `python -m venv` hands you the pip
   its base Python bundled, which for 3.11 and 3.12 is 24.0. So the probe
   above correctly found no raw support and the longest step of the install
   showed no number at all. `pip_ready` upgrades pip once per interpreter
   before the first long install and asks again. It never raises: an upgrade
   that fails costs the percentage, not the install.
16c. **The unpacking step is measured, not guessed, and never given a bar.**
   pip prints nothing between "Installing collected packages" and
   "Successfully installed" — minutes, for a 2.7 GB torch. The heartbeat weighs
   site-packages against a baseline taken before pip ran, so a real byte count
   climbs. It is deliberately not shown as a percentage of the download: a
   wheel unpacks to more than it downloads, so that reads "199 MB of 88 MB" and
   looks like a fault. `Progress.detail(pct=None)` *clears* the bar rather than
   leaving the last one up, or the download's final 100% sits there for the
   whole silent stretch and reads as a run that finished and hung.
17. **ComfyUI reads `custom_nodes` once, at startup.** Installing the Qwen-TTS
   nodes into an engine that is already running leaves it running without
   them — the "Nodes not loaded" warning with nothing behind it. `run_setup`
   restarts the engine it owns; the Engine panel offers Restart, and if the
   nodes still do not appear, `node_import_error` imports the package in
   ComfyUI's own interpreter and reports the real exception. "Check the ComfyUI
   console" is not an instruction anyone running from a launcher can follow.
18. **The primary button names the blocker it can actually clear.** It read
   "Set up the engine" and opened the setup dialog for every not-ready state,
   including an engine that is set up and merely needs restarting — which that
   dialog cannot do. `blocker(status)` returns the label and where to go.

19. **Two engines, and everything that differs between them lives in
   `ENGINES`.** Node repo, node folder, the file that proves it is installed,
   the models sub-folder, the folder layout and the model list are one table
   entry each, so a third engine is a row rather than a hunt through four files.
   `engine_of(repo)` answers which engine a model belongs to from the tables
   themselves, and every model entry `wanted_models` returns carries its own
   `engine` because the caller downloading it has to know which layout to use.
20. **The two engines do not agree on where a model folder goes, and neither
   layout is a preference.** Qwen nests `models/qwen-tts/<Org>/<Name>`, which
   is where its node searches. MOSS flattens to `models/moss-tts/<Org>--<Name>`,
   because its loader builds that path from `repo_id.replace("/", "--")`. Put a
   MOSS folder in the Qwen shape and the node does not see it — it downloads a
   second copy of a model already on disk. Same trap in reverse in
   `local_models`: walking the nested shape over a flat folder lists nothing,
   so a downloaded MOSS model reads as never downloaded.
21. **MOSS is two nodes, and `local_model_path` is only ever a folder that is
   really there.** `MossTTSModelLoader` holds the weights and hands a
   `MOSS_TTS_PIPE` to `MossTTSGenerate` or `MossTTSVoiceDesign`. Its
   `_resolve_local_dir` treats that path as a path only when it can stat it and
   as a HuggingFace repo id otherwise — so passing a folder that has not been
   downloaded becomes `snapshot_download("D:\...\MOSS-TTS")`, which is not a
   repo id and fails. `moss_dirs()` leaves absent folders out, "" reaches the
   node, and the node fetches the model itself.
22. **MOSS has no preset speakers, and the page says so rather than showing an
   empty list.** There is no speaker enum on any MOSS node. `/api/voices`
   returns an empty list with `fallback` False — not the Qwen fallback names,
   which cannot be used — and the Voices card's first button reads *Own voice*:
   with neither a clip nor a description the base model speaks in a voice of
   its own, which changes with the seed.
23. **A designed voice loads MOSS-VoiceGenerator whatever the picker says.**
   `MossTTSVoiceDesign` warns and misbehaves on any other checkpoint. Same
   reasoning as Qwen's VoiceDesign forcing 1.7B — and, as there, which enum
   entry means that model is read off the node by substring
   (`MOSS_VARIANT_HINTS`), because the repo id each display name maps to lives
   in the node's constants and never reaches `/object_info`.
24. **Readiness is per engine.** With MOSS selected, a missing Qwen folder is
   not what stands between the script and a take; reporting it as one sends
   people to download a model they are not about to use. `/api/status` takes an
   `engine`, and `engine_nodes` reports both so the Engine panel can show a row
   each.

27. **The card's memory is read, and nothing is offered that cannot be held.**
   Every model entry carries a `vram_gb`, `nvidia_gpu()` reports `vram_mb`
   (`--query-gpu=name,memory.total`), and `ComfyClient.vram_mb()` is the second
   opinion for a portable ComfyUI that carries its own CUDA where nvidia-smi is
   not on PATH — rule 5b's gap again. `fits_vram` returns **None** when the
   card is unknown, and None is never treated as "too small": hiding models
   because nvidia-smi was missing would be rule 5b in a new coat. The model
   picker shows what will not fit and disables it, the Models page needs a
   confirm before downloading it, and everything a first run fetches by default
   is asserted to fit 8 GB.
28. **The quantized 8B is checked, never assumed.** OpenMOSS do fit the 8B on
   an 8 GB card, but through their own llama.cpp pipeline — Q4_K_M weights,
   staged loading, numpy LM heads — not through this ComfyUI node. Two of the
   five prerequisites cannot be downloaded at all: llama.cpp is compiled from
   source, and the TensorRT engines are built against the card in front of you
   ("we do **not** provide pre-built TensorRT engines"). So `GGUF_STEPS`
   describes and `/api/moss/8b` reports; neither installs, and
   `gguf_available()` looks the two HuggingFace repos up rather than taking
   them on trust. A first launch that promised an 8B it could not deliver
   would fail in the middle of someone's first take instead of here.

29. **The dependency report cannot answer "does it work", so there is a
   self-test.** Every row there can read ok while the first take still fails:
   folders present but holding no weights, classes loaded from a version whose
   inputs were renamed, a model larger than the card. `manager.selftest` runs
   the whole path in order — engine answering, nodes loaded, folders whole,
   graph builds, ComfyUI accepts it, audio comes back — and stops at the first
   step that breaks, naming it. It **forces a fresh schema read**: the cache
   lasts two minutes and the reason anyone presses Test is usually that
   something just changed, so reading it once reported "nodes are loaded"
   about a ComfyUI that had just been shown not to have them.
29b. **Silence is a failure, and a byte count is not a test.** A clip of the
   right length full of zeros decodes perfectly and plays nothing, which is
   exactly what a model that loaded and generated nothing sounds like —
   `_peak` catches it, and returns -1 rather than 0 for a width it cannot
   measure so 24-bit audio is never called silent. The folder check looks for
   weight **files**, not a size: the right floor for a tokenizer is not the
   right floor for an 8B, and picking one number gets both wrong.
29c. **The self-test reads ComfyUI's console over the run.** It is the only
   way to see what the API never reports — a model reaching for HuggingFace
   mid-generation because a processor could not find its codec locally, which
   is live for MOSS: `codec_local_path` is only used for TTSD, so the Local
   1.7B and VoiceGenerator resolve their audio tokenizer through
   `AutoProcessor.from_pretrained` instead. Where someone else started
   ComfyUI, that step reports skipped rather than passing on an empty tail.

30. **Each engine gets its own ComfyUI, and nothing below it is shared.**
   `ComfyUI-Qwen3-TTS` on 8188 and `ComfyUI-MOSS-TTS` on 8189, each a separate
   clone with its own environment, its own node pack and its own models folder
   inside it. Qwen wants `transformers` 4.57.3 or 5.0+, MOSS wants 4.40+, and
   they resolved together — until the day they do not. Separate installs mean
   one engine's requirements can never break the other's, and a broken node
   install takes down one engine rather than both. `venv_python` names the
   environment after the install (`comfy-venv-ComfyUI-MOSS-TTS`) because both
   sit under the same parent and a bare `comfy-venv` would be one environment
   shared by two — the thing this layout exists to prevent.
30b. **Per-engine settings live in `cfg["engines"][id]`, and a single-install
   config migrates into it.** `_migrate` gives the old top-level `comfy_dir`,
   `comfy_url`, `models_dir` and `python` to Qwen — it is the engine the app
   was built around and the one that ComfyUI was set up for — and starts MOSS
   from defaults, which means its own install to fetch. It is idempotent, and
   it leaves the old keys alone: deleting settings out from under someone who
   might downgrade is not worth the tidiness.
31. **Only the engine being used is left running.** Two ComfyUIs that have
   both generated each hold their models in their own process's VRAM, and
   `unload_all_models` only reaches inside one process — neither can free the
   other's. On an 8 GB card the second engine is the one that fails to
   allocate. `activate(engine)` stops the others and starts this one, and
   `/api/speak` and the self-test both go through it before a single line is
   queued, under `engine_lock` so two takes started together cannot leave both
   resident. `run_both_engines` turns it off where there is memory to spare.
31b. **`wait_for_prompt` is told which engine queued the line.** There are two
   clients now, and reading the selected one inside the wait loop polls the
   wrong ComfyUI the moment someone switches engines mid-take.
32. **The dependency report is per engine, and so are the install ids.**
   `comfyui_moss`, `torch_qwen`, `node_reqs_moss` — Python and Git are the only
   rows left that both engines share. `install_dependency` reads the engine off
   the suffix, and a bare id means the default engine, which is what a page
   written before the split would send.

## Why line-by-line, not DialogueInferenceNode

`DialogueInferenceNode` takes a `RoleBankNode`, which takes prompts from
`VoiceClonePromptNode` — so every role needs reference audio. Preset voices
cannot be used with it at all. Generating per line keeps preset, cloned and
designed voices interchangeable, lets one line be retried without redoing the
script, and gives the per-block highlight during playback. The pause and the
join are done in `server.py`, not in the node.

## Tests — run after any edit

```bash
node tests/check.mjs     # the gate: everything compiles, the inline script parses
npm run test:units       # 131 unit tests, standard library only
npm test                 # 81 checks driving the real page in headless Chromium
```

The gate is not optional: a missing function declaration in the inline script
kills all interactivity silently, and nothing else catches it.

Nothing in the suite needs a GPU, a model download or the network.
`tests/mock_comfy.py` answers for ComfyUI — its `/object_info` is transcribed
from both real node packs, and it runs ComfyUI's own graph validation, so a
graph that passes here passes there. `POST /mock/hide/<engine>` drops one
engine's classes, which is how "ComfyUI never loaded those nodes" is reproduced
without breaking an install. The suite runs **two** stand-in ComfyUIs, one per engine on its own port,
because that is the shape the app installs — and a MOSS graph arriving at
Qwen's ComfyUI is a failure the single-stand-in version could not have seen.
`tests/mock_hf.py` answers for HuggingFace, with
`Range` support and switches to cut a transfer off mid-file or ignore a resume.

`SCRIPT_BUILDER_DATA` moves `data/`, and the suite points it at a temporary
directory. Without that, running the tests would overwrite a real library.

Every check stands for something that broke once, so the names say what would
break rather than what the function is called. Add to them when you fix
something: a fault worth fixing is worth the test that would have caught it.

**A job name in `.github/workflows/test.yml` is also a required check in
`.github/branch-protection.json`.** GitHub accepts a required check that names
no job and then gates nothing, silently, so renaming a job without editing the
JSON would switch the gate off without a word. `tests/check.mjs` compares the
two — including expanding the `matrix.python` values — and fails on either a
required check with no job or a job nothing requires. It parses the YAML by
hand on purpose: the gate has to keep running with nothing installed.

## Version floor

Node classes used: `CustomVoiceNode`, `VoiceCloneNode`, `VoiceDesignNode`,
`LoadAudio`, `SaveAudioAdvanced` (falls back to `SaveAudio`), and from
MOSS `MossTTSModelLoader`, `MossTTSGenerate`, `MossTTSVoiceDesign`. Qwen3-TTS
needs `transformers==4.57.3` or `>=5.0` — the Engine panel checks this
explicitly because it is the usual cause of IMPORT FAILED. MOSS asks only for
`>=4.40.0`, so the Qwen floor is the binding one when both are installed into
the same interpreter, which they are.

MOSS models: `OpenMOSS-Team/MOSS-Audio-Tokenizer`,
`MOSS-TTS-Local-Transformer` (1.7B) and `MOSS-VoiceGenerator` (1.7B) are the
default set and all three run on 8 GB; `MOSS-TTS` (8B) is the only optional
one. The repo ids in `MOSS_MODEL_REPOS` are the ones in the node's own
`utils/constants.py` `MODEL_VARIANTS` — keep them in step with that file, the
same way `MODEL_REPOS` tracks the Qwen node's `HF_MODEL_MAP`.

25. **Model sizes come from OpenMOSS's table, never from the ComfyUI node's
   README.** That README lists MOSS-VoiceGenerator as "Delay 8B, ~18 GB",
   conflating the architecture with the size — `MossTTSDelay` is the
   architecture and OpenMOSS publishes VoiceGenerator at 1.7B. Believing it put
   MOSS voice design behind a warning that it would not run on an 8 GB card
   when it fits as easily as the base model does. Rule 3's reasoning again: the
   upstream source is the truth, a downstream README is a copy that drifts.
26. **The 8B is optional because of this node, not because of the model.**
   `MossTTSModelLoader` loads bf16 weights through
   `AutoModel.from_pretrained`, so 8B wants ~18 GB here. OpenMOSS's own
   llama.cpp path fits it on 8 GB with Q4_K_M weights, staged loading and a
   quantized KV cache; the ComfyUI node implements no part of that — no GGUF,
   no ONNX, no `low_memory`. Say which of the two is the limit when explaining
   it, or the next person removes the tick and runs out of VRAM.

Models: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` and `Qwen/Qwen3-TTS-Tokenizer-12Hz` are
required; the 1.7B Base and 1.7B VoiceDesign folders are optional. They live in
`ComfyUI/models/qwen-tts/Qwen/<name>/`, which is where the node searches. The
repo names in `MODEL_REPOS` are the ones in the node's own `HF_MODEL_MAP` — keep
them in step with it, not with the node's README, which lists fewer.

`VoiceDesignNode` raises on `model_choice="0.6B"`; only the 1.7B build exists.
`comfy.py` picks the 1.7B entry out of the node's own enum for a designed voice,
whatever the model picker says, because the picker offers 0.6B for cloning.

## Stitching

`stitch_wavs` joins clips with `wave` only — no ffmpeg. It refuses when channel
count, sample width or rate differ between clips, and the caller falls back to a
zip. Keep it that way: adding a resampler would pull in a dependency the app
does not otherwise need.

The gap between clips is a whole number of **frames**, never a rounded byte
count. `int(rate * pause * sampwidth * nchannels)` can land on half a frame —
0.75s at 22050 Hz stereo is one such — and every sample after it plays in the
wrong channel.

A refusal **raises**, so the one `except` deletes the half-built file. Returning
False from inside the `with` left the clips written so far on disk as `take.wav`
— a file that looks exactly like the joined take, holding one line of it, next
to the zip the caller then made.
