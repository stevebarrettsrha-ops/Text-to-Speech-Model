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
npm run test:units       # 102 unit tests, standard library only
npm test                 # 68 checks driving the real page in headless Chromium
```

The gate is not optional: a missing function declaration in the inline script
kills all interactivity silently, and nothing else catches it.

Nothing in the suite needs a GPU, a model download or the network.
`tests/mock_comfy.py` answers for ComfyUI — its `/object_info` is transcribed
from both real node packs, and it runs ComfyUI's own graph validation, so a
graph that passes here passes there. `POST /mock/hide/<engine>` drops one
engine's classes, which is how "ComfyUI never loaded those nodes" is reproduced
without breaking an install. `tests/mock_hf.py` answers for HuggingFace, with
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
