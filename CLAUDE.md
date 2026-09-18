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
12. **`takes.json` is the record; a folder missing from it is rubbish.** A run
   that fails or is cancelled records no take, so the clips it already fetched
   are unreachable — no card lists them, no Delete removes them. `run_job`
   clears the folder on every exit that is not a recorded take, and
   `sweep_orphan_takes()` clears what an earlier crash left.
13. **The page carries its own favicon, inline.** The server has no static
   route, so without it every load asks for `/favicon.ico` and logs a 404.
14. **Long work reports a percentage, and pip is asked what it supports.**
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

## Why line-by-line, not DialogueInferenceNode

`DialogueInferenceNode` takes a `RoleBankNode`, which takes prompts from
`VoiceClonePromptNode` — so every role needs reference audio. Preset voices
cannot be used with it at all. Generating per line keeps preset, cloned and
designed voices interchangeable, lets one line be retried without redoing the
script, and gives the per-block highlight during playback. The pause and the
join are done in `server.py`, not in the node.

## Validation gate — run after any edit

```bash
python -m py_compile server.py comfy.py bootstrap.py manager.py
python - <<'PY'
import re, pathlib
src = pathlib.Path('web/index.html').read_text()
pathlib.Path('/tmp/sb.js').write_text('\n'.join(re.findall(r'<script>(.*?)</script>', src, re.S)))
PY
node --check /tmp/sb.js
```

A missing function declaration in the inline script kills all interactivity
silently — `node --check` is not optional.

## Version floor

Node classes used: `CustomVoiceNode`, `VoiceCloneNode`, `VoiceDesignNode`,
`LoadAudio`, `SaveAudioAdvanced` (falls back to `SaveAudio`). Qwen3-TTS needs
`transformers==4.57.3` or `>=5.0` — the Engine panel checks this explicitly
because it is the usual cause of IMPORT FAILED.

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
