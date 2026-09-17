# Script Builder — invariants

## Hard rules (do not revisit)

1. **`web/index.html` stays one file with no build step.** It is the YuE Studio
   shell — rail, five pages, workspace column, player bar — with the script
   builder as the Create page. Same tokens and class names as YuE Studio so the
   two stay in step.
1b. **Textareas in the Script card are auto-sized, and `scrollHeight` reads 0
   while a page is hidden.** Any code path that renders blocks off-screen must
   call `resizeAll()` once the page is shown, or every line renders clipped.
2. **Never hard-code a ComfyUI workflow.** `comfy.py` builds each graph from
   `/object_info` and matches inputs through candidate-name lists. The
   Qwen-TTS node renames inputs between releases; a schema read turns that into
   a clear message instead of a wrong value.
3. **Preset voices, models and attention modes are read from the node, never
   typed into the source.** `CustomVoiceNode.speaker` is the only truth about
   which voices exist.
4. **Node requirements install into the interpreter ComfyUI runs on.** Portable
   `python_embeded\python.exe` first, then the managed venv. Never the system
   Python. This mirrors the documented install step.
5. **Python detection is by execution, never PATH lookup.** Windows Store stubs
   resolve on PATH and fail to run.
6. **Downloads are resumable.** Stream to `<name>.part`, `Range` on retry,
   atomic `replace()`. Whole-repo downloads skip `.bin` duplicates of
   safetensors and repo furniture.
7. **One line of dialogue is one graph.** Do not switch to
   `DialogueInferenceNode` — see below.
8. **Model deletes are path-checked**: repo must contain `/`, no `..`, and the
   resolved path must sit under `models_dir/qwen-tts`.

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
`ComfyUI/models/qwen-tts/Qwen/<name>/`, which is where the node searches.

## Stitching

`stitch_wavs` joins clips with `wave` only — no ffmpeg. It refuses when channel
count, sample width or rate differ between clips, and the caller falls back to a
zip. Keep it that way: adding a resampler would pull in a dependency the app
does not otherwise need.
