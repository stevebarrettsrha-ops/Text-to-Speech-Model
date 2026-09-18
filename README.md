# Script Builder

A local text-to-speech studio. Write a script with named speakers, press Run,
and Qwen3-TTS reads it on your own machine — no account, no upload, no per-word
billing.

The layout is the Script Builder file you already had: raw structure on the
left, dialogue blocks in the middle, speaker settings on the right. What changed
is underneath — the browser's own voices are gone, and every line is spoken by
Qwen3-TTS running through ComfyUI.

---

## Running it

**Windows** — double-click `run.bat`
**macOS / Linux** — `./run.sh`

Either one finds a Python 3.10+, offers to install one if there is none, puts
Script Builder's two packages into a `.venv` beside the script — never into the
Python it found — and starts the server. The browser opens at
<http://127.0.0.1:7799>.

On the first run a setup panel appears with three routes:

| Route | What happens |
|---|---|
| Use an existing ComfyUI | Any install found is listed. Only the nodes and models are added. |
| Install a fresh ComfyUI | Clones ComfyUI into `./ComfyUI`, adds the Qwen-TTS nodes, builds its own environment, downloads the models. |
| Connect to one I start myself | Set the address in Settings. Setup adds the nodes and models only. |

Setup does, in order, what the instructions you were given do by hand:

1. `git clone https://github.com/flybirdxx/ComfyUI-Qwen-TTS.git` into
   `ComfyUI/custom_nodes/`
2. `pip install -r ComfyUI/custom_nodes/ComfyUI-Qwen-TTS/requirements.txt`
3. downloads the Qwen3-TTS weights into `ComfyUI/models/qwen-tts/Qwen/`

Step 2 picks the interpreter ComfyUI actually runs on. On a portable install
that is `python_embeded\python.exe` — the same one your instructions name — and
on a managed install it is the virtual environment beside the ComfyUI folder.
Nothing goes into your system Python.

### Requirements

- Python 3.10 or newer (on Debian and Ubuntu, `python3-venv` too)
- Git
- An NVIDIA GPU with 8 GB or more is comfortable. Less works with **Free GPU
  memory after each run** switched on. CPU works but is slow.

### Model folders

All six repos in the [Qwen3-TTS collection](https://huggingface.co/collections/Qwen/qwen3-tts),
pulled into `ComfyUI/models/qwen-tts/Qwen/`:

| Folder | Size | What it does |
|---|---|---|
| `Qwen3-TTS-Tokenizer-12Hz` | 0.2B | Speech tokenizer — every voice needs it |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.9B | Preset speakers, fast |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 2B | Preset speakers, better |
| `Qwen3-TTS-12Hz-0.6B-Base` | 0.9B | Voice cloning, fast |
| `Qwen3-TTS-12Hz-1.7B-Base` | 2B | Voice cloning, better |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 2B | Voice from a written description |

Setup downloads only what you tick: the tokenizer and 0.6B CustomVoice always,
cloning and voice design if you want them, and the 1.7B versions of whichever of
those you chose. The rest are one button each on the Models page.

Which checkpoint feeds which node is worth knowing: **CustomVoice** carries the
preset speakers, **Base** does zero-shot cloning, **VoiceDesign** builds a voice
from a description. The node's own README only lists Base and VoiceDesign, so
the CustomVoice mapping is read off the model names — if a preset voice errors,
grab the matching Base folder from the Models page and the node will find it.

Duplicate weights are skipped: where a repo ships both `.safetensors` and
`.bin`, only the safetensors are fetched.

---

## Around the app

A left rail holds five pages, and the player bar sits across the bottom wherever
you are.

- **Home** — paste a script or type a line and press Read it. Lines shaped like
  `Speaker 1: hello` are split by speaker automatically.
- **Create** — the script builder. Script, Voices and More options as collapsible
  cards; the raw structure and your takes stay visible in the workspace column on
  the right.
- **Library** — every take as a card.
- **Models** — all the HuggingFace work.
- **Engine** — dependencies, PyTorch build, activity log.

## Writing a script

**Style instructions** set the reading tone for the whole script and are passed
to every line.

**Lines** are what gets said. The coloured chip flips a line between Speaker 1
and Speaker 2; the arrows reorder and × deletes. **Two speakers / One speaker**
at the top collapses everything onto Speaker 1 when you only need one voice.

`Ctrl` + `Enter` reads the script from anywhere.

### Three ways to set a voice

Each speaker picks one in the Voices card:

- **Preset** — the voices built into Qwen3-TTS. The list is read from the node
  itself, so it stays right when the node is updated. **Hear it** generates a
  one-line sample.
- **Clone** — upload a clean 5–15 second clip and type what is said in it.
  Matching the reference text properly makes a large difference.
- **Design** — describe a voice in words. Needs the VoiceDesign model, which
  only exists as 1.7B; a designed line uses it whatever the model picker says.

### More options

- **Pause between lines** — silence inserted when the lines are joined.
- **Expressiveness** — sampling temperature. Higher wanders more.
- **Attention** — leave on `auto`. Installing `sageattention` or `flash_attn`
  makes generation two to three times faster.
- **Free GPU memory after each run** — for cards under 8 GB. Slower, because the
  model reloads each time.

### Takes and playback

Every run becomes a take. Lines are generated one at a time and joined into a
single wav with your pause between them, so the workspace shows progress line by
line and the block being spoken lights up as it plays. The player's ⏮ ⏭ skip
between lines rather than between takes. Download gives you the joined file, or a
zip of the clips if the format could not be joined.

## Engine and Models panels

**Engine** lists Python, Git, ComfyUI, the Qwen-TTS nodes, PyTorch, the node's
own packages, the models and the running engine. Anything missing has an Install
button; **Install everything missing** works through them in order. Output
streams into the activity log.

It also checks the version of `transformers`, which is the dependency that
actually breaks Qwen3-TTS: it needs **4.57.3, or 5.0 and up**. Anything else is
flagged.

**Models** is all the HuggingFace work: token for gated repos, mirror endpoint,
the four recommended folders with a button each, browsing any repo to see its
files and sizes before downloading, live progress with a Stop button, and
deleting a folder from disk. Downloads resume where they stopped.

---

## Troubleshooting

**"Nodes not loaded"** — ComfyUI is running but has not imported the Qwen-TTS
nodes. Restart ComfyUI. If it persists, look in the ComfyUI console for
`IMPORT FAILED`, which is almost always a missing requirement. Press Install on
**Qwen-TTS packages** in the Engine panel.

**transformers flagged as wrong** — install `transformers==4.57.3` or
`transformers>=5.0`. The Install button on Qwen-TTS packages does this from the
node's own requirements file.

**PyTorch will not install** — pick a build by hand in the Engine panel: CUDA
12.8 for recent NVIDIA drivers, 12.1 for older ones, ROCm for AMD, or CPU.

**Out of memory** — switch to the 0.6B model and turn on Free GPU memory after
each run.

**A line takes forever** — the first line after a restart loads the model, which
is slow. Later lines are much quicker unless memory freeing is on.

---

## Layout

```
run.sh, run.bat  Launchers — find Python, build .venv, start server.py
requirements.txt Flask and requests. That is the whole list.
tests/           Unit tests, a browser smoke test and stand-ins for
                 ComfyUI and HuggingFace so neither is needed to run them
server.py        Flask API — speech jobs, takes, setup, dependencies, HuggingFace
bootstrap.py     Python/ComfyUI/node discovery, installs, model snapshots, process
manager.py       Dependency checks and installers, HuggingFace browsing
comfy.py         Builds Qwen-TTS graphs from ComfyUI's live schema
web/index.html   The interface — one file, no build step
data/            config.json, takes.json, takes/   (created on first run)
```

Port: set `SCRIPT_BUILDER_PORT`. Set `SCRIPT_BUILDER_NO_BROWSER=1` to stop it
opening a tab. `SCRIPT_BUILDER_DATA` moves `data/`, which lets a second copy
run without touching the first one's takes.

---

## Tests

```bash
node tests/check.mjs     # everything compiles and the inline script parses
npm run test:units       # 43 unit tests, standard library only
npm install && npx playwright install chromium
npm test                 # 52 checks driving the real page in headless Chromium
```

None of it needs a GPU, a model download or the network: `tests/mock_comfy.py`
and `tests/mock_hf.py` stand in for ComfyUI and HuggingFace, and the suite runs
against a temporary data folder rather than your library. The same three run in
CI on every push and pull request.
