# Script Builder

A local text-to-speech studio. Write a script with named speakers, press Run,
and it is read on your own machine — no account, no upload, no per-word billing.

Two engines, switched from the picker on the Create page:

| | [Qwen3-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS) | [MOSS-TTS](https://github.com/richservo/comfyui-moss-tts) |
|---|---|---|
| Preset speakers | nine, read off the node | none — see below |
| Clone from a clip | yes | yes |
| Voice from a description | yes, 1.7B | yes, 1.7B VoiceGenerator |
| Smallest useful model | 0.9B | 1.7B, about 5 GB of VRAM |

The layout is the Script Builder file you already had: raw structure on the
left, dialogue blocks in the middle, speaker settings on the right. What changed
is underneath — the browser's own voices are gone, and every line is spoken by a
local model running through ComfyUI.

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
3. downloads the Qwen3-TTS weights into `ComfyUI/models/qwen-tts/`

Step 2 picks the interpreter ComfyUI actually runs on. On a portable install
that is `python_embeded\python.exe` — the same one your instructions name — and
on a managed install it is the virtual environment beside the ComfyUI folder.
Nothing goes into your system Python.

### Requirements

- Python 3.10 or newer (on Debian and Ubuntu, `python3-venv` too)
- Git
- An NVIDIA GPU with 8 GB or more runs everything downloaded by default:
  Qwen3-TTS, MOSS's 1.7B and MOSS-VoiceGenerator. Only the MOSS 8B is out of
  reach through these nodes, and it is left un-ticked. Less works with **Free
  GPU memory after each run** switched on. CPU works but is slow.

### Model folders

**The two engines do not share a folder shape**, and neither shape is a
preference — each is where that node looks. Qwen drops the organisation,
`models/qwen-tts/<Name>`, because its node lists `models/qwen-tts` one level
deep and downloads to `<Name>` (its README draws a `Qwen/` folder the code has
never looked in); MOSS flattens the slash, `models/moss-tts/<Org>--<Name>`,
because its loader builds that path from `repo_id.replace("/", "--")`. Put a
folder anywhere else and that node cannot see it — it quietly downloads a
second copy, or offline, fails the line. Folders an earlier version of Script
Builder left in `models/qwen-tts/Qwen/` are moved into place at launch.

#### Qwen3-TTS

All six repos in the [Qwen3-TTS collection](https://huggingface.co/collections/Qwen/qwen3-tts),
pulled into `ComfyUI/models/qwen-tts/`:

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

#### MOSS-TTS

Pulled into `ComfyUI/models/moss-tts/`:

| Folder | Size | VRAM | What it does |
|---|---|---|---|
| `OpenMOSS-Team--MOSS-Audio-Tokenizer` | codec | — | Shared codec; every MOSS model needs it |
| `OpenMOSS-Team--MOSS-TTS-Local-Transformer` | 1.7B | ~5 GB | Speech and cloning, and the fast one |
| `OpenMOSS-Team--MOSS-VoiceGenerator` | 1.7B | ~5 GB | Voice from a description |
| `OpenMOSS-Team--MOSS-TTS` | 8B | ~18 GB | Delay 8B — better, far slower |

The codec, the 1.7B and VoiceGenerator are downloaded by default: all three run
on an 8 GB card, and since MOSS has no preset speakers, describing a voice is
one of only two ways to pin one down.

**Sizes come from [OpenMOSS's own model table](https://github.com/OpenMOSS/MOSS-TTS#released-models),
not from the ComfyUI node's README**, which lists MOSS-VoiceGenerator as
"Delay 8B, ~18 GB". `MossTTSDelay` is the *architecture*; OpenMOSS publishes
VoiceGenerator at 1.7B. Taking the node README at its word had voice design
hidden behind a warning that it would not run on 8 GB, when it fits about as
comfortably as the base model.

The 8B is a tick rather than a default, and the reason is this node rather than
the model: it loads bf16 weights through `AutoModel.from_pretrained`, so 8B
really does want ~18 GB here. OpenMOSS's own llama.cpp path fits the 8B on an
8 GB card with Q4_K_M weights, staged loading and a quantized KV cache
(`configs/llama_cpp/trt-8gb.yaml`) — but the ComfyUI node implements none of
it, and that path is not a download. `GET /api/moss/8b` lists what it would
take, and two of the five cannot be fetched at any speed:

| Prerequisite | Can it be downloaded? |
|---|---|
| llama.cpp compiled from source, plus the C bridge | **No** — a build, not a package |
| `pip install -e ".[llama-cpp-onnx]"` from OpenMOSS/MOSS-TTS | Yes |
| `OpenMOSS-Team/MOSS-TTS-GGUF` (Q4_K_M + 33 embeddings + 33 LM heads) | Yes |
| `OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX` | Yes |
| TensorRT engines | **No** — OpenMOSS ship none; they are built against your GPU |

Add `?check=1` and the two HuggingFace repos are looked up rather than taken on
trust. Nothing in first launch assumes any of this is present.

### Two engines, two installs

Each engine gets its own everything, so neither can break the other:

| | Qwen3-TTS | MOSS-TTS |
|---|---|---|
| ComfyUI | `ComfyUI-Qwen3-TTS/` | `ComfyUI-MOSS-TTS/` |
| Python environment | `comfy-venv-ComfyUI-Qwen3-TTS/` | `comfy-venv-ComfyUI-MOSS-TTS/` |
| Nodes | `custom_nodes/ComfyUI-Qwen-TTS` | `custom_nodes/comfyui-moss-tts` |
| Models | `ComfyUI-Qwen3-TTS/models/qwen-tts/` | `ComfyUI-MOSS-TTS/models/moss-tts/` |
| Port | 8188 | 8189 |

First launch builds both. It costs disk — two ComfyUI clones and two PyTorch
installs, on the order of 10 GB before any models — and buys isolation: Qwen
wants `transformers` 4.57.3 or 5.0+, MOSS wants 4.40+, and nothing either node
pack pulls in can disturb the other.

**Only the engine you are using runs.** Two ComfyUIs that have both generated
each hold their models in their own process's video memory, and neither can
free the other's — on an 8 GB card the second one is what fails to allocate.
Switching engines stops one and starts the other, which costs a ComfyUI start.
If your card has room, turn on **Run both engines at once** and switching
becomes instant.

Upgrading from a single install? Your existing ComfyUI becomes Qwen's, keeping
its address, folder and models. MOSS then wants an install of its own.

### Does it actually work?

The Engine page has **Test Qwen3-TTS** and **Test MOSS-TTS**. Each generates one
line on your own machine and reports every step:

```
ok    ComfyUI is answering                 http://127.0.0.1:8188
ok    MOSS-TTS nodes are loaded
ok    Model folders are on disk            MOSS-Audio-Tokenizer 1.1 GB · MOSS-TTS-Local-Transformer 3.4 GB
ok    A graph can be built for it          MossTTSModelLoader → MossTTSGenerate → SaveAudioAdvanced
ok    ComfyUI accepts the graph            prompt 0620e2bc
ok    Speech comes back                    4.2s of audio, 24000 Hz, mono · 31s to generate · peak 62%
warn  Ran without reaching for the network ComfyUI fetched something while generating…
```

It stops at the first step that breaks and names it, which is the difference
between "it does not work" and "the weights in that folder never finished
downloading". Silence counts as a failure: a clip of the right length full of
zeros decodes perfectly and plays nothing.

The last step is worth watching on MOSS. The node only passes `codec_local_path`
to the TTSD model, so the 1.7B and VoiceGenerator resolve their audio tokenizer
through `AutoProcessor.from_pretrained` — if that reaches HuggingFace during
generation, this is where you will see it.

### Will it run on this card?

Every model carries a VRAM figure, and the app reads what the card actually
has — `nvidia-smi --query-gpu=memory.total`, or ComfyUI's `/system_stats` when
nvidia-smi is not on PATH. A model larger than the card is shown but greyed in
the picker, and the Models page asks before downloading it. On an 8 GB card
everything fetched by default runs: Qwen3-TTS in full, MOSS speech, MOSS
cloning and MOSS voice design. Only the MOSS 8B is out of reach.

Where the card cannot be read at all, nothing is hidden — an unknown card is
not assumed to be a small one.

The repo names come from the nodes' own source — `HF_MODEL_MAP` for Qwen,
`utils/constants.py` for MOSS — not from either README, which list fewer.

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
  one-line sample. On MOSS this button reads **Own voice** instead: MOSS has no
  speaker list at all, and with neither a clip nor a description it speaks in a
  voice of its own that changes with the seed.
- **Clone** — upload a clean 5–15 second clip and type what is said in it.
  Matching the reference text properly makes a large difference. Both engines
  do this.
- **Design** — describe a voice in words. On Qwen it needs the VoiceDesign
  model, which only exists as 1.7B; on MOSS it needs MOSS-VoiceGenerator. Either
  way a designed line loads that model whatever the model picker says, because
  it is the only one trained for it.

Switching engines keeps the script and the speaker names — only the voice
sources change, since the two engines do not offer the same ones.

### More options

- **Pause between lines** — silence inserted when the lines are joined.
- **Expressiveness** — sampling temperature. Higher wanders more. On MOSS it
  scales each model's own tuned temperature, so the resting value of 0.90 runs
  every MOSS checkpoint exactly as OpenMOSS tuned it.
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

### The engine console, Restart, and starting itself

**Engine console** is the panel under the buttons: the engine's own output,
live, with a line of app-side narration mixed into it for everything Script
Builder does *to* that engine — which process was holding the port, what the
system said when it was asked to stop, what started in its place. There is no
terminal behind a launcher, so this panel is the ComfyUI console.

The sentence above it names the state, including the two that otherwise look
like a healthy app that simply does not work:

- *a different ComfyUI is answering this address* — 8188 is the port every
  ComfyUI picks, so the one holding it is often somebody else's, and its
  missing nodes cannot be installed away;
- *everything is on disk and this engine cannot reach it* — ComfyUI reads
  `custom_nodes` once, at startup, so a node pack installed behind a running
  engine leaves a complete install with no classes in it.

**Restart ComfyUI** is the cure for both. It stops the engine it started and
starts it again — and where the one answering is not ours, it takes the
address over rather than giving up: ComfyUI-Manager's own reboot first, and
failing that the process holding the port is found, confirmed to look like a
ComfyUI, and closed, with one of this app's own started in its place. Anything
that is not a ComfyUI is named and left alone. A refusal says which obstacle
it hit — access denied, something supervising it, a process it could not
identify — instead of sending you to hunt a windowless python in Task Manager.

Launching the app does all of this by itself, with no button pressed: it
starts the engine if the address is quiet, adopts the one already running if
it is healthy, and replaces it through the same guard if it is not. A ComfyUI
you run yourself (external mode in Settings) is never touched — the app says
what is wrong with it and leaves it to you.

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
npm run test:units       # 131 unit tests, standard library only
npm install && npx playwright install chromium
npm test                 # 81 checks driving the real page in headless Chromium
```

None of it needs a GPU, a model download or the network: `tests/mock_comfy.py`
and `tests/mock_hf.py` stand in for ComfyUI and HuggingFace, and the suite runs
against a temporary data folder rather than your library. The same three run in
CI on every push and pull request.

### Making CI actually gate a merge

CI reports on its own; it only blocks anything once `main` requires those
checks. `.github/branch-protection.json` is that rule, ready to apply:

```bash
gh api -X PUT repos/<owner>/<repo>/branches/main/protection \
  --input .github/branch-protection.json
```

It requires the four checks and forbids force pushes and branch deletion, but
leaves admins able to push directly and asks for no reviews — a solo project
can still merge its own work.

The file is in the repo because GitHub accepts a required check whose name
matches no job and then silently gates nothing, so renaming a job would quietly
switch the gate off. `node tests/check.mjs` compares the two and fails if they
disagree.
