# 8 GB memory audit — 9 October 2026

## Fixed

- Qwen's cache-hit path returns before updating its unload callback. Loading
  the model on line one without unloading could make the final line's unload
  request ineffective. Its private model cache also sits outside ComfyUI's
  model registry, so the existing `/free` recovery did not reach it.
- A small custom node now connects ComfyUI's unload operation to Qwen's
  private cache. It resolves registered classes at execution time, so node
  import order does not matter, and leaves unrelated node caches alone.
- A requested release now runs explicitly after the take; failures and
  cancellation also request cleanup. Successful intermediate lines still
  reuse the model. Managed Qwen startup/restart installs the custom node.
- Synthetic torch-install tests are isolated from the test host's actual
  installed torch, avoiding false reports of two installed versions.
- Second pass: MOSS also retains its model and processor in module globals.
  The hook now offloads the model and audio tokenizer before clearing those
  globals, including when a ComfyUI output-cache tuple still references them.
  Startup/restart installs the hook for both engines, and the free-memory
  switch is available for MOSS as well.
- Engines still running without the updated hook now raise an Engine badge
  and explain how to restart or install it remotely. A versioned node marker
  distinguishes the earlier Qwen-only hook; this warning does not block a
  remote engine from generating.
- Cancelling between lines or during the final clip download no longer
  queues another line or publishes a cancelled take. A timed-out line now
  sends an interrupt for its own prompt instead of leaving it running.

## Validation

The second-pass validation gate and **18 focused memory/lifecycle/readiness
tests** pass locally. The full **338-test suite passed on Python 3.10 and
3.13 in GitHub Actions, with one intentional skip per version**
([run 37993823099](https://github.com/stevebarrettsrha-ops/Text-to-Speech-Model/actions/runs/37993823099)).
That run also passed validation and the browser smoke test. The four process-identification/
takeover failures observed locally on both unchanged code and the first patch
passed on the clean GitHub runners.

No CUDA synthesis was run. MOSS offloading needs system RAM, and neither cache
cleanup nor the published model-size estimates prove that an arbitrary long
line fits an 8 GB GPU. Switching engines still stops the previously managed
engine. MOSS 8B remains unsupported through this node on an 8 GB GPU.

## Apply

Update the app, then restart each engine's ComfyUI from the Engine page. Keep
**Free GPU memory after each run** on, use Qwen 0.6B, and try one short line
with other AI engines closed. For remote ComfyUI, copy
`compat/script_builder_memory` into its `custom_nodes` directory and restart
it. Capture the final console error, GPU model, system RAM and line length
if synthesis still fails.
