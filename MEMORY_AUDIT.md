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

## Validation

Validation gate passed. Full Python suite: **326 of 330 tests passed**.
Four external-process identification/takeover tests fail in this runner;
the same four failed against unchanged code before these fixes. They are
not reported as passes. The new cache-cleanup and take-lifecycle regression
checks passed, as did all 28 focused lifecycle/torch-isolation checks.

No CUDA synthesis was run. The shim covers Qwen, not MOSS's private cache;
switching engines still stops the previously managed engine. MOSS 8B remains
unsupported through this node on an 8 GB GPU.

## Apply

Update the app, then restart Qwen's ComfyUI from the Engine page. Keep
**Free GPU memory after each run** on, use Qwen 0.6B, and try one short line
with other AI engines closed. For remote ComfyUI, copy
`compat/script_builder_memory` into its `custom_nodes` directory and restart
it. Capture the final console error, GPU model, system RAM and line length
if synthesis still fails.
