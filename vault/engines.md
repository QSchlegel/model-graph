---
tags: [reference, engines]
updated: 2026-07-20
---

# Engines — capability matrix

Selected in the chat header ([chat-ui](components.md#chat-ui)).
Availability-aware viz: the metric cycle and explainer cards only offer
modes the current run's data supports.

| capability | server api | browser (stock) | browser (observable) |
|---|---|---|---|
| chat (SSE streaming) | ✓ | ✓ | ✓ |
| tool calling (OpenAI `tools`) | ✓ | — | — |
| multimodal (image parts) | ✓ (VL models) | — | — |
| token strip + run boundaries | ✓ | ✓ | ✓ |
| top-k predictions | ✓ | ✓ (logits tap) | ✓ |
| entropy / margin | ✓ | — | — |
| mlp / attn write norms | ✓ | — | — |
| per-head norms + fingerprint cells | ✓ | — | — |
| attention sources + arcs | ✓ | — | — |
| resid norms heatmap | ✓ | — | ✓ |
| logit lens (ladder, dots) | ✓ | — | — |
| embedding trajectory + context trace | ✓ | — | ✓ |
| router flow (MoE) | ✓ | — | — |
| curator auto-recording | ✓ | — | — |

Engine notes
- **server api** — hooked python model (`api-real` launch config). The only
  engine with full internals; required for reasoning-analysis work.
- **browser stock** — SmolLM2-135M/360M, Qwen2.5-0.5B via transformers.js,
  weights from huggingface.co, WebGPU (wasm fallback). ONNX graphs expose
  only logits → chat + predictions only.
- **browser observable** — `export_observable.py` artifact served from
  `/models/observable/`; patched `model.forward` reads `hidden.*` →
  resid norms, trajectory + context, predictions computed client-side.
  Extending mlp/attn/head coverage = more surgeried outputs (roadmap).
> [!tip] Adding an engine
> Option in the chat `engine` select + a `loadLocal` branch — then update
> this matrix and the in-app explainer's engines section (per
> [AGENTS.md](../AGENTS.md)).

Model notes (local defaults)
- Server: `LiquidAI/LFM2.5-1.2B-Instruct` (hybrid conv/attn, tool-use
  trained). VL verified: `LiquidAI/LFM2-VL-450M`.
- Observable export default: `HuggingFaceTB/SmolLM2-135M-Instruct` q4f16.
