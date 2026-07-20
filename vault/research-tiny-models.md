---
tags: [research]
updated: 2026-07-20
---

# Research — tiny models & long context (as of 2026-07-20)

Feeds model choices in [engines](engines.md) and next steps in
[roadmap](roadmap.md).

## Tiny-model shortlist

| model | size | context | notes |
|---|---|---|---|
| Qwen3.5-0.8B (2026-03) | 0.8B | **262K native** | hybrid Gated-Delta + sparse MoE, native early-fusion multimodal, function calling, 201 languages, <2 GB VRAM — strongest all-round tiny model |
| LFM2.5-1.2B (current server default) | 1.2B | 32K | hybrid conv/attn, tool-use trained, our fully instrumented reference |
| SmolLM2-135M/360M | 0.135–0.36B | 8K | our browser/observable workhorses; ONNX everywhere |
| SmolVLM-256M | 0.256B | — | <1 GB VLM, beats 300×-larger Idefics-80B |
| LFM2-VL-450M | 0.45B | 32K | verified multimodal + full hooks in this stack |
| Llama-3.2-1B | 1B | 128K | solid baseline, wide ecosystem |
| RWKV-7 "G1" / Mamba-3 small | 0.4–3B | streaming | constant-memory state; unbounded length ≠ unbounded recall |

## The 1M-context question

> [!important] Short answer
> **No tiny (<1B) model ships a trained 1M window.** Smallest credible
> 1M-capable model: RWKV-X-3.6B. Closest honest tiny long-context:
> Qwen3.5-0.8B at 262K native.

Current reality:

- True 1M+ advertised windows live in big models: GLM-5.2 (754B MoE, DSA
  sparse attention), Qwen3.5/3.6 large MoEs, DeepSeek V4, Llama-4 Scout
  (10M claimed).
- Smallest credible 1M-capable model: **RWKV-X-3.6B** — hybrid sparse
  attention + recurrence, stable decode demonstrated at 1M tokens.
- Linear/SSM tiny models (RWKV-7, Mamba-3; LFM2's conv layers are the same
  family) stream arbitrarily long with constant memory, but effective
  recall is far shorter than "unlimited".
- Effective ≠ advertised everywhere: RULER-style testing puts most models
  at 50–65% usable window.
- Closest honest tiny long-context: **Qwen3.5-0.8B at 262K**.

Implication for this project: the architecture class that wins
context-per-parameter (hybrid linear/attention) is exactly what the
inspector already instruments (conv vs attention division-of-labour views).

Sources: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) ·
[Qwen3.5 family](https://enclaveai.app/blog/2026/03/08/qwen-3-5-complete-model-family-local-ai/) ·
[context comparison](https://www.morphllm.com/llm-context-window-comparison) ·
[GLM-5.2](https://www.mindstudio.ai/blog/what-is-glm-5-2-open-weight-model-agentic-workflows) ·
[RWKV-X](https://arxiv.org/html/2504.21463v2) ·
[SSM showdown](https://algorithmine.com/learn/mamba-rwkv-vs-transformers-2026) ·
[SmolVLM](https://arxiv.org/abs/2504.05299)
