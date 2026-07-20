---
tags: [roadmap]
updated: 2026-07-20
---

# Roadmap — the reasoning program

Goal: develop and improve model reasoning capability, with this stack as the
measurement + inspection harness. Model landscape:
[research-tiny-models](research-tiny-models.md) · harness pieces:
[components](components.md)

> [!important] The loop
> **measure** (curator suites) → **inspect** (viz/internals on recorded
> failures) → **iterate** (prompts, models, later fine-tuning) — every
> failure arrives pre-instrumented.

## Now possible (built)

- Behavioral gates: `suites/smoke.json`, `suites/tools.json` (should-call /
  args / selection / abstain / grounded chains) with exit codes + failure
  events. Baseline: LFM2.5-1.2B = 4/4 tool reasoning.
- Reasoning-relevant signals per token: entropy, top-1 margin, lens settle
  depth, attention sources/spread — all recorded per run.
- First captured reasoning failure: "3+2−1 apples → 6" with attention mass
  on the `<|im_start|>` sink instead of the numbers.

## Next steps (ordered)

1. **Reasoning suite v1** — arithmetic / multi-step logic prompt battery with
   expectations on answers AND internals (entropy ceilings at answer tokens,
   settle-depth bounds); baseline LFM2.5 vs Qwen3.5-0.8B.
2. **Harder tool cases** — tempting-but-wrong tools, multi-hop chains
   (call → result → second call), parallel calls, noisy tool results.
3. **Failure atlas** — curate recorded failure runs in the vault (run ids +
   what the internals showed); look for signatures (sink-attention, late
   settle, low margin) that predict wrong answers.
4. **Compare-driven eval** — `curator compare` between prompt variants
   (with/without CoT) to quantify what step-by-step changes internally.
5. **Observable export v2** — surgery mlp/attn block outputs into the ONNX
   export so browser runs get write-norm views too.
6. **Logit lens top-k** — extend lens to per-layer top-k (protocol allows).
7. **Fine-tuning loop** (later) — once failure signatures are stable, use
   suites as before/after gates for LoRA/DPO experiments on the tiny models.

## Open threads

- Attention matrices per head (full maps) — heavy; only if head-level
  reasoning analysis demands it.
- Dashboard: consume attn_srcs/entropy (currently chat-only).
- Curator: golden-run pinning per model revision.
