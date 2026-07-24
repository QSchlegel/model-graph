---
tags: [component, reference]
updated: 2026-07-21
---

# Components

How they connect: [architecture](architecture.md) · in motion:
[flows](flows.md)

> [!important] Anchors are load-bearing
> Section names below are referenced from [AGENTS.md](../AGENTS.md) — keep
> them stable when editing.

## server

`server.py` — the ground-truth instrument. `HookedModel` loads any HF causal
or vision-language model (auto-detected via `vision_config`; decoder layers
resolved across `model.layers` / `model.language_model.layers` / …) with
**eager attention**, registers forward hooks on token-mixer / channel-mixer /
layer / o_proj (heads) / router (MoE) / dt_proj (SSM), and
`generate_stream()` yields the v1 protocol per token: norms, head norms,
expert weights, logit lens (final_norm+lm_head over per-layer residuals),
seeded 3D projections, top-k, entropy/margin, attention sources
(mean-over-heads top-8 + spread; hybrid-compacted tuples realigned via head
hooks). Prefill emits the full context trace. `parse_tool_calls()` converts
LFM2-pythonic / JSON tool output to structured calls. Standalone: ws server
on :8765.

## api-server

`api_server.py` — the front door on :8080. OpenAI-compatible
`/v1/chat/completions` (SSE + non-stream, `tools`, image content parts via
`to_hf_messages`, temperature, usage, `finish_reason` incl. `tool_calls`),
`/v1/models`, CORS everywhere. `Hub` broadcasts every run over ws :8765 and
**replays the last run** to new clients. Serves the UIs (no-cache),
`/models/` (observable exports), `/curator/*` REST, and the static pages:
`/` (landing incl. the step-by-step walkthrough), `/intro` (transformers
intro along the papers, `web/intro.html`), `/six-pager`, `/blog/`,
sitemap/robots. Auto-records every run via curator (`--no-curate` off
switch). `--mock` = full-featured fake backend for wiring tests.

## chat-ui

`web/chat.html` — chat window whose background IS the visualization.
Modes: mlp / attn (per-head subdivided cells) / resid heatmap + embed
(3D residual trajectory, blue context trace, per-layer minimap, zoom,
labels auto/all/off). Interactions: hover tooltips (norms, lens, looks-at,
head, entropy), click → cut-out panel (norms, attention sources, lens
ladder, heads/router/Δ, sparkline, predictions) with arrow-key navigation,
attention arcs over the token strip, availability-aware metric cycling,
lens flip dots, run boundaries, drag/pinch/touch, `?` explainer (the
in-app documentation — keep it truthful). Engine select: server api /
browser models / observable (see [engines](engines.md)). Loading bar with
race-safe model switching. An **agent-mode** toggle (header) wraps generation
in the tool-use loop with a compact state-pill strip — see
[agent-harness](agent-harness.md#two-surfaces).

## agent-harness

`web/agent.html` (served at `/agent`) + the chat's agent-mode toggle — a
client-side **finite state machine** that drives a tiny model through a bounded
ReAct tool-use loop (THINK → ROUTE → ACT → OBSERVE → ANSWER/GUARD/STOP),
entirely in the browser. `/agent` is the guided education page: pick a micro
model (WebGPU via transformers.js) → pick a task + tools → watch the
state-machine graph, transcript, trace tape and per-state explainer, with a
scripted-demo fallback for no-WebGPU/no-download. Local tools run in the page
(real: calculator/convert/clock/random_int; sandbox-labelled: weather/lookup).
Full design, states, tools and parsing: [agent-harness](agent-harness.md).

## dashboard

`web/dashboard.html` — generic block-registry drill-down (run › layer ›
block › part), mock presets, trace loading, ws live mode, deep-link
bootstrap (`?ws=1&layer&block&t`) that auto-connects, unfolds, seeks.

## nav

`web/nav.js` — the **single source of truth for site navigation**, loaded via
`<script src="/nav.js" defer>` on every page (served by api_server at `/nav.js`).
Every header shows just the brand + one **☰ menu** button (kept uncluttered);
the button — or ⌘K / `/` — opens a grouped, searchable, keyboard-driven command
palette (Tools / Learn / Source, type-to-filter, ↑↓/↵) that marks the current
page. On marketing headers the button replaces the old inline `<nav>` link list;
on dense app headers (chat/dashboard) it drops into the control row. Edit the
link groups in `nav.js`, not seven page headers. The dashboard's `nav#crumb`
(in-app run/layer breadcrumb) is left untouched.

## harness

`harness.py` — pipeline test: `validate` a local model folder (incl. MLX
non-loadable detection), `capture` a reference trace (hub fallback),
`verify` trace invariants against config (message shape, finiteness, depth
trend, layer kinds, head counts). `all` = the gate.

## curator

`curator.py` — hook-fed context curator. Server-side auto-recording of every
run (`runs/` + index). CLI: list/show/test/compare/run/suite. Checks:
finite_norms, frames_consistent, depth_trend, eos_terminated, settle_depth,
confidence, uncertainty, attention_focus, repetition, throughput. Suites:
prompt cases with expectations (contains / eos / max_settle_mean) and
tool-reasoning cases (should-call, args_include, tool selection, abstain,
result-grounded chains) — exit-code CI gate, `--on-fail` shell hook,
`on("run_saved"|"test_fail"|"test_done")` python events. REST mirror under
`/curator/`.

## export-observable

`export_observable.py` — ONNX graph surgery: exposes every layer's residual
stream (the `input_skip_bias_sum` outputs of fused SkipLayerNorm nodes, final
layer's optional output declared) as fp32 `hidden.N` graph outputs; ships
tokenizer/config + the seeded projection basis; validates vs PyTorch hooks
(argmax parity, norm drift ≤ q4f16 noise). Output served at
`/models/observable/` and consumed by the observable browser engine.

## replay & mocks

`replay.py` streams any recorded ndjson trace over ws (rate, loop).
`make_mock.py` generates protocol-correct mock traces; `--mock` in
api_server fakes the full pipeline including lens/context/proj.
