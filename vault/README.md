---
aliases: [Home, Index]
tags: [map]
updated: 2026-07-20
---

# model-graph vault

The living map of the project. Maintenance contract: [AGENTS.md](../AGENTS.md).

> [!tip] New here?
> Read the orientation paragraph below, then [architecture](architecture.md)
> for the graphs and [flows](flows.md) for how a request actually moves.

## Notes

- [architecture](architecture.md) — component graph + data-flow graph
- [flows](flows.md) — user & component sequence flows (chat, engines, tools,
  deep-link handoff, curator)
- [components](components.md) — per-component reference (server, api, UIs,
  harness, curator, exports)
- [protocol](protocol.md) — the v1 ndjson/websocket protocol
- [engines](engines.md) — engine capability matrix (server / browser /
  observable / VL)
- [agent-harness](agent-harness.md) — the in-browser state machine that drives
  tiny models through a tool-use loop (`/agent` + chat agent mode)
- [agent-benchmark](agent-benchmark.md) — the agentic tool-use benchmark
  (`agent_harness.py` + `bench.py` + `suites/agentic.json`) to fine-tune +
  rank models on a weighted AGENTIC score
- [research-tiny-models](research-tiny-models.md) — tiny-model landscape,
  long-context findings (mid-2026)
- [roadmap](roadmap.md) — the reasoning program and open threads
- [six-pager](six-pager.md) — the open-source product narrative
  (problem, tenets, product, CX, architecture, OSS plan)

## One-paragraph orientation

`server.py` wraps a HF model in forward hooks (eager attention) and yields a
per-token stream of internals. `api_server.py` fronts it with an
OpenAI-compatible API (+ tool calling, + multimodal), broadcasts every run
over ws :8765 (with last-run replay), records runs via `curator.py`, and
serves the two UIs: `web/chat.html` (chat with the full visualization as its
background) and `web/dashboard.html` (block-level drill-down). Browser-side
engines run curated models on WebGPU via transformers.js; the
`export_observable.py` pipeline re-exports ONNX models with residual-stream
outputs so even in-browser runs are inspectable. `harness.py` validates the
capture pipeline; suites under `suites/` make behavior + internals testable. A
client-side **agent harness** (`web/agent.html` at `/agent`, plus an agent-mode
toggle in the chat) drives curated browser models through a state-machine
tool-use loop — [agent-harness](agent-harness.md).

## Ports & artifacts

| thing | where |
|---|---|
| HTTP (API, UIs, /models/, /curator/) | :8080 |
| websocket internals stream | :8765 |
| run recordings | `runs/` (gitignored) |
| observable ONNX exports | `models/observable/` (gitignored) |
| launch configs | `.claude/launch.json` |
