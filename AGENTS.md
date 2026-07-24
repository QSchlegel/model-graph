# AGENTS.md — model-graph

Guidance for AI agents (and humans) working in this repo. The deep map lives
in the [vault](vault/README.md) — **this file is the contract for keeping it
true**.

## What this is

A live model-internals inspector: a hooked PyTorch server streams per-token
internals (norms, heads, router flow, logit lens, attention sources,
projections) over an OpenAI-compatible API + websocket into two web UIs, with
a run-recording curator for testing/automation, in-browser WebGPU engines,
and an observable-ONNX export pipeline. Long-term goal: develop and improve
model reasoning capability, using this instrumentation as the measurement
harness.

## Run / verify

- Python env: `.venv/` (has torch+MPS via system-site-packages, transformers,
  aiohttp, onnx, onnxruntime). Always use `.venv/bin/python`.
- Dev servers: `.claude/launch.json` — `api-real` (full stack :8080 + ws
  :8765), `api-mock`, `ws-live`, `ws-replay`. Prefer preview-managed starts.
- **server.py or api_server.py changed → restart the server** (preview_stop +
  preview_start api-real; model reload ≈ 40 s). chat/dashboard HTML is served
  with `Cache-Control: no-cache` — a reload suffices.
- Test gates before calling work done:
  - `python harness.py all --model-dir <mlx-dir> --fallback-hub
    LiquidAI/LFM2.5-1.2B-Instruct --device mps --tokens 24` (pipeline sanity)
  - `python curator.py suite suites/smoke.json` and `suites/tools.json`
    (behavioral + internals checks; exit code ≠ 0 = fail)
  - agent harness / tools / benchmark changed → `python test_agent.py` (tool
    golden values = JS↔Python registry parity, matcher, scoring; exit ≠ 0 =
    fail). Full agentic benchmark: `python bench.py suites/agentic.json
    --models lfm=http://localhost:8080#<model>` (needs a running api_server).
  - UI changes: verify in the browser pane (screenshot / read_page), not by
    assertion.

## Keep the vault current (the actual contract)

When you change...                     → update...
- ndjson protocol (server.py docstring) → `vault/protocol.md` + README
  protocol block + check consumers: dashboard normFrame, chat onProto,
  harness verify, curator checks, Hub replay
- viz modes / interactions (chat.html)  → in-app explainer (`#explain`),
  `vault/components.md#chat-ui`, README "Reading the chat viz"
- API endpoints (api_server.py)         → `vault/components.md#api-server`,
  `vault/flows.md` sequence diagrams
- engines (chat.html engine select)     → `vault/engines.md` capability matrix
- curator checks / suite schema         → `vault/components.md#curator`,
  suites/ examples
- agent harness / tools / prompt         → keep `agent_harness.py` (Python) and
  `web/agent.html` + `web/chat.html` (JS) in lockstep; `test_agent.py` golden
  values gate the parity; update `vault/agent-harness.md`
- benchmark suite / metrics / scoring    → `suites/agentic.json`, `bench.py`,
  `vault/agent-benchmark.md` (keep the AGENTIC weights + gate thresholds true)
- site navigation / menu links           → `web/nav.js` ONLY (shared command
  menu, one source of truth, served at `/nav.js`; every page includes it) +
  `vault/components.md#nav` — do not re-add per-page nav `<a>` lists
- architecture (new component/port)     → `vault/architecture.md` graphs +
  `vault/README.md` index + `.claude/launch.json` if runnable
- research findings / model choices     → `vault/research-tiny-models.md`,
  `vault/roadmap.md`
- product story / features / RYO steps  → `web/landing.html` (mock views +
  quickstart must match reality), `vault/six-pager.md`, `CONTRIBUTING.md`

Rule of thumb: **if a diagram or matrix in the vault would now lie, fix it in
the same change.** New markdown notes go under `vault/`, linked from
`vault/README.md`.

Vault conventions (it is opened as an Obsidian vault): every note carries
YAML frontmatter (`tags` from: map/diagram/component/reference/research/
roadmap — the graph view colors by these — plus `updated: YYYY-MM-DD`);
callouts use the GitHub-compatible subset (`> [!note|tip|important|warning]`)
so notes render well in both; links are relative markdown links (work in
Obsidian and GitHub); diagrams are mermaid fenced blocks — validate syntax
when editing (render them; don't guess). `vault/.obsidian/` ships graph
color groups + app defaults; `workspace*` is gitignored user state.

## Hard-won gotchas (do not relearn these)

- transformers 5.x: `apply_chat_template(..., return_tensors=...)` returns a
  BatchEncoding — take `.input_ids`.
- Hybrid models (LFM2) **compact** `output_attentions` — align via layers
  whose head-hooks fired, never by tuple index.
- Attention maps require `attn_implementation="eager"` at load.
- HF `output_hidden_states` final entry is post-final-norm; hooks are the
  ground truth for residuals.
- MLX quantized folders (LM Studio) are NOT HF-loadable — harness falls back
  to hub bf16 weights.
- VL models: decoder layers live at varying paths; HookedModel resolves
  `model.layers` / `model.language_model.layers` / etc.
- Canvas needs explicit CSS width/height (attribute feedback loop otherwise)
  and a ResizeObserver; author `display:flex` beats the UA `[hidden]` rule —
  add `#el[hidden]{display:none}`.
- Browser-pane synthetic keys/clicks are unreliable — verify handlers via
  dispatched events; real input works.
- The ws Hub replays the last run to new clients — tests that count frames
  must account for the greeting.
- LFM2 tool calls are pythonic between `<|tool_call_start|>` markers —
  `server.parse_tool_calls` handles pythonic + JSON.

## Style

- Python: stdlib-lean, ~80 cols, comments only for non-obvious constraints.
- JS/HTML: single-file, compact style matching web/chat.html; palette vars in
  `:root`; canvas drawing over DOM where the viz is concerned; textContent
  over innerHTML.
- Honesty rule for the viz: never render data that wasn't measured — show an
  explicit "not captured / needs X" note instead.
