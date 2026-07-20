# Contributing to model-graph

Thanks for helping build an honest window into running language models.
Contributions from humans and AI agents are both welcome — agents should
read [AGENTS.md](AGENTS.md) first; humans will find it useful too.

## Ground rules

1. **Honest pixels.** The viz never renders data that wasn't measured.
   If a signal is unavailable, show an explicit note — never interpolate,
   never fake. This is the product's core promise.
2. **Keep the vault true.** [vault/](vault/README.md) holds the
   architecture/flow diagrams, protocol, and engine matrix. If your change
   makes a diagram or table lie, fix it in the same PR
   (the change→doc matrix in [AGENTS.md](AGENTS.md) says exactly what).
3. **Protocol changes fan out.** The ndjson protocol has five consumers
   (dashboard, chat, harness, curator, Hub replay) — touch all of them or
   don't touch the protocol.

## Dev setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python api_server.py --mock          # full UI, no GPU/weights
# real model (Apple silicon example):
.venv/bin/python api_server.py --model LiquidAI/LFM2.5-1.2B-Instruct --device mps
```

UIs: `http://localhost:8080` (landing) · `/chat` · `/dashboard`.
Server code changed → restart the process; HTML/JS is served no-cache.

## Before you open a PR

```bash
# pipeline sanity (capture + verify a real trace)
.venv/bin/python harness.py all --model-dir <any-local-model-dir> \
  --fallback-hub LiquidAI/LFM2.5-1.2B-Instruct --tokens 24

# behavioral + internals gates (exit code matters)
.venv/bin/python curator.py suite suites/smoke.json
.venv/bin/python curator.py suite suites/tools.json
```

UI changes: verify in a real browser (both light pages and mobile width),
and update the in-app `?` explainer if behavior changed.

## Good first contributions (roughly ascending effort)

- **Suite cases & curator checks** — no ML background needed; add
  reasoning/tool cases to `suites/`, or a new check in `curator.py`.
- **Curated browser engines** — add a transformers.js-compatible model to
  the chat engine select (+ engines matrix row).
- **Dashboard consumers** — the dashboard doesn't yet render attention
  sources or entropy (chat does); ports welcome.
- **Observable export coverage** — extend `export_observable.py` to
  surgery mlp/attn block outputs (see vault/roadmap).
- **New architecture support** — a hooks + topology mapping for a model
  family (see `HookedModel._register_hooks` and the layer-path resolver).

## Style

- Python: stdlib-lean, ~80 columns, comments only where the code can't
  speak (constraints, gotchas).
- Web: single-file pages, shared `:root` palette, `textContent` over
  `innerHTML`, canvas for the viz. Match `web/chat.html` idiom.
- Commits/PRs: state what changed *and* what you verified (paste harness/
  suite output). Attach a curator run id for behavior claims.

## Reporting issues

Best issues come with a run id: reproduce with the server engine, then
`python curator.py list` and attach the `runs/<id>.ndjson` (it contains
prompt + full internals — scrub anything private).
