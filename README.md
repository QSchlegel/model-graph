# model-graph

> Project map: [vault/](vault/README.md) (architecture + flow graphs,
> components, protocol, engines, research, roadmap) · agent maintenance
> contract: [AGENTS.md](AGENTS.md) · product story:
> [six-pager](vault/six-pager.md)
> ([📄 PDF](docs/model-graph-six-pager.pdf)) ·
> [CONTRIBUTING](CONTRIBUTING.md) · MIT

Deploy note: the included [Dockerfile](Dockerfile) ships the demo variant —
mock backend, torch-free (~60 MB image), gzip + caching + single-port ws
(`/ws`) to minimize PaaS egress; browser engines fetch weights from
huggingface.co directly so the service serves no model bytes.

Runtime node-graph visualizer for LLM internals. Streams per-token, per-layer
activation stats (norms, attention heads, MoE routing, SSM selectivity) from a
hooked HF model over a websocket into a live browser dashboard. Works with
transformers, hybrid conv/attention models (LFM2.5), Mamba-style SSMs, and MoE.

## Layout

    server.py           hooked HF model → NDJSON protocol over ws://localhost:8765
    api_server.py       OpenAI-compatible /v1/chat/completions in front of the
                        hooked model; broadcasts internals to the dashboard ws
                        and serves the web UIs over http (/, /chat, /dashboard,
                        /agent, /intro, /six-pager, /blog)
    export_observable.py  build an *observable* ONNX model: graph-surgery every
                        layer's residual stream (already computed inside the
                        fused SkipLayerNorm nodes) into declared outputs
                        hidden.0..N-1, copy tokenizer/config, ship the seeded
                        projection basis, validate vs PyTorch hooks (argmax
                        parity + norm drift). Output: models/observable/<name>/
                        (gitignored, served by api_server at /models/)
    curator.py          hook-based context curator: api_server records every
                        run (prompt + params + full internals trace) under
                        runs/ (gitignored; --no-curate disables). CLI:
                        list / show / test (8 built-in checks over internals:
                        finite norms, depth trend, settle depth, confidence,
                        repetition, ...) / compare (token diff + per-layer
                        norm drift) / run (drive live api, then test) /
                        suite <file> --on-fail CMD (prompt batteries with
                        expectations; tool-reasoning cases assert should-call
                        / should-abstain / tool selection / argument values /
                        result-grounded final answers — see suites/tools.json).
                        REST: GET /curator/runs[/{id}],
                        POST /curator/runs/{id}/test. Python automation:
                        Curator().on("run_saved"|"test_fail"|"test_done", fn)
    replay.py           stream a recorded .ndjson trace over the same websocket
    make_mock.py        generate realistic mock traces (v1 protocol)
    harness.py          test harness: validate model folder / capture / verify trace
    agent_harness.py    the ReAct agent harness in Python (faithful port of
                        web/agent.html): local tools, tolerant parser (ReAct +
                        native tool_calls), bounded loop; drives any
                        OpenAI-compatible model. run_agent(base,model,task,tools)
    bench.py            agentic tool-use benchmark: run the harness over
                        suites/agentic.json against one or more models, score a
                        weighted AGENTIC (0-100), emit a leaderboard + report
                        (reports/, gitignored). fine-tune before/after gate +
                        model ranking. `python bench.py suites/agentic.json
                        --models lfm=http://localhost:8080#LiquidAI/LFM2.5-1.2B-Instruct`
    suites/agentic.json 34 graded agentic cases (11 categories); golds are exact
                        tool outputs, locked by test_agent.py
    test_agent.py       CI gate: tool golden values (JS↔Python registry parity),
                        whole-token matcher, scoring logic. `python test_agent.py`
    web/dashboard.html  THE app — generic block-registry dashboard (mock presets,
                        trace loading, live ws, drill-down run›layer›block›part)
    web/chat.html       chat window on the OpenAI endpoint (SSE streaming,
                        multi-turn, tok/s); background is a live explorable
                        layers×tokens heatmap fed by the ws stream — token
                        strip with predicted tokens, hover to inspect (norms +
                        top-k), click a cell for a dashboard cut-out (logit-
                        lens ladder with per-layer flips + settle layer, head
                        norms / router flow + expert utilization / conv info /
                        Δ / prediction bars / per-layer sparkline) navigable
                        with arrow keys; heatmap dots mark lens flips (red =
                        settles on final token), empty click cycles metric,
                        drag pans, dashed request bounds
    web/agent.html      /agent — in-browser agent state machine: a ReAct
                        tool-use loop over a curated micro model on WebGPU,
                        with a live state-machine graph, transcript, trace tape,
                        per-state explainer and a scripted-demo fallback (no
                        GPU / no download). chat.html has an agent-mode toggle
                        running the same loop with the full internals viz.
    web/demo.html       earlier standalone in-chat demo (self-contained mock)
    web/graph.html      earlier minimal ws client (spine graph only)
    traces/             real LFM2.5-1.2B traces (verified) + mock traces

## Quickstart

    pip install -r requirements.txt

    # mock, no model needed
    python server.py --mock            # or --mock-moe
    open web/dashboard.html            # header → ws

    # real model, live
    python server.py --model LiquidAI/LFM2.5-1.2B-Instruct --device mps

    # replay a recorded trace
    python replay.py traces/lfm25_trace.ndjson --rate 8 --loop
    # or: dashboard header → trace → pick a .ndjson

    # test pipeline against a local model folder (e.g. LM Studio download)
    python harness.py all --model-dir <dir> \
      --fallback-hub LiquidAI/LFM2.5-1.2B-Instruct --device mps --tokens 24

    # OpenAI-compatible API + chat UI + live dashboard, all in one process
    python api_server.py --model LiquidAI/LFM2.5-1.2B-Instruct --device mps
    # tool calling: pass OpenAI `tools`; LFM2-style pythonic calls are parsed
    #   into tool_calls + finish_reason "tool_calls"; role:"tool" and
    #   assistant tool_calls messages round-trip through the chat template
    # multimodal: point --model at a VL model (e.g. LiquidAI/LFM2-VL-450M);
    #   OpenAI image_url content parts (data:/http) are decoded and the
    #   hooks attach to the language tower — full viz for image chats
    # → chat http://localhost:8080/  dashboard http://localhost:8080/dashboard
    #   any OpenAI client: base_url http://localhost:8080/v1, any api_key
    # wiring test without weights: python api_server.py --mock

## Protocol

NDJSON messages. v1 (emitted by server.py, auto-upgraded by the dashboard):

    {"type":"topology","model":str,"layers":[{"i":int,
        "attn":{"kind":"local"|"global"|"conv"|"attn","heads":int},
        "mlp":{"kind":"dense"|"moe","experts":int,"topk":int}}]}
    {"type":"token","t":int,"text":str,
        "topk":[[text,prob],...],              # final-logits top-5 predictions
        "layers":[{"i":int,
        "attn_norm":float,"mlp_norm":float,"resid_norm":float,
        "head_norms":[float,...],
        "expert_weights":[[e,w],...],          # MoE only
        "delta_mean":float,"delta":[float,...] # SSM only (dt_proj hook)
    }]}                                        # + "eos":true on the eos token
    {"type":"done"}                            # server adds prompt/completion_tokens

v2 (native dashboard schema, adapter in dashboard.html `normTopo`/`normFrame`):
layers carry ordered typed blocks `{id,kind,subkind,heads/experts/topk}`;
frames carry `{i,resid,blocks:{id:{norm,...metrics}}}`. Unknown kinds render
via a default card + raw metric dump.

## Reading the chat viz (also: `?` button in the chat header)

Rows = layers, columns = generated tokens; each cell is the L2 norm of a
block's output for that token — how strongly it wrote into the residual
stream. Colors normalize per row against the layer's running max (absolute
norms grow with depth, raw values would drown early layers) — compare along
a row, not down a column.

- `mlp`   — channel-mixer write. MLPs hold most params and act like
            key→value memories; best single proxy for "knowledge injected
            here". Look for content-word bursts, hot mid-depth rows.
- `attn`  — token-mixer write (short-conv on C rows). The only place
            information moves between positions; spikes = the token pulled
            hard on context. Compare A vs C rows for the hybrid's division
            of labour. A cells subdivide into per-head panels (normalized
            within the cell = which heads) framed by a border carrying the
            aggregate attn norm (= how much); hover resolves the head.
- `resid` — residual stream norm after the layer. Growth with depth is
            expected; the signal is deviation (jumps, flat columns, eos).
- `embed` — residual trajectory: each token's hidden state projected
            through a fixed random orthonormal 3D basis (seeded — stable
            while streaming, unlike PCA) and connected in generation
            order; the blue dotted trace is the full prompt/context from
            the prefill pass, linked into t=0. Right-edge layer minimap
            shows every layer's trajectory (click to switch depth). Drag
            rotates, wheel/pinch zooms, d toggles 2d/3d, ↑/↓ walks depth,
            t cycles labels auto (collision-culled, recent wins) / all /
            off, c context trace, click a point → cut-out. Loops =
            repeated structure, jumps = topic shifts.

The chat is touch-ready (drag/pinch/tap on the canvas) and responsive.
Engine select in the header: "server api" (hooked python model, full
internals) or curated in-browser models — SmolLM2-135M/360M, Qwen2.5-0.5B —
with weights streamed from huggingface.co (aggregated progress bar,
switch-while-loading safe, models cached per session) and inference on
WebGPU (wasm fallback) via transformers.js. Browser runs feed the viz
with what is honestly observable client-side: real tokens in the strip
plus top-k prediction probabilities tapped from the logits via a custom
LogitsProcessor, under a topology fetched from the model's config.json —
per-layer cells stay pale because compiled ONNX graphs expose no hidden
states; norms/lens/trajectory need the server engine. The exception is
the "SmolLM2-135M observable" entry: it loads the export_observable.py
artifact from /models/ and patches model.forward to read the declared
hidden.* outputs — resid norms, the full embedding trajectory + context
trace and predictions are then computed client-side from real tensors
(mlp/attn/head norms and the logit lens remain server-only).
- dots    — logit-lens flips: layer's top-1 prediction differs from the
            layer below; red = flips to the emitted token (settle layer).
            Settle depth ≈ per-token difficulty.
- attention sources (server runs; eager attention) — every token frame
  carries, per attention layer, the top-8 positions it attended to (mean
  over heads) + attention spread, plus per-token uncertainty (entropy in
  bits, top-1 margin). Select an A cell → sources draw as arcs over the
  token strip (blue ◂ = mass on the prompt); cut-out lists them with
  weights. Curator checks: uncertainty, attention_focus.

## Known constraints

- LM Studio's API exposes only tokens/logprobs — internals need HF (or MLX)
  weights. MLX 8-bit exports (config `quantization.mode=affine`) are NOT
  HF-loadable; harness detects this and falls back to hub bf16.
- Attention matrices and GeLU histograms are mock-only so far; real capture
  needs `output_attentions=True` + `attn_implementation="eager"` and an
  `act_fn` hook (dashboard already renders honest placeholders).
- Fused Mamba kernels may bypass `dt_proj`; Δ strip reports when empty.

## Next steps (rough priority)

1. Real attention matrices: eager-attn capture behind a `--attn-maps` flag
   (memory: T×T per layer — window it).
2. mlx-lm capture path for MLX-quantized local models (wrap layer __call__).
3. Emit protocol v2 natively from server.py (introspect layer children).
4. Logit-lens panel: DONE for top-1 per layer (server emits "lens", chat.html
   renders ladder + flip dots); extend to per-layer top-k.
5. Port the dashboard into Loupe as a pane (DOM-per-layer approach transfers).
