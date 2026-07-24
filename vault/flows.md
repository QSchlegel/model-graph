---
tags: [diagram, flows]
updated: 2026-07-20
---

# Flows

Who's who in these diagrams: [components](components.md) · message shapes:
[protocol](protocol.md) · per-engine capability: [engines](engines.md)

## User flow — inspecting a generation (server engine)

```mermaid
flowchart LR
  A[type message] --> B[watch heatmap fill<br/>token strip + lens dots]
  B --> C{explore}
  C -->|hover| D[tooltip: norms · lens ·<br/>looks-at · head under cursor]
  C -->|click cell| E[cut-out panel<br/>arrows navigate]
  C -->|empty click / viz btn| F[cycle available metrics<br/>mlp/attn/resid/embed]
  C -->|viz: embed| G[trajectory + context trace<br/>minimap picks layer]
  E --> H[attention arcs over strip<br/>sources · lens ladder · heads]
  E --> I[full drill-down link<br/>deep-link → dashboard]
  G --> E
```

## Chat request — server engine

```mermaid
sequenceDiagram
  participant U as chat UI / SDK
  participant A as api_server
  participant M as HookedModel
  participant H as Hub(ws)
  participant C as curator
  U->>A: POST /v1/chat/completions (messages, tools?, images?)
  A->>H: cast topology (dashboards reset)
  A->>M: generate_stream(hf_messages, tools)
  M-->>A: context msg (prefill projections)
  A->>H: cast context
  loop per token
    M-->>A: token frame (norms, lens, proj, topk, attn_srcs, entropy)
    A->>H: cast frame → all ws clients
    A-->>U: SSE delta (text)
  end
  M-->>A: done (+usage)
  A->>C: save_run(meta, full trace)
  A-->>U: final chunk (finish_reason, tool_calls?)
```

## Tool-calling agent loop

```mermaid
sequenceDiagram
  participant Ag as agent (OpenAI SDK)
  participant A as api_server
  participant M as model
  Ag->>A: messages + tools
  A->>M: template(tools) → generate
  M-->>A: "<|tool_call_start|>[get_weather(city='Paris')]…"
  A-->>Ag: tool_calls[] · finish_reason=tool_calls
  Ag->>Ag: execute tool
  Ag->>A: + assistant(tool_calls) + tool(result)
  A->>M: template round-trip (calls re-serialized pythonic)
  M-->>A: grounded final answer
  A-->>Ag: content · finish_reason=stop
```

## In-browser agent harness (ReAct state machine)

Drives a tiny browser model through a bounded tool-use loop, client-side —
`/agent` (guided + graph) and the chat's agent-mode toggle. Full states, tools
and parsing: [agent-harness](agent-harness.md).

```mermaid
flowchart TD
  T[task + selected tools] --> SP[system prompt<br/>ReAct format + tool list]
  SP --> TH[THINK<br/>model generates one step]
  TH --> R{ROUTE<br/>parse the text}
  R -->|tool call| A[ACT<br/>run local JS tool]
  R -->|final answer| AN([ANSWER, done])
  R -->|neither| G[GUARD]
  A --> O[OBSERVE<br/>append result] --> TH
  G -->|retry with the format| TH
  G -->|retries / step cap| ST([STOP])
```

## Browser engines

```mermaid
sequenceDiagram
  participant UI as chat UI
  participant T as transformers.js
  participant O as ORT (WebGPU)
  UI->>T: pipeline(model) — HF hub or /models/observable/
  Note over UI,T: progress bar · switch-while-loading safe · per-model cache
  UI->>T: generate(messages, streamer [,logits_processor])
  T->>O: session.run per step
  alt observable model
    O-->>T: logits + hidden.0..N (fp32)
    T-->>UI: patched forward → resid norms · proj · topk · context
  else stock model
    O-->>T: logits only
    T-->>UI: text chunks + top-k (logits tap)
  end
  UI->>UI: onProto(frames) → same viz pipeline as ws
```

## Deep-link handoff (cut-out → dashboard)

```mermaid
sequenceDiagram
  participant C as chat cut-out
  participant D as dashboard
  participant H as Hub
  C->>D: open /dashboard?ws=1&layer=8&block=0&t=14
  D->>H: ws connect
  H-->>D: replay last run (topology+context+tokens+done)
  D->>D: auto-unfold L8/block0 · scroll into view
  D->>D: seek selT=14 once frame streamed
```

## Curator suite (CI gate)

```mermaid
flowchart LR
  S[suites/*.json] --> R[curator.py suite]
  R -->|POST per case| API[api_server]
  API -->|auto-record| RUNS[(runs/)]
  R --> CHK["checks: internals + expectations<br/>(contains · eos · settle · tool_call ·<br/>args · abstain · grounded chain)"]
  CHK -->|fail| HOOK["--on-fail CMD · test_fail event"]
  CHK --> EXIT[exit code → CI]
  RUNS --> INSPECT["failure triage:<br/>curator show/test id ·<br/>dashboard deep-link"]
```
