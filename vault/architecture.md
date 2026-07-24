---
tags: [diagram, architecture]
updated: 2026-07-20
---

# Architecture

Component detail: [components](components.md) · wire format:
[protocol](protocol.md) · request sequences: [flows](flows.md)

## Component graph

```mermaid
flowchart LR
  subgraph clients["Clients"]
    CHAT["web/chat.html<br/>chat + background viz<br/>+ agent mode"]
    DASH["web/dashboard.html<br/>block drill-down"]
    AGENT["web/agent.html · /agent<br/>agent state machine<br/>+ local JS tools"]
    SDK["OpenAI SDKs / agents<br/>(native tool calling)"]
    CLI["curator.py CLI<br/>run/test/suite"]
  end

  subgraph api["api_server.py · :8080"]
    OAI["/v1/chat/completions<br/>tools · images · SSE"]
    HUB["Hub<br/>ws broadcast + last-run replay"]
    STATIC["static<br/>/ /dashboard /models/"]
    CREST["/curator/* REST"]
  end

  HM["server.py HookedModel<br/>torch hooks · eager attention<br/>lens · proj · attn sources"]
  CUR["curator.py<br/>runs/ library · checks · events"]
  EXP["export_observable.py<br/>ONNX graph surgery"]
  OBS[("models/observable/<br/>hidden.* outputs + basis")]
  HF[("Hugging Face hub<br/>weights · ONNX · configs")]
  TJS["transformers.js + ORT<br/>WebGPU in-browser"]

  CHAT -->|POST chat| OAI
  SDK -->|POST chat| OAI
  OAI --> HM
  HM -->|token frames| HUB
  HUB -->|"ws :8765"| CHAT
  HUB -->|"ws :8765"| DASH
  OAI -->|record run| CUR
  CREST --> CUR
  CLI -->|drive api| OAI
  CLI --> CUR
  HF --> HM
  HF --> EXP
  EXP --> OBS
  OBS --> STATIC
  STATIC -->|observable model| TJS
  HF -->|stock models| TJS
  TJS -->|local frames| CHAT
  TJS -->|per-step frames| AGENT
  AGENT -->|generate step| TJS
  AGENT -.->|server-engine loop| OAI
```

> [!note] The agent harness is client-side
> `web/agent.html` and the chat's agent mode run a ReAct tool-use **state
> machine** in the browser — local JS tools, no server round-trip (except the
> optional server-engine loop, dashed above). It reuses the same transformers.js
> loader and `onProto` viz funnel — see [agent-harness](agent-harness.md).

## Data flow (one server-engine token)

```mermaid
flowchart TD
  P["prompt / messages<br/>(+tools, +images)"] --> T["chat template<br/>tokenizer/processor"]
  T --> F["model forward<br/>(eager, output_attentions)"]
  F --> HK["forward hooks<br/>attn/mlp/resid norms · head norms<br/>full-seq hidden vectors"]
  F --> LG["logits"]
  HK --> LENS["logit lens<br/>final_norm + lm_head per layer"]
  HK --> PRJ["3D projection<br/>seeded QR basis"]
  LG --> TK["top-k · entropy · margin"]
  F --> AS["attention sources<br/>mean-over-heads top-8 + spread"]
  LENS --> FR["token frame (ndjson)"]
  PRJ --> FR
  TK --> FR
  AS --> FR
  HK --> FR
  FR --> WS["Hub → ws clients"]
  FR --> REC["curator run record"]
  FR --> SSE["SSE delta → API client"]
```

> [!note] Three details the graphs can't show
> - The prefill step additionally emits a `context` message: every prompt
>   position's per-layer projection (the blue trace in embed view).
> - The Hub keeps `[topology, context, token*, done]` of the current run and
>   replays it to late-joining ws clients — the deep-link handoff in
>   [flows](flows.md) depends on this.
> - Observable browser runs produce the same frame shape client-side
>   (resid_norm + proj + topk only) via a patched `model.forward` reading
>   `hidden.*` outputs — see [engines](engines.md).
