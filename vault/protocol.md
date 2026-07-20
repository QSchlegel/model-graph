---
tags: [reference, protocol]
updated: 2026-07-20
---

# Protocol (v1) — ndjson over ws :8765

One JSON object per message. Canonical source: `server.py` docstring.
Emitted by [server](components.md#server), fanned out by
[api-server](components.md#api-server)'s Hub.

> [!warning] Protocol changes fan out
> Five consumers must move together: dashboard `normFrame`, chat `onProto`,
> harness `verify`, curator checks, and the Hub replay buffer. If you change
> a message shape, touch all five (and this note, and the README block).

```
{"type":"topology","model":str,"layers":[{"i":int,
    "attn":{"kind":"local"|"global"|"attn"|"conv"|"ssm","heads":int},
    "mlp":{"kind":"dense"|"moe","experts":int,"topk":int}}]}

{"type":"context","tokens":[str,...],          # prefill: full prompt trace
 "proj":[[[x,y,z]×n_layers] per position]}

{"type":"token","t":int,"text":str,
 "topk":[[text,prob]×5],                       # final-logits top-5
 "entropy":float,"margin":float,               # uncertainty (bits, p1-p2)
 "lens":[[text,prob]×n_layers],                # logit lens per-layer argmax
 "proj":[[x,y,z]×n_layers],                    # resid → seeded 3D basis
 "eos":true?,                                  # on the eos token only
 "layers":[{"i":int,
    "attn_norm":float,"mlp_norm":float,"resid_norm":float,
    "head_norms":[float×heads],                # attention layers
    "attn_srcs":[[abs_pos,weight]×8],          # where this token looked
    "attn_entropy":float,                      # attention spread 0..1
    "expert_weights":[[idx,w]×topk],           # MoE layers
    "delta_mean":float,"delta":[float]         # SSM layers
 }]}

{"type":"done","prompt_tokens":int,"completion_tokens":int}
```

Semantics
- Sequence per run: `topology → context? → token* → done`. The api_server
  re-broadcasts topology at each request start (clients treat it as a run
  boundary; shape change = hard reset).
- `attn_srcs` positions are absolute (context positions + generated
  positions); clients map via context length + run-start boundary.
- Fields are additive and optional — clients must tolerate absence (browser
  runs emit only text/topk[/layers with resid_norm + proj]).
- Projection basis: `torch.linalg.qr(randn(hidden,3, seed 0))` — identical in
  server.py and export_observable.py (`proj_basis.json`).
