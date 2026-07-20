#!/usr/bin/env python3
"""
make_mock.py — generate realistic-ish mock activation traces (NDJSON) for
gemma-graph. First line: topology message; then one token message per line;
final line: done. Consumable by replay.py or any protocol-compatible client.

Realism heuristics baked in (so the graph looks like a real run, not noise):
  * residual norms grow monotonically with depth (~3-4x from L0 to L(n-1))
  * a few "spike" layers with persistently hot MLP norms
  * global-attention layers run slightly hotter than local ones
  * per-head norms: 1-2 dominant heads per layer, stable across tokens
  * MoE routing is sticky: each token position drifts between a small
    expert set instead of resampling uniformly

Usage:
  python make_mock.py --out mock_dense.ndjson
  python make_mock.py --moe --out mock_moe.ndjson
  python make_mock.py --layers 32 --tokens 200 --seed 7 --out big.ndjson
"""

import argparse, json, random

TEXT = ("the model reads each token and updates its residual stream layer by "
        "layer until the final norm projects back into vocabulary space and "
        "the next token is sampled from the resulting distribution .").split()


def gen(layers: int, tokens: int, moe: bool, heads: int, experts: int,
        topk: int, seed: int):
    rng = random.Random(seed)
    topo = {"type": "topology",
            "model": f"mock-{layers}L" + ("-moe" if moe else "-dense"),
            "layers": [{
                "i": i,
                "attn": {"kind": "global" if (i + 1) % 4 == 0 else "local",
                         "heads": heads},
                "mlp": ({"kind": "moe", "experts": experts, "topk": topk}
                        if moe else {"kind": "dense", "experts": 0, "topk": 0}),
            } for i in range(layers)]}
    yield topo

    spikes = set(rng.sample(range(2, layers), max(1, layers // 8)))
    dom_heads = [rng.sample(range(heads), 2) for _ in range(layers)]
    # per-layer expert affinity pool (sticky routing)
    pools = [rng.sample(range(experts), 4) for _ in range(layers)] if moe else None
    walk = [rng.uniform(0.8, 1.2) for _ in range(layers)]

    for t in range(tokens):
        out_layers = []
        for i in range(layers):
            walk[i] = min(2.0, max(0.4, walk[i] + rng.uniform(-0.08, 0.08)))
            depth = 1.0 + 3.2 * (i / max(layers - 1, 1)) ** 1.3
            g = 1.15 if topo["layers"][i]["attn"]["kind"] == "global" else 1.0
            attn = depth * g * walk[i] * rng.uniform(0.85, 1.15)
            mlp = depth * walk[i] * (2.2 if i in spikes else 1.0) * rng.uniform(0.9, 1.25)
            resid = depth * 6.0 * rng.uniform(0.95, 1.05)
            hn = []
            for h in range(heads):
                base = attn * (1.6 if h in dom_heads[i] else 0.55)
                hn.append(round(base * rng.uniform(0.8, 1.2), 4))
            entry = {"i": i, "attn_norm": round(attn, 4),
                     "mlp_norm": round(mlp, 4), "resid_norm": round(resid, 4),
                     "head_norms": hn}
            if moe:
                pool = pools[i]
                if rng.random() < 0.08:                     # occasional pool drift
                    pool[rng.randrange(4)] = rng.randrange(experts)
                chosen = rng.sample(pool, topk)
                w = sorted([rng.uniform(0.5, 0.85)] +
                           [rng.uniform(0.05, 0.4) for _ in range(topk - 1)],
                           reverse=True)
                s = sum(w)
                entry["expert_weights"] = [[e, round(x / s, 4)]
                                           for e, x in zip(chosen, w)]
            out_layers.append(entry)
        yield {"type": "token", "t": t, "text": TEXT[t % len(TEXT)],
               "layers": out_layers}
    yield {"type": "done"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--moe", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    with open(a.out, "w") as f:
        for msg in gen(a.layers, a.tokens, a.moe, a.heads, a.experts, a.topk, a.seed):
            f.write(json.dumps(msg) + "\n")
    print(f"wrote {a.out}")
