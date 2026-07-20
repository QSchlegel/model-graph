#!/usr/bin/env python3
"""
harness.py — test harness for the model-graph visualizer pipeline.

Stages:
  validate  Check a model folder: config, layer_types, tokenizer, chat
            template, weight shards, quantization format. Prints the
            topology the visualizer should derive.
  capture   Run a short hooked generation (server.HookedModel) and write
            a protocol trace (NDJSON). Accepts an HF hub id or local dir.
  verify    Structural + physical checks on a trace, optionally
            cross-checked against a config.json.
  all       validate --model-dir, capture (dir, falling back to
            --fallback-hub if the dir is unloadable), verify the result.

Examples:
  python harness.py validate --model-dir ~/models/lfm2.5-1.2b-mlx-8bit
  python harness.py capture  --model LiquidAI/LFM2.5-1.2B-Instruct --tokens 8 --out t.ndjson
  python harness.py verify   t.ndjson --config ~/models/lfm2.5-1.2b-mlx-8bit/config.json
  python harness.py all --model-dir DIR --fallback-hub LiquidAI/LFM2.5-1.2B-Instruct

Exit code 0 = all checks passed; 1 = failures (CI-friendly).
"""

import argparse, json, os, sys

OK, WARN, FAIL = "✓", "!", "✗"
_failures = []

def report(status, msg):
    print(f"  {status} {msg}")
    if status == FAIL:
        _failures.append(msg)


# ─────────────────────────────────────────────────────────────── validate ──
def kind_of(layer_type: str) -> str:
    if "conv" in layer_type: return "conv"
    if "mamba" in layer_type or "ssm" in layer_type: return "ssm"
    if "full" in layer_type or "global" in layer_type: return "global"
    return "local"

def validate(model_dir: str) -> dict:
    print(f"[validate] {model_dir}")
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        report(FAIL, "config.json missing"); return {}
    cfg = json.load(open(cfg_path))
    n = cfg.get("num_hidden_layers")
    lts = cfg.get("layer_types")
    report(OK, f"arch={cfg.get('architectures')} model_type={cfg.get('model_type')} layers={n}")
    if lts:
        kinds = [kind_of(t) for t in lts]
        from collections import Counter
        report(OK, f"layer_types: {dict(Counter(kinds))} → visualizer cards: "
                   + "".join({'conv':'C','ssm':'S','global':'A','local':'a'}[k] for k in kinds))
        if len(lts) != n:
            report(FAIL, f"layer_types length {len(lts)} != num_hidden_layers {n}")
    else:
        report(WARN, "no layer_types in config — topology will default to attention")
    report(OK, f"heads={cfg.get('num_attention_heads')} kv_heads={cfg.get('num_key_value_heads')} "
               f"hidden={cfg.get('hidden_size')}")

    q = cfg.get("quantization") or cfg.get("quantization_config")
    loadable_hf = True
    if q and "mode" in q and q.get("mode") == "affine":
        loadable_hf = False
        report(WARN, f"MLX affine quantization ({q.get('bits')}-bit, group={q.get('group_size')}) "
                     "— NOT loadable by HF transformers; use mlx-lm, or capture from hub bf16 weights")
    elif q:
        report(WARN, f"quantization present: {q}")

    idx_path = None
    for cand in ("model.safetensors.index.json", "model_safetensors_index.json"):
        p = os.path.join(model_dir, cand)
        if os.path.exists(p): idx_path = p; break
    shards_ok = False
    if idx_path:
        idx = json.load(open(idx_path))
        shards = sorted(set(idx.get("weight_map", {}).values()))
        missing = [s for s in shards if not os.path.exists(os.path.join(model_dir, s))]
        total = idx.get("metadata", {}).get("total_size", 0)
        if missing:
            report(FAIL, f"weight shards missing: {missing} (index expects {total/1e9:.2f} GB)")
        else:
            shards_ok = True
            report(OK, f"all {len(shards)} weight shard(s) present ({total/1e9:.2f} GB)")
    elif os.path.exists(os.path.join(model_dir, "model.safetensors")):
        shards_ok = True
        report(OK, "model.safetensors present (single file, no index)")
    else:
        report(FAIL, "no safetensors index or weights found")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
        ids = tok.apply_chat_template([{"role": "user", "content": "ping"}],
                                      add_generation_prompt=True)
        if not isinstance(ids, list): ids = ids["input_ids"]
        report(OK, f"tokenizer loads · vocab={tok.vocab_size} · chat template renders "
                   f"({len(ids)} tokens, bos={tok.bos_token!r} eos={tok.eos_token!r})")
    except Exception as e:
        report(FAIL, f"tokenizer/chat template: {e}")

    return {"cfg": cfg, "shards_ok": shards_ok, "loadable_hf": loadable_hf and shards_ok}


# ──────────────────────────────────────────────────────────────── capture ──
def capture(model: str, tokens: int, out: str, prompt: str, device: str, dtype: str):
    print(f"[capture] {model} → {out} ({tokens} tokens, {device}/{dtype})")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server import HookedModel
    hm = HookedModel(model, device, dtype)
    with open(out, "w") as f:
        f.write(json.dumps(hm.topology(model)) + "\n")
        n = 0
        for msg in hm.generate_stream(prompt, tokens):
            f.write(json.dumps(msg) + "\n")
            if msg["type"] == "token":
                n += 1
                print(f"  t={msg['t']} {msg['text']!r}")
    report(OK, f"captured {n} tokens")


# ───────────────────────────────────────────────────────────────── verify ──
def finite(x): return isinstance(x, (int, float)) and x == x and abs(x) != float("inf")

def verify(trace: str, config: str | None):
    print(f"[verify] {trace}" + (f" against {config}" if config else ""))
    msgs = [json.loads(l) for l in open(trace) if l.strip()]
    topos = [m for m in msgs if m["type"] == "topology"]
    toks  = [m for m in msgs if m["type"] == "token"]
    if len(topos) != 1: report(FAIL, f"expected 1 topology message, got {len(topos)}")
    else: report(OK, "exactly one topology message")
    if not toks: report(FAIL, "no token frames"); return
    if msgs[-1]["type"] == "done": report(OK, "trace ends with done")
    else: report(WARN, "trace does not end with done")

    topo = topos[0]
    nl = len(topo["layers"])
    if all(len(t["layers"]) == nl for t in toks):
        report(OK, f"{len(toks)} frames × {nl} layers, consistent")
    else:
        report(FAIL, "layer count varies across frames")
    if [t["t"] for t in toks] == list(range(len(toks))):
        report(OK, "token indices sequential")
    else: report(FAIL, "token indices not sequential")

    bad = sum(1 for t in toks for L in t["layers"]
              for k in ("attn_norm", "mlp_norm", "resid_norm")
              if k in L and not finite(L[k]))
    report(OK if bad == 0 else FAIL, f"norm finiteness: {bad} bad values")

    # per-kind head expectations
    kinds = [l["attn"]["kind"] for l in topo["layers"]]
    attn_idx = [i for i, k in enumerate(kinds) if k in ("global", "local", "attn")]
    conv_idx = [i for i, k in enumerate(kinds) if k in ("conv", "ssm")]
    if attn_idx:
        hs = {len(toks[0]["layers"][i].get("head_norms", [])) for i in attn_idx}
        report(OK if len(hs) == 1 and 0 not in hs else FAIL,
               f"attention layers carry head norms: sizes={sorted(hs)}")
    if conv_idx:
        empty = all(not toks[0]["layers"][i].get("head_norms") for i in conv_idx)
        report(OK if empty else FAIL, "conv/ssm layers have no head norms")

    # depth trend: resid norm should grow front → back (averaged over frames)
    first = sum(t["layers"][0].get("resid_norm", 0) for t in toks) / len(toks)
    last  = sum(t["layers"][-1].get("resid_norm", 0) for t in toks) / len(toks)
    report(OK if last > first else WARN,
           f"residual depth trend: L0 avg {first:.3f} → L{nl-1} avg {last:.3f}")

    if config:
        cfg = json.load(open(config))
        if cfg.get("num_hidden_layers") == nl:
            report(OK, f"layer count matches config ({nl})")
        else:
            report(FAIL, f"layer count {nl} != config {cfg.get('num_hidden_layers')}")
        lts = cfg.get("layer_types")
        if lts:
            want = [kind_of(t) for t in lts]
            got = ["conv" if k in ("conv", "ssm") else "global" if k == "global" else "local"
                   for k in kinds]
            want2 = ["conv" if k in ("conv", "ssm") else k for k in want]
            report(OK if want2 == got else FAIL, "layer kinds match config.layer_types")
        heads = cfg.get("num_attention_heads")
        if heads and attn_idx:
            got_h = len(toks[0]["layers"][attn_idx[0]].get("head_norms", []))
            report(OK if got_h == heads else FAIL,
                   f"head count {got_h} vs config {heads}")


# ─────────────────────────────────────────────────────────────────── main ──
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("--model-dir", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--model", required=True); c.add_argument("--tokens", type=int, default=8)
    c.add_argument("--out", default="harness_trace.ndjson")
    c.add_argument("--prompt", default="Reply with one short sentence: why is the sky blue?")
    c.add_argument("--device", default="cpu"); c.add_argument("--dtype", default="bfloat16")
    w = sub.add_parser("verify"); w.add_argument("trace"); w.add_argument("--config")
    a = sub.add_parser("all")
    a.add_argument("--model-dir", required=True); a.add_argument("--fallback-hub")
    a.add_argument("--tokens", type=int, default=8)
    a.add_argument("--device", default="cpu"); a.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    if args.cmd == "validate":
        validate(args.model_dir)
    elif args.cmd == "capture":
        capture(args.model, args.tokens, args.out, args.prompt, args.device, args.dtype)
    elif args.cmd == "verify":
        verify(args.trace, args.config)
    elif args.cmd == "all":
        info = validate(args.model_dir)
        src = args.model_dir if info.get("loadable_hf") else args.fallback_hub
        if not src:
            report(FAIL, "folder not HF-loadable and no --fallback-hub given")
        else:
            if src != args.model_dir:
                print(f"  → folder not HF-loadable, capturing from {src}")
            out = "harness_trace.ndjson"
            capture(src, args.tokens, out,
                    "Reply with one short sentence: why is the sky blue?",
                    args.device, args.dtype)
            verify(out, os.path.join(args.model_dir, "config.json"))

    print(f"\n{'PASS' if not _failures else 'FAIL'} ({len(_failures)} failure(s))")
    sys.exit(1 if _failures else 0)

if __name__ == "__main__":
    main()
