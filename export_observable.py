#!/usr/bin/env python3
"""
export_observable.py — build an *observable* ONNX model for in-browser
internals: take the stock transformers.js-compatible ONNX export and
graph-surgery every layer's residual stream into a declared graph output
(`hidden.0` … `hidden.N-1`, shape [batch, seq, hidden]).

A compiled ONNX graph only exposes declared outputs — that is why the
browser engine is normally chat-only. This script re-declares the
residual tensors that already flow through the fused
SkipSimplifiedLayerNormalization nodes, so the browser can compute
norms / logit-lens / trajectories client-side with zero extra compute.

  python export_observable.py                    # SmolLM2-135M q4f16
  python export_observable.py --repo <hf-repo> --variant model_q4f16

Output: models/observable/<name>/
  onnx/model_q4f16.onnx    surgeried graph (logits + present.* + hidden.*)
  config.json, tokenizer.json, ...              copied for transformers.js
  proj_basis.json          seeded random orthonormal basis (hidden × 3),
                           same construction as server.py's trajectory view
Validated against PyTorch: logits argmax parity + per-layer residual-norm
agreement on a test prompt.
"""

import argparse
import json
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def find_residuals(graph, n_layers):
    """Map layer index → tensor name of the residual stream after that layer.

    optimum's fused export computes residual sums inside
    SkipSimplifiedLayerNormalization; output_3 (input_skip_bias_sum) of the
    NEXT block's input_layernorm is exactly 'residual after layer i'.
    The final layer's sum lives on the final_norm node as an optional
    output that is not emitted — we extend that node to emit it.
    """
    nodes = {n.name: n for n in graph.node}
    resid = {}
    for i in range(n_layers - 1):
        node = nodes.get(f"/model/layers.{i+1}/input_layernorm/SkipLayerNorm")
        if not (node and len(node.output) >= 4 and node.output[3]):
            raise RuntimeError(f"no residual sum output for layer {i} "
                               f"(unexpected export layout)")
        resid[i] = node.output[3]
    fin = nodes.get(f"/model/layers.{n_layers}/final_norm_layernorm/"
                    "SkipLayerNorm")
    if fin is None:
        raise RuntimeError("final norm SkipLayerNorm not found")
    while len(fin.output) < 4:                 # declare the optional output
        fin.output.append("")
    if not fin.output[3]:
        fin.output[3] = fin.name.rsplit("/", 1)[0] + "/output_3"
    resid[n_layers - 1] = fin.output[3]
    return resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--variant", default="model_q4f16")
    ap.add_argument("--out", default="models/observable")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    name = args.repo.split("/")[1]
    out = Path(args.out) / name
    (out / "onnx").mkdir(parents=True, exist_ok=True)

    print(f"[fetch] {args.repo} · onnx/{args.variant}.onnx")
    src = hf_hub_download(args.repo, f"onnx/{args.variant}.onnx")
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
              "generation_config.json"):
        shutil.copy(hf_hub_download(args.repo, f), out / f)
    cfg = json.loads((out / "config.json").read_text())
    n_layers, hidden = cfg["num_hidden_layers"], cfg["hidden_size"]

    print(f"[surgery] exposing {n_layers} residual streams (hidden={hidden})")
    model = onnx.load(src)
    graph = model.graph
    resid = find_residuals(graph, n_layers)
    for i in range(n_layers):                  # Cast → fp32, stable output name
        graph.node.append(helper.make_node(   # (fp32 = plain Float32Array in
            "Cast", [resid[i]], [f"hidden.{i}"],   # onnxruntime-web, no half
            to=TensorProto.FLOAT,                  # decoding client-side)
            name=f"/observable/hidden.{i}"))
        graph.output.append(helper.make_tensor_value_info(
            f"hidden.{i}", TensorProto.FLOAT,
            ["batch_size", "sequence_length", hidden]))
    dst = out / "onnx" / f"{args.variant}.onnx"
    onnx.save(model, dst)
    print(f"  → {dst} ({dst.stat().st_size/1048576:.0f} MB, "
          f"{len(graph.output)} outputs)")

    # trajectory basis: identical construction to server.py (seed 0, QR)
    import torch
    gen = torch.Generator().manual_seed(0)
    basis = torch.linalg.qr(torch.randn(hidden, 3, generator=gen))[0]
    (out / "proj_basis.json").write_text(json.dumps(
        {"hidden": hidden, "seed": 0,
         "basis": [[round(float(v), 6) for v in row] for row in basis]}))
    print(f"[basis] proj_basis.json ({hidden}×3, seed 0)")

    # ---- validate: ONNX (with cache inputs) vs PyTorch reference ----------
    print("[verify] onnxruntime vs transformers on a test prompt")
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.repo)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "Why is the sky blue?"}],
        add_generation_prompt=True, return_tensors="np")
    if hasattr(ids, "input_ids"):              # transformers 5.x BatchEncoding
        ids = ids.input_ids
    S = ids.shape[1]
    kv = cfg["num_key_value_heads"]
    hd = hidden // cfg["num_attention_heads"]
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    feeds = {"input_ids": ids.astype(np.int64),
             "attention_mask": np.ones((1, S), np.int64),
             "position_ids": np.arange(S, dtype=np.int64)[None]}
    for i in range(n_layers):
        for k in ("key", "value"):
            feeds[f"past_key_values.{i}.{k}"] = \
                np.zeros((1, kv, 0, hd), np.float16)
    outs = sess.run(["logits"] + [f"hidden.{i}" for i in range(n_layers)],
                    feeds)
    logits, hiddens = outs[0], outs[1:]

    ref = AutoModelForCausalLM.from_pretrained(args.repo,
                                               torch_dtype=torch.float32)
    # capture true layer outputs via hooks — same semantics as server.py
    # (HF's output_hidden_states applies the final norm to the last entry)
    ref_resid = {}
    for i, layer in enumerate(ref.model.layers):
        layer.register_forward_hook(
            lambda _m, _i, out, i=i: ref_resid.__setitem__(
                i, (out[0] if isinstance(out, tuple) else out).detach()))
    with torch.no_grad():
        r = ref(torch.tensor(ids))

    fails = 0
    om, rm = int(np.argmax(logits[0, -1])), int(r.logits[0, -1].argmax())
    ok = om == rm
    fails += not ok
    print(f"  {'✓' if ok else '✗'} next-token argmax: onnx={tok.decode([om])!r}"
          f" ref={tok.decode([rm])!r}")
    drifts = []
    for i in range(n_layers):
        on = float(np.linalg.norm(hiddens[i][0, -1].astype(np.float32)))
        rn = float(ref_resid[i][0, -1].norm())
        drifts.append(abs(on - rn) / max(rn, 1e-6))
    worst = max(drifts)
    ok = worst < 0.15 and all(np.isfinite(h).all() for h in hiddens)
    fails += not ok
    print(f"  {'✓' if ok else '✗'} residual norms track reference "
          f"(worst drift {worst*100:.1f}% — q4f16 quantization)")
    ok = all(h.shape == (1, S, hidden) for h in hiddens)
    fails += not ok
    print(f"  {'✓' if ok else '✗'} {n_layers} hidden.* outputs, "
          f"shape (1,{S},{hidden})")
    print("PASS" if fails == 0 else f"FAIL ({fails})")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
