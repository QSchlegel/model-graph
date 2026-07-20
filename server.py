#!/usr/bin/env python3
"""
gemma-graph server — streams per-token, per-layer runtime stats of a
transformer (e.g. Gemma 4) as NDJSON over a websocket, for the node-graph
frontend in graph.html.

Usage:
  pip install websockets                # mock mode needs nothing else
  python server.py --mock               # fake 24-layer model, no weights
  python server.py --mock --mock-moe    # exercise the MoE/expert view

  pip install torch transformers accelerate
  python server.py --model google/gemma-4-E4B-it --prompt "Explain KV caching"

Protocol (one JSON object per websocket message):
  {"type":"topology","model":str,"layers":[{"i":int,
      "attn":{"kind":"local"|"global"|"attn","heads":int},
      "mlp":{"kind":"dense"|"moe","experts":int,"topk":int}}]}
  {"type":"context","tokens":[str,...],           # prefill: full prompt trace
   "proj":[[[x,y,z]×n_layers] per position]}
  {"type":"token","t":int,"text":str,
   "topk":[[text,prob],...],                      # final-logits top-5
   "entropy":float,"margin":float,                # uncertainty (bits, p1-p2)
   "lens":[[text,prob],...],                      # logit lens: per-layer argmax
   "proj":[[x,y,z],...],                          # resid → fixed random 3D basis
   "layers":[{"i":int,
      "attn_norm":float,"mlp_norm":float,"resid_norm":float,
      "head_norms":[float,...],
      "attn_srcs":[[abs_pos,weight]×8],           # attention layers: where this
      "attn_entropy":float,                       #   token looked (mean heads)
      "expert_weights":[[expert_idx,weight],...]  # MoE layers only
  }]}
  {"type":"done"}
"""

import argparse
import ast
import asyncio
import json
import math
import random
import re


# ----------------------------------------------------------------------------
# Mock backend — lets you build/test the frontend without downloading weights.
# ----------------------------------------------------------------------------

def mock_topology(n_layers: int, moe: bool) -> dict:
    layers = []
    for i in range(n_layers):
        layers.append({
            "i": i,
            # Gemma-style interleave: 3 local sliding-window layers, 1 global
            "attn": {"kind": "global" if (i + 1) % 4 == 0 else "local", "heads": 8},
            "mlp": ({"kind": "moe", "experts": 16, "topk": 2} if moe
                    else {"kind": "dense", "experts": 0, "topk": 0}),
        })
    return {"type": "topology", "model": "mock-24L" + ("-moe" if moe else ""), "layers": layers}


async def mock_stream(send, n_layers: int, moe: bool, n_tokens: int):
    await send(mock_topology(n_layers, moe))
    words = ("the quick brown fox jumps over the lazy dog and keeps on "
             "running through fields of attention heads").split()
    state = [random.uniform(0.5, 1.5) for _ in range(n_layers)]
    for t in range(n_tokens):
        layers = []
        for i in range(n_layers):
            state[i] = max(0.1, state[i] + random.uniform(-0.15, 0.15))
            depth = 1.0 + 2.5 * i / n_layers          # norms grow with depth
            attn = state[i] * depth * random.uniform(0.7, 1.3)
            mlp = state[i] * depth * random.uniform(0.9, 1.6)
            entry = {
                "i": i,
                "attn_norm": round(attn, 4),
                "mlp_norm": round(mlp, 4),
                "resid_norm": round((attn + mlp) * random.uniform(2.0, 3.0), 4),
                "head_norms": [round(attn * random.uniform(0.5, 1.5), 4) for _ in range(8)],
            }
            if moe:
                idx = random.sample(range(16), 2)
                w = random.uniform(0.55, 0.9)
                entry["expert_weights"] = [[idx[0], round(w, 3)], [idx[1], round(1 - w, 3)]]
            layers.append(entry)
        await send({"type": "token", "t": t, "text": words[t % len(words)], "layers": layers})
        await asyncio.sleep(0.25)
    await send({"type": "done"})


# ----------------------------------------------------------------------------
# Tool calling — parse model-emitted calls into structured form.
# LFM2-style: pythonic `[fn(a=1)]` (or JSON) between tool-call markers.
# ----------------------------------------------------------------------------

TOOL_CALL_RE = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>",
                          re.S)


def parse_tool_calls(text: str):
    calls = []
    for block in TOOL_CALL_RE.findall(text):
        block = block.strip()
        try:
            tree = ast.parse(block, mode="eval").body
            nodes = tree.elts if isinstance(tree, ast.List) else [tree]
            for c in nodes:
                if isinstance(c, ast.Call):
                    calls.append({
                        "name": (getattr(c.func, "id", None)
                                 or getattr(c.func, "attr", "?")),
                        "arguments": {k.arg: ast.literal_eval(k.value)
                                      for k in c.keywords}})
            continue
        except Exception:
            pass
        try:                            # JSON-emitting models
            j = json.loads(block)
            for c in (j if isinstance(j, list) else [j]):
                calls.append({"name": c.get("name"), "arguments":
                              c.get("arguments", c.get("parameters", {}))})
        except Exception:
            pass
    return calls


# ----------------------------------------------------------------------------
# Real backend — HF transformers + forward hooks.
# ----------------------------------------------------------------------------

class HookedModel:
    def __init__(self, model_id: str, device: str, dtype: str):
        import torch
        from transformers import (AutoConfig, AutoModelForCausalLM,
                                  AutoTokenizer)
        self.torch = torch
        probe = AutoConfig.from_pretrained(model_id)
        self.is_vl = hasattr(probe, "vision_config")
        kw = dict(torch_dtype=getattr(torch, dtype), device_map=device,
                  attn_implementation="eager")  # eager → maps observable
        if self.is_vl:                 # vision-language model → processor
            from transformers import (AutoModelForImageTextToText,
                                      AutoProcessor)
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.tok = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, **kw)
        else:
            self.processor = None
            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.eval()
        # decoder layers live at different paths per architecture
        self.layers = None
        for path in ("model.layers", "model.language_model.layers",
                     "language_model.model.layers",
                     "model.text_model.layers"):
            obj = self.model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                self.layers, lpath = obj, path
                break
            except AttributeError:
                continue
        if self.layers is None:
            raise RuntimeError("could not locate decoder layers")
        base = self.model
        for part in lpath.split(".")[:-1]:
            base = getattr(base, part)
        self.cfg = getattr(self.model.config, "text_config",
                           self.model.config)
        self.capture = {}          # scratch dict filled by hooks each forward
        # logit lens: final norm module (naming varies) + lm_head
        self.final_norm = (getattr(base, "norm", None)
                           or getattr(base, "final_layernorm", None)
                           or getattr(base, "embedding_norm", None))
        self.lens_ok = self.final_norm is not None and hasattr(self.model, "lm_head")
        # embedding-trajectory view: fixed random orthonormal 3D basis
        # (seeded → stable across runs; no refit drift, unlike PCA)
        gen = torch.Generator().manual_seed(0)
        basis = torch.randn(self.cfg.hidden_size, 3, generator=gen)
        self.proj_basis = torch.linalg.qr(basis)[0].to(
            next(self.model.parameters()).device)
        self._register_hooks()

    def _register_hooks(self):
        for i, layer in enumerate(self.layers):
            # token mixer: attention or short-conv (hybrid archs like LFM2.5)
            mixer = getattr(layer, "self_attn", None) or getattr(layer, "conv", None)
            if mixer is not None:
                mixer.register_forward_hook(self._save("attn", i))
            # channel mixer: mlp (Gemma/Llama) or feed_forward (LFM2)
            mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
            if mlp is not None:
                mlp.register_forward_hook(self._save("mlp", i))
            layer.register_forward_hook(self._save("resid", i))
            # Per-head norms: capture the input to the attn output projection.
            if hasattr(layer, "self_attn"):
                op = (getattr(layer.self_attn, "o_proj", None)
                      or getattr(layer.self_attn, "out_proj", None))
                if op is not None:
                    op.register_forward_pre_hook(self._save_heads(i))
            # Best-effort MoE router capture — module naming varies by arch.
            if mlp is not None:
                router = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
                if router is not None:
                    router.register_forward_hook(self._save_router(i))
            # Best-effort Mamba/SSM Δ capture: dt_proj output ≈ pre-softplus Δ.
            if mixer is not None:
                dtp = getattr(mixer, "dt_proj", None)
                if dtp is not None:
                    dtp.register_forward_hook(self._save_delta(i))

    def _save_delta(self, i):
        def hook(_mod, _inp, out):
            import torch
            d = torch.nn.functional.softplus(
                (out[0] if isinstance(out, tuple) else out)[0, -1].float())
            self.capture[("delta_mean", i)] = round(float(d.mean()), 4)
            step = max(1, d.numel() // 24)
            self.capture[("delta", i)] = [round(float(v), 4) for v in d[::step][:24]]
        return hook

    def _save(self, kind, i):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.capture[(kind, i)] = float(h[0, -1].float().norm())
            if kind == "resid":    # keep vectors (full seq) for lens/proj/ctx
                self.capture[("hid", i)] = h[0].detach()
        return hook

    def _save_heads(self, i):
        def hook(mod, inp):
            x = inp[0][0, -1].float()                       # [heads * head_dim]
            heads = getattr(self.cfg, "num_attention_heads", None) or 1
            self.capture[("heads", i)] = [float(v) for v in
                                          x.view(heads, -1).norm(dim=-1)]
        return hook

    def _save_router(self, i):
        def hook(_mod, _inp, out):
            logits = (out[0] if isinstance(out, tuple) else out)[0, -1].float()
            probs = logits.softmax(-1)
            topk = probs.topk(min(4, probs.numel()))
            self.capture[("experts", i)] = [[int(a), round(float(b), 4)]
                                            for a, b in zip(topk.indices, topk.values)]
        return hook

    def topology(self, model_id: str) -> dict:
        layer_types = getattr(self.cfg, "layer_types", None)
        n_experts = getattr(self.cfg, "num_local_experts",
                            getattr(self.cfg, "num_experts", 0)) or 0
        topk = getattr(self.cfg, "num_experts_per_tok", 0) or 0
        layers = []
        for i in range(len(self.layers)):
            kind = "attn"
            if layer_types:
                lt = layer_types[i]
                kind = ("conv" if "conv" in lt else
                        "global" if ("full" in lt or "global" in lt) else "local")
            elif hasattr(self.layers[i], "conv"):
                kind = "conv"
            layers.append({
                "i": i,
                "attn": {"kind": kind,
                         "heads": getattr(self.cfg, "num_attention_heads", 0)},
                "mlp": ({"kind": "moe", "experts": n_experts, "topk": topk}
                        if n_experts else {"kind": "dense", "experts": 0, "topk": 0}),
            })
        return {"type": "topology", "model": model_id, "layers": layers}

    @property
    def device(self):
        return next(self.model.parameters()).device

    def generate_stream(self, prompt, max_new_tokens: int,
                        temperature: float = 0.0, tools=None):
        torch = self.torch
        msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        extra = {}                     # first-forward kwargs (pixel_values…)
        tpl_kw = {"tools": tools} if tools else {}
        try:
            if self.processor and any(isinstance(m.get("content"), list)
                                      for m in msgs):
                enc = self.processor.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt", **tpl_kw)
                ids = enc["input_ids"].to(self.device)
                extra = {k: v.to(self.device) for k, v in enc.items()
                         if k not in ("input_ids", "attention_mask")
                         and torch.is_tensor(v)}
            else:
                ids = self.tok.apply_chat_template(
                    msgs, add_generation_prompt=True,
                    return_tensors="pt", **tpl_kw)
                if not torch.is_tensor(ids):  # transformers 5.x BatchEncoding
                    ids = ids["input_ids"]
                ids = ids.to(self.device)
        except Exception:
            ids = self.tok(str(prompt), return_tensors="pt").input_ids.to(
                self.device)
        n_prompt = ids.shape[1]
        past = None
        cur = ids
        with torch.no_grad():
            for t in range(max_new_tokens):
                self.capture.clear()
                out = self.model(cur, past_key_values=past, use_cache=True,
                                 output_attentions=True, **extra)
                extra = {}             # vision inputs only on the prefill
                past = out.past_key_values
                attns = getattr(out, "attentions", None) or []
                logits = out.logits[0, -1].float()
                if temperature > 0:
                    probs = (logits / temperature).softmax(-1)
                    next_id = torch.multinomial(probs, 1)[0]
                else:
                    probs = logits.softmax(-1)
                    next_id = logits.argmax()
                tk = probs.topk(5)
                topk = [[self.tok.decode(i), round(float(p), 4)]
                        for i, p in zip(tk.indices, tk.values)]
                # per-token uncertainty: distribution entropy (bits) + margin
                entropy = float(-(probs.clamp_min(1e-12).log2()
                                  * probs).sum())
                margin = float(tk.values[0] - tk.values[1])
                # logit lens: what each layer's residual would predict
                lens = []
                if self.lens_ok:
                    try:
                        hids = [self.capture[("hid", i)][-1]
                                for i in range(len(self.layers))]
                        lp = self.model.lm_head(
                            self.final_norm(torch.stack(hids))).float().softmax(-1)
                        top = lp.max(-1)
                        lens = [[self.tok.decode(ix), round(float(pv), 4)]
                                for ix, pv in zip(top.indices, top.values)]
                    except Exception:
                        self.lens_ok = False   # arch mismatch — drop silently
                # residual → 3D coords per layer for the trajectory view
                proj = []
                try:
                    hids = torch.stack([self.capture[("hid", i)][-1]
                                        for i in range(len(self.layers))])
                    proj = [[round(float(a), 3) for a in row]
                            for row in hids.float() @ self.proj_basis]
                except Exception:
                    pass
                if t == 0:         # prefill holds every position → context trace
                    try:
                        Hs = torch.stack([self.capture[("hid", i)]
                                          for i in range(len(self.layers))])
                        cp = (Hs.float() @ self.proj_basis).permute(1, 0, 2)
                        yield {"type": "context",
                               "tokens": [self.tok.decode(x) for x in ids[0]],
                               "proj": [[[round(float(v), 3) for v in lay]
                                         for lay in pos] for pos in cp]}
                    except Exception:
                        pass
                # hybrid models compact the attentions tuple (maps only for
                # attention layers) — align it to true layer indices via the
                # layers whose head hooks fired this step
                if len(attns) == len(self.layers):
                    attn_by_layer = dict(enumerate(attns))
                else:
                    attn_by_layer = dict(zip(
                        (i for i in range(len(self.layers))
                         if ("heads", i) in self.capture), attns))
                layers = []
                for i in range(len(self.layers)):
                    entry = {
                        "i": i,
                        "attn_norm": round(self.capture.get(("attn", i), 0.0), 4),
                        "mlp_norm": round(self.capture.get(("mlp", i), 0.0), 4),
                        "resid_norm": round(self.capture.get(("resid", i), 0.0), 4),
                        "head_norms": [round(v, 4) for v in
                                       self.capture.get(("heads", i), [])],
                    }
                    if ("experts", i) in self.capture:
                        entry["expert_weights"] = self.capture[("experts", i)]
                    if ("delta_mean", i) in self.capture:
                        entry["delta_mean"] = self.capture[("delta_mean", i)]
                        entry["delta"] = self.capture[("delta", i)]
                    if attn_by_layer.get(i) is not None:
                        # where this token looked: mean-over-heads attention
                        a = attn_by_layer[i][0, :, -1, :].float()  # [H, S]
                        mean = a.mean(0)
                        S2 = mean.shape[0]
                        ts = mean.topk(min(8, S2))
                        entry["attn_srcs"] = [
                            [int(p), round(float(w), 4)]
                            for p, w in zip(ts.indices, ts.values)]
                        ent = -(mean.clamp_min(1e-12).log() * mean).sum() \
                            / math.log(max(S2, 2))
                        entry["attn_entropy"] = round(float(ent), 4)
                    layers.append(entry)
                is_eos = next_id.item() == self.tok.eos_token_id
                frame = {"type": "token", "t": t,
                         "text": self.tok.decode(next_id),
                         "topk": topk, "layers": layers,
                         "entropy": round(entropy, 3),
                         "margin": round(margin, 4)}
                if lens:
                    frame["lens"] = lens
                if proj:
                    frame["proj"] = proj
                if is_eos:
                    frame["eos"] = True
                yield frame
                if is_eos:
                    break
                cur = next_id.view(1, 1)
        yield {"type": "done", "prompt_tokens": n_prompt, "completion_tokens": t + 1}


async def real_stream(send, args):
    hm = HookedModel(args.model, args.device, args.dtype)
    await send(hm.topology(args.model))
    loop = asyncio.get_event_loop()
    gen = hm.generate_stream(args.prompt, args.max_new_tokens)
    while True:
        msg = await loop.run_in_executor(None, lambda: next(gen, None))
        if msg is None:
            break
        await send(msg)
        if msg["type"] == "done":
            break


# ----------------------------------------------------------------------------

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--prompt", default="Explain how a KV cache works, briefly.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--mock-moe", action="store_true")
    p.add_argument("--mock-layers", type=int, default=24)
    p.add_argument("--mock-tokens", type=int, default=80)
    args = p.parse_args()

    import websockets

    async def handler(ws):
        async def send(obj):
            await ws.send(json.dumps(obj))
        try:
            if args.mock or args.mock_moe:
                await mock_stream(send, args.mock_layers, args.mock_moe, args.mock_tokens)
            else:
                await real_stream(send, args)
        except websockets.ConnectionClosed:
            pass

    async with websockets.serve(handler, "localhost", args.port):
        print(f"gemma-graph: ws://localhost:{args.port}  "
              f"({'MOCK' if args.mock or args.mock_moe else args.model}) — open graph.html")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
