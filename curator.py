#!/usr/bin/env python3
"""
curator.py — hook-based context curator for model-graph.

The forward hooks in server.py already produce a complete record of every
generation (topology, prompt context, per-token internals). The curator
captures each run as a durable artifact, keeps a queryable library, and
offers testing + automation on top:

  python curator.py list                     # recorded runs
  python curator.py show  <id|last>          # metadata + summary
  python curator.py test  <id|last>          # built-in checks on internals
  python curator.py compare <a> <b>          # regression diff of two runs
  python curator.py run "prompt" [--max 64]  # drive the live api, then test
  python curator.py suite suites/smoke.json [--on-fail CMD]

Recording happens server-side: api_server saves every request's messages
+ frames under runs/ (disable with --no-curate). REST mirror:
  GET  /curator/runs            GET /curator/runs/{id}
  POST /curator/runs/{id}/test

Python automation API:
  cur = Curator(); cur.on("run_saved", fn); cur.on("test_fail", fn)
Events fire with keyword payloads; exceptions in hooks are swallowed.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path

RUNS = Path(os.path.dirname(os.path.abspath(__file__))) / "runs"


# ─────────────────────────────── storage ────────────────────────────────────

class Curator:
    def __init__(self, runs_dir=RUNS):
        self.dir = Path(runs_dir)
        self.dir.mkdir(exist_ok=True)
        self._hooks = defaultdict(list)

    # event hooks — the automation surface
    def on(self, event, fn):
        self._hooks[event].append(fn)
        return fn

    def _emit(self, event, **kw):
        for fn in self._hooks[event]:
            try:
                fn(**kw)
            except Exception:
                pass

    @property
    def index_path(self):
        return self.dir / "index.json"

    def _index(self):
        try:
            return json.loads(self.index_path.read_text())
        except Exception:
            return []

    def save_run(self, meta, messages):
        rid = time.strftime("r-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        with open(self.dir / f"{rid}.ndjson", "w") as f:
            f.write(json.dumps({"type": "meta", **meta}) + "\n")
            for m in messages:
                f.write(json.dumps(m) + "\n")
        idx = self._index()
        idx.append({"id": rid, "ts": meta.get("ts"),
                    "model": meta.get("model"),
                    "prompt": str(meta.get("messages", [{}])[-1]
                                  .get("content", ""))[:80],
                    "tokens": meta.get("usage", {}).get("completion_tokens"),
                    "finish": meta.get("finish")})
        self.index_path.write_text(json.dumps(idx, indent=1))
        self._emit("run_saved", run_id=rid, meta=meta)
        return rid

    def resolve(self, rid):
        idx = self._index()
        if not idx:
            raise SystemExit("no recorded runs — send a request first")
        if rid == "last":
            return idx[-1]["id"]
        hits = [e["id"] for e in idx if e["id"].startswith(rid)]
        if len(hits) != 1:
            raise SystemExit(f"run id {rid!r}: {len(hits)} matches")
        return hits[0]

    def load(self, rid):
        rid = self.resolve(rid)
        run = {"id": rid, "meta": {}, "topology": None, "context": None,
               "frames": [], "done": None}
        with open(self.dir / f"{rid}.ndjson") as f:
            for line in f:
                m = json.loads(line)
                t = m.get("type")
                if t == "meta":
                    run["meta"] = m
                elif t == "topology":
                    run["topology"] = m
                elif t == "context":
                    run["context"] = m
                elif t == "token":
                    run["frames"].append(m)
                elif t == "done":
                    run["done"] = m
        return run


# ─────────────────────────────── checks ─────────────────────────────────────
# Each check: fn(run) → (ok: bool|None, detail: str).  ok=None → metric only.

CHECKS = {}


def check(fn):
    CHECKS[fn.__name__] = fn
    return fn


def _norms(run):
    for f in run["frames"]:
        for l in f.get("layers", []):
            for k in ("attn_norm", "mlp_norm", "resid_norm"):
                yield l.get(k)


@check
def finite_norms(run):
    import math
    bad = sum(1 for v in _norms(run)
              if v is None or math.isnan(v) or math.isinf(v))
    return bad == 0, f"{bad} non-finite norm values"


@check
def frames_consistent(run):
    n = len(run["topology"]["layers"]) if run["topology"] else 0
    seq = all(f["t"] == i for i, f in enumerate(run["frames"]))
    lay = all(len(f.get("layers", [])) in (0, n) for f in run["frames"])
    return seq and lay, (f"{len(run['frames'])} frames, "
                         f"sequential={seq}, layer-count-ok={lay}")


@check
def depth_trend(run):
    frames = [f for f in run["frames"] if f.get("layers")]
    if not frames:
        return None, "no layer data (browser run?)"
    lo = sum(f["layers"][0]["resid_norm"] for f in frames) / len(frames)
    hi = sum(f["layers"][-1]["resid_norm"] for f in frames) / len(frames)
    return hi > lo, f"resid L0 avg {lo:.2f} → Llast avg {hi:.2f}"


@check
def eos_terminated(run):
    fin = run["meta"].get("finish")
    return fin == "stop", f"finish_reason={fin}"


@check
def settle_depth(run):
    n = len(run["topology"]["layers"]) if run["topology"] else 0
    depths = []
    for f in run["frames"]:
        lens = f.get("lens")
        if not lens:
            continue
        for i, (w, _p) in enumerate(lens):
            if w == f["text"]:
                depths.append(i)
                break
        else:
            depths.append(n)
    if not depths:
        return None, "no lens data"
    return None, (f"mean {sum(depths)/len(depths):.1f} · "
                  f"max {max(depths)} of {n} layers")


@check
def confidence(run):
    ps = [f["topk"][0][1] for f in run["frames"] if f.get("topk")]
    if not ps:
        return None, "no topk data"
    return None, (f"mean top-1 prob {sum(ps)/len(ps):.2f} · "
                  f"min {min(ps):.2f}")


@check
def uncertainty(run):
    es = [f["entropy"] for f in run["frames"] if "entropy" in f]
    if not es:
        return None, "no entropy data"
    return None, (f"mean {sum(es)/len(es):.2f} bits · "
                  f"max {max(es):.2f} at t={es.index(max(es))}")


@check
def attention_focus(run):
    sp = [l["attn_entropy"] for f in run["frames"]
          for l in f.get("layers", []) if "attn_entropy" in l]
    if not sp:
        return None, "no attention-map data"
    return None, (f"mean spread {sum(sp)/len(sp)*100:.0f}% · "
                  f"most focused layer-step {min(sp)*100:.0f}%")


@check
def repetition(run):
    toks = [f["text"] for f in run["frames"]]
    grams = defaultdict(int)
    for i in range(len(toks) - 3):
        grams[tuple(toks[i:i + 4])] += 1
    worst = max(grams.values(), default=0)
    return worst <= 3, f"most repeated 4-gram × {worst}"


@check
def throughput(run):
    d = run["meta"].get("duration_s") or 0
    n = len(run["frames"])
    if not d:
        return None, "no timing metadata"
    return None, f"{n} tok in {d}s · {n/d:.1f} tok/s"


def run_checks(run, cur=None):
    results = []
    for name, fn in CHECKS.items():
        try:
            ok, detail = fn(run)
        except Exception as e:
            ok, detail = False, f"check crashed: {e}"
        results.append({"check": name, "ok": ok, "detail": detail})
        if ok is False and cur:
            cur._emit("test_fail", run_id=run["id"], check=name,
                      detail=detail)
    if cur:
        cur._emit("test_done", run_id=run["id"], results=results)
    return results


def print_results(rid, results):
    print(f"[test] {rid}")
    fails = 0
    for r in results:
        mark = "·" if r["ok"] is None else ("✓" if r["ok"] else "✗")
        fails += r["ok"] is False
        print(f"  {mark} {r['check']:<18} {r['detail']}")
    print("PASS" if fails == 0 else f"FAIL ({fails})")
    return fails


# ─────────────────────────────── compare ────────────────────────────────────

def compare(cur, a, b):
    ra, rb = cur.load(a), cur.load(b)
    ta = [f["text"] for f in ra["frames"]]
    tb = [f["text"] for f in rb["frames"]]
    div = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y),
               min(len(ta), len(tb)) if len(ta) != len(tb) else None)
    if div is None:
        dtxt = "none"
    else:
        xa = repr(ta[div]) if div < len(ta) else "∅"
        xb = repr(tb[div]) if div < len(tb) else "∅"
        dtxt = f"t={div} {xa} vs {xb}"
    print(f"[compare] {ra['id']} vs {rb['id']}")
    print(f"  tokens: {len(ta)} vs {len(tb)} · first divergence: {dtxt}")
    fa = [f for f in ra["frames"] if f.get("layers")]
    fb = [f for f in rb["frames"] if f.get("layers")]
    if fa and fb and ra["topology"] and len(ra["topology"]["layers"]) == \
            len(rb["topology"]["layers"]):
        n = len(ra["topology"]["layers"])
        m = min(len(fa), len(fb))
        drift = []
        for i in range(n):
            av = sum(f["layers"][i]["resid_norm"] for f in fa[:m]) / m
            bv = sum(f["layers"][i]["resid_norm"] for f in fb[:m]) / m
            drift.append(abs(av - bv) / max(bv, 1e-9))
        print(f"  resid-norm drift over {m} shared frames: "
              f"mean {sum(drift)/n*100:.1f}% · worst L"
              f"{drift.index(max(drift))} {max(drift)*100:.1f}%")
    else:
        print("  (no comparable layer data)")
    return 0


# ─────────────────────────── live run + suites ──────────────────────────────

def api_chat(base, messages, max_tokens, temperature=0.0, tools=None):
    body = {"messages": messages, "max_tokens": max_tokens,
            "temperature": temperature}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def eval_tool_case(base, case):
    """Tool-reasoning case: should the model call, with the right args,
    and can it ground a final answer on the result?  Returns issue list."""
    exp = case.get("expect", {})
    mt = exp.get("max_tokens", 96)
    prompt = case["prompt"]
    out = api_chat(base, [{"role": "user", "content": prompt}], mt,
                   tools=case["tools"])
    choice = out["choices"][0]
    tcs = choice["message"].get("tool_calls") or []
    reply = choice["message"].get("content") or ""
    bad = []
    if exp.get("no_tool_call"):
        if tcs:
            bad.append(f"called {tcs[0]['function']['name']} needlessly")
        for needle in exp.get("contains", []):
            if needle.lower() not in reply.lower():
                bad.append(f"missing {needle!r}")
        return bad
    want = exp.get("tool_call", {})
    if not tcs:
        return [f"no tool call (answered: {reply[:40]!r})"]
    fn = tcs[0]["function"]
    if want.get("name") and fn["name"] != want["name"]:
        bad.append(f"called {fn['name']}, wanted {want['name']}")
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except Exception:
        args = {}
        bad.append("unparseable arguments")
    for k, v in (want.get("args_include") or {}).items():
        got = args.get(k)
        ok = (str(v).lower() in str(got).lower()
              if isinstance(v, str) else got == v)
        if not ok:
            bad.append(f"arg {k}={got!r} != {v!r}")
    if case.get("tool_result") is not None and not bad:
        msgs2 = [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": None, "tool_calls": tcs},
                 {"role": "tool", "content": case["tool_result"]}]
        out2 = api_chat(base, msgs2, mt, tools=case["tools"])
        reply2 = out2["choices"][0]["message"].get("content") or ""
        for needle in case.get("then_expect", {}).get("contains", []):
            if needle.lower() not in reply2.lower():
                bad.append(f"final answer missing {needle!r} "
                           f"({reply2[:40]!r})")
    return bad


def cmd_run(cur, args):
    print(f"[run] → {args.base}  {args.prompt!r}")
    out = api_chat(args.base, [{"role": "user", "content": args.prompt}],
                   args.max)
    print(f"  reply: {out['choices'][0]['message']['content'][:100]!r}")
    time.sleep(0.3)                            # server flushes the record
    rid = cur.resolve("last")
    return print_results(rid, run_checks(cur.load(rid), cur))


def cmd_suite(cur, args):
    spec = json.loads(Path(args.file).read_text())
    fails = 0
    if args.on_fail:
        cur.on("test_fail", lambda **kw: subprocess.run(
            args.on_fail, shell=True,
            env={**os.environ, "RUN_ID": kw["run_id"],
                 "CHECK": kw["check"], "DETAIL": kw["detail"]}))
    print(f"[suite] {spec.get('name', args.file)} · "
          f"{len(spec['prompts'])} prompts → {args.base}")
    for case in spec["prompts"]:
        prompt = case["prompt"]
        exp = case.get("expect", {})
        if case.get("tools"):               # tool-reasoning case
            bad = eval_tool_case(args.base, case)
            time.sleep(0.3)
            run = cur.load("last")
            mark = "✗" if bad else "✓"
            fails += bool(bad)
            print(f"  {mark} {prompt[:50]:<52} {run['id']}"
                  + (f"  [{'; '.join(bad)}]" if bad else ""))
            if bad:
                cur._emit("test_fail", run_id=run["id"],
                          check=";".join(bad), detail=prompt)
            continue
        out = api_chat(args.base, [{"role": "user", "content": prompt}],
                       exp.get("max_tokens", 128))
        reply = out["choices"][0]["message"]["content"]
        time.sleep(0.3)
        run = cur.load("last")
        results = run_checks(run, cur)
        bad = [r["check"] for r in results if r["ok"] is False]
        for needle in exp.get("contains", []):
            if needle.lower() not in reply.lower():
                bad.append(f"contains:{needle!r}")
        for needle in exp.get("not_contains", []):
            if needle.lower() in reply.lower():
                bad.append(f"not_contains:{needle!r}")
        if exp.get("eos") and out["choices"][0]["finish_reason"] != "stop":
            bad.append("eos")
        st = next((r["detail"] for r in results
                   if r["check"] == "settle_depth"), "")
        if "max_settle_mean" in exp and "mean" in st:
            if float(st.split("mean ")[1].split(" ")[0]) > \
                    exp["max_settle_mean"]:
                bad.append(f"settle_depth>{exp['max_settle_mean']}")
        mark = "✗" if bad else "✓"
        fails += bool(bad)
        print(f"  {mark} {prompt[:50]:<52} {run['id']}"
              + (f"  [{', '.join(bad)}]" if bad else ""))
        if bad:
            cur._emit("test_fail", run_id=run["id"],
                      check=",".join(bad), detail=prompt)
    print("PASS" if fails == 0 else f"FAIL ({fails}/{len(spec['prompts'])})")
    return fails


# ─────────────────────────────── CLI ────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--base", default="http://localhost:8080")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for c in ("show", "test"):
        sub.add_parser(c).add_argument("run_id", nargs="?", default="last")
    p = sub.add_parser("compare")
    p.add_argument("a"), p.add_argument("b")
    p = sub.add_parser("run")
    p.add_argument("prompt"), p.add_argument("--max", type=int, default=96)
    p = sub.add_parser("suite")
    p.add_argument("file"), p.add_argument("--on-fail", default=None)
    args = ap.parse_args()
    cur = Curator()

    if args.cmd == "list":
        for e in cur._index():
            print(f"{e['id']}  {e.get('ts','')}  "
                  f"{e.get('tokens','?'):>4} tok  {e.get('finish','')}  "
                  f"{e.get('prompt','')!r}")
        return 0
    if args.cmd == "show":
        run = cur.load(args.run_id)
        m = run["meta"]
        print(json.dumps({k: m[k] for k in
                          ("ts", "model", "params", "usage", "finish",
                           "duration_s") if k in m}, indent=1))
        print(f"frames: {len(run['frames'])} · context: "
              f"{len(run['context']['tokens']) if run['context'] else 0} tok")
        print(f"reply: {m.get('reply','')[:200]!r}")
        return 0
    if args.cmd == "test":
        run = cur.load(args.run_id)
        return print_results(run["id"], run_checks(run, cur))
    if args.cmd == "compare":
        return compare(cur, args.a, args.b)
    if args.cmd == "run":
        return cmd_run(cur, args)
    if args.cmd == "suite":
        return cmd_suite(cur, args)


if __name__ == "__main__":
    raise SystemExit(main())
