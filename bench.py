#!/usr/bin/env python3
"""
bench.py — agentic tool-use benchmark runner + leaderboard.

Runs the ReAct harness (agent_harness.py) over a suite of graded tasks against
one or MORE OpenAI-compatible models, scores each trajectory with a normalized
(whole-token, comma/unit-tolerant) matcher, and reports a weighted AGENTIC
score (0-100) plus a per-model / per-category leaderboard. Use it to rank
models and to gate a fine-tune before/after.

  # rank one model
  python bench.py suites/agentic.json \
      --models lfm=http://localhost:8080#LiquidAI/LFM2.5-1.2B-Instruct

  # before/after fine-tune gate (two endpoints, same suite)
  python bench.py suites/agentic.json \
      --models base=http://localhost:8080#base ft=http://localhost:8081#ft \
      --min-score 60          # exit != 0 if any model scores below 60

AGENTIC = 100 * (0.45*Capability + 0.25*Precision + 0.20*Discipline + 0.10*Efficiency)
  Capability  = 0.55*task_success + 0.20*grounding + 0.25*hallucination_override
  Precision   = 0.40*tool_precision + 0.35*tool_recall + 0.25*abstention
  Discipline  = 0.60*format_adherence + 0.40*(1 - degeneration)
  Efficiency  = 0.70*step_efficiency + 0.30*throughput
(weights + metric set from the adversarial design review.)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import agent_harness as ah

try:
    import curator                                # optional internals tie-in
except Exception:
    curator = None

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
REPORTS = ROOT / "reports"


# ─────────────────────── normalized matching ────────────────────────────────

def _num_tokens(s):
    """Numeric tokens with thousands-commas stripped: '8,849 m' -> ['8849']."""
    return re.findall(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))


def _canon(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def val_in(text, gold, rel=1e-2, floor=0.05, int_tol=0.5):
    """Whole-token numeric match. INTEGER golds match near-exactly (993==993.0,
    but 1991 != 2000 and 100 != 1000); DECIMAL golds tolerate sensible rounding
    (6.21 ~ 6.2137). Comma-normalized. String golds: canonical substring."""
    text = text or ""
    gs = str(gold).replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", gs):                  # numeric gold
        g = float(gs)
        tol = int_tol if "." not in gs else max(floor, abs(g) * rel)
        for tok in _num_tokens(text):
            try:
                if abs(float(tok) - g) <= tol:
                    return True
            except ValueError:
                pass
        return False
    return _canon(str(gold)) in _canon(text)                  # string gold


def is_grounded(answer, tool_calls):
    """The answer traces to a real tool result: shares a numeric token, or a
    4+ letter word, with a successful Observation."""
    atoks = set(_num_tokens(answer))
    alow = _canon(answer)
    for tc in tool_calls:
        if not tc.get("ok"):
            continue
        res = tc.get("result", "")
        rtoks = _num_tokens(res)
        if rtoks and any(rt in atoks for rt in rtoks):
            return True
        for w in re.findall(r"[a-z]{4,}", res.lower()):
            if w in alow:
                return True
    return False


# ───────────────────────────── scoring ──────────────────────────────────────

def score_case(case, traj):
    exp = case["expect"]
    ans = traj.get("final_answer") or ""
    offered = case.get("tools", [])
    should_call = exp.get("should_call", [])
    should_not = exp.get("should_not_call", [])
    called = [tc["name"] for tc in traj["tool_calls"]]
    called_set = set(called)
    answered = traj["outcome"] == "answered"
    steps = traj["steps"]

    is_abstain = bool(exp.get("abstain"))
    is_did_call = bool(exp.get("grade_did_call_only"))
    is_grounded_case = bool(exp.get("grounded"))
    is_trap = "trap_wrong" in exp
    has_should_call = bool(should_call)

    # task success
    if is_did_call:
        success = answered and all(t in called_set for t in should_call)
    elif is_abstain:
        no_bad = not any(t in called_set for t in should_not)
        declined = any(m in ans.lower() for m in exp.get("decline_markers", []))
        fact = (bool(exp.get("answer_contains")) and
                all(val_in(ans, g) for g in exp.get("answer_contains", [])))
        success = answered and no_bad and (declined or fact)
    else:
        ok = all(val_in(ans, g) for g in exp.get("answer_contains", []))
        ok = ok and not any(val_in(ans, g)
                            for g in exp.get("answer_not_contains", []))
        if exp.get("min_tool_calls"):
            ok = ok and len(called) >= exp["min_tool_calls"]
        success = answered and ok

    grounded_ok = is_grounded(ans, traj["tool_calls"]) if is_grounded_case \
        else True
    override_ok = (all(val_in(ans, g) for g in exp.get("answer_contains", []))
                  and not val_in(ans, exp.get("trap_wrong", "∅"))) \
        if is_trap else True

    precision = 1.0 if not called else sum(
        1 for n in called if n in offered and n not in should_not) / len(called)
    recall = (sum(1 for t in should_call if t in called_set) /
              len(should_call)) if should_call else 1.0
    abstain_ok = (not any(t in called_set for t in should_not)) \
        if is_abstain else 1.0

    parseable = sum(1 for s in steps if s["kind"] in ("act", "answer"))
    fmt = parseable / len(steps) if steps else 1.0
    degen = any(s.get("degenerate") for s in steps)

    optimal = exp.get("optimal_steps", traj["n_steps"])
    efficiency = min(1.0, optimal / max(traj["n_steps"], 1)) if answered else 0.0
    sps = traj["seconds"] / max(traj["n_steps"], 1)
    throughput = max(0.0, min(1.0, 1.0 - (sps - 3) / 57))     # 3s/step->1, 60->0

    status = ("PASS" if (success and grounded_ok and override_ok and
                         (recall == 1.0 if has_should_call else True) and
                         precision == 1.0)
              else "PARTIAL" if success else "FAIL")

    return {
        "id": case["id"], "category": case["category"], "status": status,
        "final_answer": ans[:120], "outcome": traj["outcome"],
        "n_steps": traj["n_steps"], "seconds": traj["seconds"],
        "called": called,
        # metric contributions
        "success": float(success), "grounded_ok": float(grounded_ok),
        "override_ok": float(override_ok), "precision": precision,
        "recall": recall, "abstain_ok": float(abstain_ok), "format": fmt,
        "degen": float(degen), "efficiency": efficiency, "throughput": throughput,
        # applicability flags
        "is_grounded_case": is_grounded_case, "is_trap": is_trap,
        "is_abstain": is_abstain, "has_should_call": has_should_call,
    }


def aggregate(scores):
    n = len(scores)

    def mean(key, flag=None):
        vals = [c[key] for c in scores if (flag is None or c.get(flag))]
        return sum(vals) / len(vals) if vals else 1.0

    task_success = mean("success")
    grounding = mean("grounded_ok", "is_grounded_case")
    override = mean("override_ok", "is_trap")
    tool_precision = mean("precision")
    tool_recall = mean("recall", "has_should_call")
    abstention = mean("abstain_ok", "is_abstain")
    fmt = mean("format")
    degen = sum(c["degen"] for c in scores) / n if n else 0.0
    efficiency = mean("efficiency")
    throughput = mean("throughput")

    cap = 0.55 * task_success + 0.20 * grounding + 0.25 * override
    prec = 0.40 * tool_precision + 0.35 * tool_recall + 0.25 * abstention
    disc = 0.60 * fmt + 0.40 * (1 - degen)
    eff = 0.70 * efficiency + 0.30 * throughput
    agentic = 100 * (0.45 * cap + 0.25 * prec + 0.20 * disc + 0.10 * eff)

    return {
        "AGENTIC": round(agentic, 1),
        "Capability": round(cap, 3), "Precision": round(prec, 3),
        "Discipline": round(disc, 3), "Efficiency": round(eff, 3),
        "task_success": round(task_success, 3),
        "grounding": round(grounding, 3),
        "hallucination_override": round(override, 3),
        "tool_precision": round(tool_precision, 3),
        "tool_recall": round(tool_recall, 3),
        "abstention": round(abstention, 3),
        "format_adherence": round(fmt, 3),
        "degeneration": round(degen, 3),
        "step_efficiency": round(efficiency, 3),
        "pass_rate": round(sum(c["status"] == "PASS" for c in scores) / n, 3),
    }


def by_category(scores):
    cats = {}
    for c in scores:
        cats.setdefault(c["category"], []).append(c["success"])
    return {k: round(sum(v) / len(v), 2) for k, v in sorted(cats.items())}


# ───────────────────────── run a model ──────────────────────────────────────

def run_model(label, base, model, suite, step_cap, native, internals, verbose):
    cur = curator.Curator() if (curator and internals) else None
    scores = []
    print(f"\n[{label}] {model or '(default)'} @ {base} · "
          f"{len(suite['cases'])} cases")
    for case in suite["cases"]:
        try:
            traj = ah.run_agent(base, model, case["task"], case["tools"],
                                step_cap=step_cap, native=native)
        except Exception as e:
            print(f"  ! {case['id']}: harness error: {e}")
            traj = {"final_answer": None, "outcome": "error", "steps": [],
                    "tool_calls": [], "n_steps": 0, "seconds": 0.0}
        sc = score_case(case, traj)
        if cur:
            sc["internals"] = _internals_summary(cur)
        scores.append(sc)
        mark = {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗"}[sc["status"]]
        print(f"  {mark} {case['id']:<30} {sc['outcome']:<14} "
              f"{sc['n_steps']}st {sc['seconds']:>5.1f}s  "
              f"call={','.join(sc['called']) or '-':<22} "
              f"→ {sc['final_answer'][:44]!r}")
    agg = aggregate(scores)
    return {"label": label, "base": base, "model": model,
            "aggregate": agg, "by_category": by_category(scores),
            "cases": scores}


def _internals_summary(cur):
    """Best-effort: pull the last recorded server run's internals signature."""
    try:
        run = cur.load("last")
        res = {r["check"]: r["detail"]
               for r in curator.run_checks(run)}
        return {k: res.get(k) for k in
                ("settle_depth", "confidence", "uncertainty", "repetition")}
    except Exception:
        return None


# ─────────────────────────── leaderboard ────────────────────────────────────

def leaderboard_md(suite, models, step_cap):
    rows = sorted(models, key=lambda m: -m["aggregate"]["AGENTIC"])
    cols = ["AGENTIC", "task_success", "grounding", "hallucination_override",
            "tool_precision", "tool_recall", "abstention", "format_adherence",
            "degeneration", "pass_rate"]
    out = [f"# Agentic benchmark — {suite.get('name', 'agentic')} "
           f"({len(suite['cases'])} cases, step_cap {step_cap})", ""]
    out.append("| model | " + " | ".join(cols) + " |")
    out.append("|" + "---|" * (len(cols) + 1))
    for m in rows:
        a = m["aggregate"]
        cells = [f"**{a['AGENTIC']}**"] + [f"{a[c]:.2f}" if c != "AGENTIC"
                                           else str(a[c]) for c in cols[1:]]
        out.append(f"| {m['label']} ({m['model'] or 'default'}) | "
                   + " | ".join(cells) + " |")
    out.append("")
    out.append("Sub-scores (0–1): "
               + " · ".join(f"{c}" for c in
                            ["Capability", "Precision", "Discipline",
                             "Efficiency"]))
    for m in rows:
        a = m["aggregate"]
        out.append(f"- **{m['label']}**: Cap {a['Capability']} · "
                   f"Prec {a['Precision']} · Disc {a['Discipline']} · "
                   f"Eff {a['Efficiency']}  ·  by-category success: "
                   + ", ".join(f"{k} {v}" for k, v in m["by_category"].items()))
    return "\n".join(out)


# ─────────────────────────────── CLI ────────────────────────────────────────

def parse_model(spec):
    label, _, rest = spec.partition("=")
    base, _, model = rest.partition("#")
    return label, base, model


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("suite")
    ap.add_argument("--models", nargs="+", required=True,
                    help="LABEL=BASE_URL[#MODEL_ID] ...")
    ap.add_argument("--step-cap", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N cases (0 = all)")
    ap.add_argument("--no-native", action="store_true",
                    help="ReAct-text only (skip native tool_calls schema)")
    ap.add_argument("--internals", action="store_true",
                    help="attach recorded internals signature per case (server)")
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="exit != 0 if any model's AGENTIC is below this")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    suite = json.loads(Path(a.suite).read_text())
    if a.limit:
        suite = {**suite, "cases": suite["cases"][:a.limit]}
    models = []
    for spec in a.models:
        label, base, model = parse_model(spec)
        models.append(run_model(label, base, model, suite, a.step_cap,
                                not a.no_native, a.internals, a.verbose))

    md = leaderboard_md(suite, models, a.step_cap)
    print("\n" + md)

    report = {"suite": suite.get("name"), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "step_cap": a.step_cap, "n_cases": len(suite["cases"]),
              "models": models}
    out = a.out or (REPORTS / f"{suite.get('name', 'agentic')}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    Path(str(out).rsplit(".", 1)[0] + ".md").write_text(md)
    print(f"\n[report] {out}")

    worst = min(m["aggregate"]["AGENTIC"] for m in models)
    if worst < a.min_score:
        print(f"GATE FAIL: min AGENTIC {worst} < {a.min_score}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
