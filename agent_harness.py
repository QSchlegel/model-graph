#!/usr/bin/env python3
"""
agent_harness.py — the micro-agent ReAct harness, in Python.

A faithful port of the in-browser harness (web/agent.html): a bounded
tool-use loop (think -> route -> act -> observe -> answer/guard/stop) that
drives ANY OpenAI-compatible model through agentic tool use. Local tools run
here; the harness parses each step (ReAct text OR native tool_calls), runs the
real tool, and feeds the observation back — overriding whatever the model
hallucinated. This is what the benchmark (bench.py) scores.

  from agent_harness import run_agent, TOOLS
  traj = run_agent("http://localhost:8080", "LiquidAI/LFM2.5-1.2B-Instruct",
                   "What is 47*19+100?", ["calculator"])

CLI:
  python agent_harness.py --base http://localhost:8080 \
      --model LiquidAI/LFM2.5-1.2B-Instruct \
      --tools calculator "What is 47 * 19 + 100?"
"""

import argparse
import ast
import json
import operator
import re
import time
import urllib.request

# ───────────────────────────── local tools ──────────────────────────────────
# Deterministic (except clock). Kept behaviourally identical to the JS registry
# so the Python benchmark measures the same harness users watch.

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _num(x):
    """Format like JS String(n): integers without a trailing .0."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return repr(x) if isinstance(x, float) else str(x)


def _reduce(node):                              # tiny arithmetic interpreter
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_reduce(node.left), _reduce(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_reduce(node.operand))
    raise ValueError("only arithmetic is allowed")


def t_calculator(a):
    e = str(a.get("expression", a.get("expr", a.get("_pos", "")))).strip()
    if not re.match(r'^[-+*/%.()\d\seE]+$', e or ""):
        raise ValueError("only arithmetic is allowed")
    try:
        tree = ast.parse(e)                     # mode=exec; no code execution
    except SyntaxError:
        raise ValueError("only arithmetic is allowed")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise ValueError("only arithmetic is allowed")
    v = _reduce(tree.body[0].value)
    if not isinstance(v, (int, float)) or v != v or v in (
            float("inf"), float("-inf")):
        raise ValueError("not a finite number")
    return _num(round(v * 1e6) / 1e6)


_LEN = {"km": 1000, "m": 1, "cm": .01, "mm": .001, "mi": 1609.344,
        "yd": .9144, "ft": .3048, "in": .0254}     # -> metres
_MASS = {"kg": 1000, "g": 1, "mg": .001, "lb": 453.59237, "oz": 28.349523}
_SYN = {"celsius": "c", "centigrade": "c", "fahrenheit": "f", "kelvin": "k",
        "kilometers": "km", "kilometres": "km", "kilometer": "km",
        "kilometre": "km", "miles": "mi", "mile": "mi", "meters": "m",
        "metres": "m", "meter": "m", "metre": "m", "centimeters": "cm",
        "centimetres": "cm", "centimeter": "cm", "millimeters": "mm",
        "feet": "ft", "foot": "ft", "inches": "in", "inch": "in",
        "yards": "yd", "yard": "yd", "kilograms": "kg", "kilogram": "kg",
        "grams": "g", "gram": "g", "milligrams": "mg", "pounds": "lb",
        "pound": "lb", "lbs": "lb", "ounces": "oz", "ounce": "oz"}


def _unit(u):
    u = str(u or "").lower().strip()
    return _SYN.get(u, u)


def t_convert(a):
    val = a.get("value", a.get("_pos"))
    frm = _unit(a.get("from", ""))
    to = _unit(a.get("to", ""))
    try:
        v = float(val)
    except (TypeError, ValueError):
        raise ValueError("value must be a number")
    temp = {"c", "f", "k"}
    if frm in temp and to in temp:
        c = v if frm == "c" else (v - 32) * 5 / 9 if frm == "f" else v - 273.15
        out = c if to == "c" else c * 9 / 5 + 32 if to == "f" else c + 273.15
        return f"{_num(round(out * 100) / 100)} {to}"
    for tbl in (_LEN, _MASS):                       # convert within a dimension
        if frm in tbl and to in tbl:
            return f"{_num(round(v * tbl[frm] / tbl[to] * 1e4) / 1e4)} {to}"
    raise ValueError(f"can't convert {frm}->{to} (units must share a "
                     "dimension: length, mass, or temperature)")


def t_clock(a):
    return time.strftime("%a %b %d %Y %H:%M:%S")


class _RNG:                                     # seeded LCG, mirrors the JS
    def __init__(self):
        self.seed = 1234567

    def reset(self):
        self.seed = 1234567

    def next(self):
        self.seed = (self.seed * 1103515245 + 12345) & 0x7fffffff
        return self.seed / 0x7fffffff


_rng = _RNG()


def t_random_int(a):
    try:
        lo, hi = int(a.get("min")), int(a.get("max"))
    except (TypeError, ValueError):
        raise ValueError("min and max must be numbers")
    if hi < lo:
        lo, hi = hi, lo
    return str(lo + int(_rng.next() * (hi - lo + 1)))


_WX = {"paris": "18°C, partly cloudy", "tokyo": "26°C, humid",
       "london": "14°C, light rain", "new york": "22°C, clear",
       "berlin": "17°C, overcast", "san francisco": "16°C, foggy"}
_KB = {"everest": "Mount Everest is 8,849 m tall — the highest mountain "
       "above sea level.",
       "photosynthesis": "Plants convert light, water and CO2 into glucose "
       "and oxygen.",
       "speed of light": "= 299,792,458 metres per second in a vacuum.",
       "python": "A high-level programming language created by Guido van "
       "Rossum, released 1991."}


def t_weather(a):
    c = str(a.get("city", a.get("_pos", ""))).lower().strip()
    return _WX.get(c, f'no demo data for "{c}"')


def t_lookup(a):
    t = str(a.get("term", a.get("_pos", ""))).lower().strip()
    return _KB.get(t, f'"{t}" is not in the demo knowledge base')


# name -> (fn, signature, description, is_demo)
TOOLS = {
    "calculator": (t_calculator, "calculator(expression)",
                   "evaluate an arithmetic expression, e.g. 47*19+100", False),
    "convert": (t_convert, "convert(value, from, to)",
                "convert units — length (km/mi/m/ft...), mass (kg/lb...), "
                "temp (c/f/k)", False),
    "clock": (t_clock, "clock()", "the current date and time", False),
    "random_int": (t_random_int, "random_int(min, max)",
                   "a reproducible random integer in [min, max]", False),
    "weather": (t_weather, "weather(city)",
                "weather for a city — sandbox data (paris, tokyo, london, "
                "berlin, new york, san francisco)", True),
    "lookup": (t_lookup, "lookup(term)",
               "look up a fact — sandbox data (everest, photosynthesis, "
               "speed of light, python)", True),
}

_FEWSHOT = {
    "calculator": ("What is 5 times 6, plus 2?", "I will use the calculator.",
                   'calculator(expression="5*6+2")', "32", "It is 32."),
    "convert": ("How many miles is 10 kilometers?",
                "I will convert km to miles.",
                "convert(value=10, from=km, to=mi)", "6.2137 mi",
                "10 km is about 6.21 miles."),
    "clock": ("What time is it?", "I will check the clock.", "clock()",
              "Mon Jan 01 2026 09:30:00", "It is 09:30 on Jan 1, 2026."),
    "random_int": ("Roll a die.", "I will pick a number from 1 to 6.",
                   "random_int(min=1, max=6)", "4", "You rolled a 4."),
    "weather": ("What's the weather in Paris?", "I will look up the weather.",
                'weather(city="Paris")', "18°C, partly cloudy",
                "Paris is 18°C and partly cloudy."),
    "lookup": ("How tall is Everest?", "I will look it up.",
               'lookup(term="everest")', "Mount Everest is 8,849 m tall.",
               "Everest is 8,849 m tall."),
}


def system_prompt(tools):
    lst = "\n".join(f"- {TOOLS[k][1]}: {TOOLS[k][2]}" for k in tools)
    ex = _FEWSHOT.get(tools[0], _FEWSHOT["calculator"]) if tools else \
        _FEWSHOT["calculator"]
    return (
        "You are a helpful assistant that can call tools to solve the user's "
        "task. Prefer a tool whenever one fits — do not do arithmetic or recall "
        "a fact yourself if a tool can do it, and base your answer on the tool's "
        "result. But if NO available tool can answer the question, say so "
        "briefly or answer from your own knowledge; never call an unrelated "
        "tool just to call something.\n\n"
        "Work one step at a time. Write a short Thought, then EITHER one "
        "Action OR one Answer. After an Action you are shown an "
        '"Observation" with the result.\n\n'
        f"Tools:\n{lst}\n\n"
        f"Example:\nUser: {ex[0]}\nThought: {ex[1]}\nAction: {ex[2]}\n"
        f"Observation: {ex[3]}\nThought: The tool returned {ex[3]}.\n"
        f"Answer: {ex[4]}\n\n"
        "Now solve the user's task the same way, using only the tools above.")


# ───────────────────────────── parsing ──────────────────────────────────────

def _lit(v):
    v = str(v).strip().rstrip(",").strip()
    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or
                        (v[0] == "'" and v[-1] == "'")):
        return v[1:-1]
    return v


def parse_args(raw):
    raw = str(raw).strip()
    out = {}
    if not raw:
        return out
    kv = re.findall(r'([a-z_]\w*)\s*=\s*("(?:[^"\\]|\\.)*"|'
                    r"'(?:[^'\\]|\\.)*'|[^,]+)", raw, re.I)
    if kv:
        for k, v in kv:
            out[k.lower()] = _lit(v)
        return out
    parts = [_lit(x) for x in raw.split(",")]
    out["_pos"] = parts[0]
    if len(parts) >= 3:
        out["value"], out["from"], out["to"] = parts[0], parts[1], parts[2]
    return out


def parse_step(text):
    """ReAct parse. Truncate at a hallucinated Observation, then take the
    earliest of Action / Answer (mirrors web/agent.html parseStep)."""
    text = text or ""
    cut = re.search(r"\n?\s*observation\s*:", text, re.I)
    t = text[:cut.start()] if cut else text
    m_act = re.search(r"action\s*:\s*([a-z_]\w*)\s*\(([\s\S]*?)\)", t, re.I)
    m_ans = re.search(r"(?:final\s*answer|answer)\s*:\s*([\s\S]+)", t, re.I)
    m_th = re.search(r"thought\s*:\s*([\s\S]*?)(?=\n\s*(?:action|answer|"
                     r"final answer)\s*:|$)", t, re.I)
    thought = re.sub(r"\s+", " ", (m_th.group(1) if m_th else "")).strip()
    i_act = m_act.start() if m_act else float("inf")
    i_ans = m_ans.start() if m_ans else float("inf")
    if m_act and i_act <= i_ans:
        return {"kind": "act", "name": m_act.group(1).lower(),
                "args_raw": m_act.group(2).strip(), "thought": thought,
                "raw": t.strip()}
    if m_ans:
        return {"kind": "answer", "answer": m_ans.group(1).strip(),
                "thought": thought, "raw": t.strip()}
    return {"kind": "none", "thought": thought, "raw": t.strip()}


def looks_degenerate(s):
    s = (s or "").strip()
    if len(s) < 48:
        return False
    probe = s[-28:-8]
    if len(probe) > 7 and s.count(probe) >= 3:
        return True
    w = s.lower().split()
    run = mx = 1
    for i in range(1, len(w)):
        if w[i] == w[i - 1] and len(w[i]) > 1:
            run += 1
            mx = max(mx, run)
        else:
            run = 1
    return mx >= 6


# ─────────────────────────── generation call ────────────────────────────────

def api_gen(base, model, messages, tools_spec=None, max_tokens=200,
            temperature=0.0, timeout=600):
    """One OpenAI-compatible completion. Returns (content, tool_calls)."""
    body = {"messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stop": ["\nObservation:"]}
    if model:
        body["model"] = model
    if tools_spec:
        body["tools"] = tools_spec
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    return msg.get("content") or "", msg.get("tool_calls") or []


_UNITS = list(_LEN) + list(_MASS) + ["c", "f", "k"]
_SCHEMA = {
    "calculator": {"expression": {"type": "string"}},
    "convert": {"value": {"type": "number"},
                "from": {"type": "string", "enum": _UNITS},
                "to": {"type": "string", "enum": _UNITS}},
    "clock": {},
    "random_int": {"min": {"type": "integer"}, "max": {"type": "integer"}},
    "weather": {"city": {"type": "string"}},
    "lookup": {"term": {"type": "string"}},
}


def _openai_tools(tools):
    """OpenAI tool specs with real parameter schemas so native-tool_calls
    models are graded fairly (not against an empty schema)."""
    return [{"type": "function", "function": {
        "name": k, "description": TOOLS[k][2],
        "parameters": {"type": "object",
                       "properties": _SCHEMA.get(k, {}),
                       "required": list(_SCHEMA.get(k, {}))}}} for k in tools]


# ─────────────────────────────── the loop ───────────────────────────────────

def run_agent(base, model, task, tools, step_cap=6, max_tokens=200,
              native=False, verbose=False):
    """Drive one agentic episode. Returns a trajectory dict the benchmark
    scores. `native=True` also offers OpenAI tool specs so tool-trained models
    can emit native tool_calls (parsed if present); otherwise pure ReAct text."""
    _rng.reset()
    tools = [t for t in tools if t in TOOLS] or ["calculator"]
    msgs = [{"role": "system", "content": system_prompt(tools)},
            {"role": "user", "content": task}]
    tools_spec = _openai_tools(tools) if native else None
    traj = {"task": task, "tools": tools, "model": model, "steps": [],
            "tool_calls": [], "final_answer": None, "outcome": "stopped"}
    guards = 0
    t0 = time.time()
    for step in range(1, step_cap + 1):
        content, tcs = api_gen(base, model, msgs, tools_spec, max_tokens)
        if tcs:                                 # native tool_calls win
            fn = tcs[0]["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            p = {"kind": "act", "name": fn["name"].lower(),
                 "args_raw": json.dumps(args), "args": args,
                 "thought": (content or "").strip(), "native": True}
        else:
            p = parse_step(content)
            p["native"] = False
            if p["kind"] == "act":
                p["args"] = parse_args(p["args_raw"])
        rec = {"step": step, "kind": p["kind"], "thought": p.get("thought"),
               "raw": content, "degenerate": looks_degenerate(content),
               "native": p["native"]}
        if verbose:
            print(f"[{step}] {p['kind']}: "
                  f"{(p.get('thought') or content or '')[:80]}")

        if p["kind"] == "answer":
            traj["final_answer"] = p["answer"]
            traj["outcome"] = "answered"
            rec["answer"] = p["answer"]
            traj["steps"].append(rec)
            break

        if p["kind"] == "act":
            guards = 0
            name, args = p["name"], p.get("args", {})
            known = name in TOOLS and name in tools
            try:
                result = str(TOOLS[name][0](args)) if known else \
                    f'error: no tool named "{name}" is enabled'
                ok = known
            except Exception as e:
                result, ok = f"error: {e}", False
            rec["action"] = {"name": name, "args": args, "result": result,
                             "ok": ok}
            traj["tool_calls"].append({"name": name, "args": args,
                                       "result": result, "ok": ok})
            traj["steps"].append(rec)
            # feed the REAL observation back (overrides any hallucination)
            msgs.append({"role": "assistant",
                         "content": f"Thought: {p.get('thought','')}\n"
                         f"Action: {name}({p['args_raw']})"})
            msgs.append({"role": "user", "content": "Observation: " + result})
            if step >= step_cap:
                traj["outcome"] = "stopped_step_cap"
            continue

        # none -> guard
        guards += 1
        rec["guard"] = "repetition_collapse" if rec["degenerate"] else "no_action"
        traj["steps"].append(rec)
        if guards <= 2 and step < step_cap:
            ex = _FEWSHOT.get(tools[0], _FEWSHOT["calculator"])[2]
            msgs.append({"role": "assistant",
                         "content": "Thought: " + (p.get("thought") or
                         "(no valid step)")})
            msgs.append({"role": "user", "content":
                         "That was not in the required format. Reply with ONE "
                         "short step only, for example:\nThought: I will use "
                         f"the tool.\nAction: {ex}\nor:\nThought: I have the "
                         "result.\nAnswer: 42"})
            continue
        traj["outcome"] = "stopped_guard"
        break

    traj["n_steps"] = len(traj["steps"])
    traj["seconds"] = round(time.time() - t0, 2)
    return traj


# ─────────────────────────────── CLI ────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("task")
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--model", default="")
    ap.add_argument("--tools", default="calculator",
                    help="comma-separated tool names")
    ap.add_argument("--step-cap", type=int, default=6)
    ap.add_argument("--native", action="store_true",
                    help="also offer OpenAI tool specs (native tool_calls)")
    a = ap.parse_args()
    traj = run_agent(a.base, a.model, a.task, a.tools.split(","),
                     step_cap=a.step_cap, native=a.native, verbose=True)
    print(json.dumps({k: traj[k] for k in
                      ("outcome", "final_answer", "n_steps", "seconds",
                       "tool_calls")}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
