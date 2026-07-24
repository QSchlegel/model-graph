#!/usr/bin/env python3
"""
test_agent.py — CI gate for the agentic harness + benchmark.

Locks down (1) the local-tool GOLDEN VALUES — the registry contract the JS
(web/agent.html, web/chat.html) MUST also satisfy, so the two harnesses can't
drift silently (the km->kg dimension bug that flipped mid-review is exactly
this failure); (2) the normalized whole-token matcher that stops the '100'-in-
'1000' gameability; (3) scoring PASS/FAIL/abstain logic. No server needed.

  python test_agent.py            # exit 0 = pass, non-zero = fail
"""

import agent_harness as ah
import bench as b

# ── the tool golden contract (JS must produce byte-identical results) ─────────
TOOL_GOLDENS = [
    ("calculator", {"expression": "47*19+100"}, "993"),
    ("calculator", {"expression": "3847+9682*2"}, "23211"),
    ("calculator", {"expression": "4567*6183"}, "28237761"),
    ("calculator", {"expression": "1000/10"}, "100"),
    ("calculator", {"expression": "256*256"}, "65536"),
    ("calculator", {"expression": "62.1371/2"}, "31.06855"),
    ("convert", {"value": 100, "from": "celsius", "to": "fahrenheit"}, "212 f"),
    ("convert", {"value": 212, "from": "f", "to": "c"}, "100 c"),
    ("convert", {"value": 10, "from": "km", "to": "mi"}, "6.2137 mi"),
    ("convert", {"value": 2, "from": "kg", "to": "g"}, "2000 g"),
    ("convert", {"value": 98.6, "from": "fahrenheit", "to": "celsius"}, "37 c"),
    ("convert", {"value": 5, "from": "miles", "to": "km"}, "8.0467 km"),
    ("convert", {"value": 1.5, "from": "kg", "to": "lb"}, "3.3069 lb"),
    ("convert", {"value": 8849, "from": "m", "to": "ft"}, "29032.1522 ft"),
    ("convert", {"value": 8849, "from": "m", "to": "cm"}, "884900 cm"),
    ("convert", {"value": 70, "from": "kg", "to": "lb"}, "154.3236 lb"),
    ("random_int", {"min": 1, "max": 6}, "6"),   # seeded LCG, reset per call
    ("weather", {"city": "London"}, "14°C, light rain"),
    ("weather", {"city": "tokyo"}, "26°C, humid"),
    ("lookup", {"term": "python"},
     "A high-level programming language created by Guido van Rossum, "
     "released 1991."),
]

# tools that MUST raise (error-recovery cases depend on it)
TOOL_ERRORS = [
    ("convert", {"value": 5, "from": "km", "to": "kg"}),      # cross-dimension
    ("convert", {"value": 5, "from": "c", "to": "km"}),       # temp vs length
    ("calculator", {"expression": "the sum of apples"}),      # non-arithmetic
]


def test_tool_goldens():
    for name, args, gold in TOOL_GOLDENS:
        ah._rng.reset()                          # random_int is deterministic
        got = str(ah.TOOLS[name][0](args))
        assert got == gold, f"{name}({args}) = {got!r}, expected {gold!r}"


def test_tool_errors():
    for name, args in TOOL_ERRORS:
        try:
            ah.TOOLS[name][0](args)
            raise AssertionError(f"{name}({args}) should have raised")
        except ValueError:
            pass


def test_matcher_whole_token():
    assert b.val_in("The result is 993.", "993")
    assert not b.val_in("29931 apples", "993")          # no substring win
    assert not b.val_in("It is 1000", "100")            # 100 != 1000
    assert b.val_in("It is 100", "100")
    assert b.val_in("8,849 metres", "8849")             # comma-normalized
    assert b.val_in("299,792,458 m/s", "299,792,458")
    assert b.val_in("about 6.21 miles", "6.2137")       # decimal rounding OK
    assert not b.val_in("about 62.1371 miles", "6.2137")  # 10x off, not OK
    assert b.val_in("993.0", "993")                     # int == int.0
    assert not b.val_in("released in 2000", "1991")     # integers are exact
    assert b.val_in("Rome is the capital", "Rome")
    assert not b.val_in("Naples", "Rome")


def test_grounding():
    assert b.is_grounded("It is 993.",
                         [{"name": "calculator", "result": "993", "ok": True}])
    assert not b.is_grounded("It is 993.",
                             [{"name": "calculator", "result": "31", "ok": True}])
    assert b.is_grounded("light rain expected",
                         [{"name": "weather", "result": "14C light rain",
                           "ok": True}])


def test_scoring():
    case = {"id": "c", "category": "single-tool-arithmetic",
            "tools": ["calculator", "weather"],
            "expect": {"answer_contains": ["993"], "should_call": ["calculator"],
                       "should_not_call": ["weather"], "min_tool_calls": 1,
                       "grounded": True, "optimal_steps": 2}}
    win = {"final_answer": "The result is 993.", "outcome": "answered",
           "n_steps": 2, "seconds": 21.0,
           "tool_calls": [{"name": "calculator", "result": "993", "ok": True}],
           "steps": [{"kind": "act", "degenerate": False},
                     {"kind": "answer", "degenerate": False}]}
    assert b.score_case(case, win)["status"] == "PASS"
    wrong = dict(win, tool_calls=[{"name": "weather", "result": "18C",
                                   "ok": True}], final_answer="18C")
    sc = b.score_case(case, wrong)
    assert sc["status"] == "FAIL" and sc["precision"] == 0.0

    abst = {"id": "a", "category": "abstain", "tools": ["weather", "calculator"],
            "expect": {"answer_contains": ["Rome"],
                       "decline_markers": ["can't", "cannot"],
                       "should_not_call": ["weather", "calculator"],
                       "abstain": True, "optimal_steps": 1}}
    good = {"final_answer": "The capital of Italy is Rome.",
            "outcome": "answered", "n_steps": 1, "seconds": 5.0,
            "tool_calls": [], "steps": [{"kind": "answer", "degenerate": False}]}
    assert b.score_case(abst, good)["success"] == 1.0
    bad = dict(good, tool_calls=[{"name": "weather", "result": "x", "ok": True}])
    assert b.score_case(abst, bad)["abstain_ok"] == 0.0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {e}")
    print("PASS" if not failed else f"FAIL ({failed}/{len(tests)})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
