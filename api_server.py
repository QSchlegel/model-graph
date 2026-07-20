#!/usr/bin/env python3
"""
api_server.py — OpenAI-compatible chat endpoint in front of the hooked model,
broadcasting per-token layer stats to the dashboard websocket in real time.

Point anything that speaks the OpenAI API at http://localhost:8080/v1 and
watch every request's internals stream into web/dashboard.html.

  pip install aiohttp websockets            # mock mode needs nothing else
  python api_server.py --mock               # LFM-like fake model, instant start
  python api_server.py --model LiquidAI/LFM2.5-1.2B-Instruct --device mps

Endpoints:
  GET  /             chat window (web/chat.html)
  GET  /dashboard    node-graph dashboard (web/dashboard.html)
  GET  /v1/models
  POST /v1/chat/completions   {messages, [stream], [max_tokens], [temperature]}
  ws://localhost:8765         dashboard protocol: topology → token* → done
                              (topology is re-sent at the start of each request)

Smoke test:
  curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"why is the sky blue?"}]}'
"""

import argparse
import asyncio
import json
import os
import random
import time
import uuid

import curator as curator_mod
import server as server_mod


def _load_image(url):
    """data: URL or http(s) URL → PIL image (multimodal content parts)."""
    import base64
    import io
    import urllib.request
    from PIL import Image
    if url.startswith("data:"):
        raw = base64.b64decode(url.split(",", 1)[1])
    else:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read()
    return Image.open(io.BytesIO(raw)).convert("RGB")


def to_hf_messages(messages):
    """OpenAI wire messages → HF chat-template messages (images decoded,
    assistant tool_calls re-serialized into LFM2's pythonic call format)."""
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if p.get("type") == "image_url":
                    u = p["image_url"]
                    parts.append({"type": "image", "image":
                                  _load_image(u["url"] if isinstance(u, dict)
                                              else u)})
                else:
                    parts.append({"type": "text",
                                  "text": p.get("text", "")})
            content = parts
        mm = {"role": role, "content": content if content is not None else ""}
        if role == "assistant" and m.get("tool_calls"):
            inner = ", ".join(
                tc["function"]["name"] + "(" + ", ".join(
                    f"{k}={v!r}" for k, v in json.loads(
                        tc["function"].get("arguments") or "{}").items()) + ")"
                for tc in m["tool_calls"])
            mm["content"] = (mm["content"] or "") + \
                f"<|tool_call_start|>[{inner}]<|tool_call_end|>"
        out.append(mm)
    return out

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


# ----------------------------------------------------------------------------
# Websocket hub — fan-out of protocol frames to connected dashboards.
# ----------------------------------------------------------------------------

class Hub:
    """Fan-out to ws clients. Clients are send-callables (`await send(str)`)
    so the :8765 `websockets` server and aiohttp's same-port /ws endpoint
    (used behind single-port proxies like Railway) share one hub."""

    def __init__(self):
        self.clients = set()
        self.topology = None       # greeting fallback before the first run
        self.replay = []           # last run (topology + frames + done) —
                                   # greets late-joining dashboards with context

    async def add(self, send):
        self.clients.add(send)
        for msg in (self.replay or
                    ([self.topology] if self.topology else [])):
            await send(json.dumps(msg))

    def drop(self, send):
        self.clients.discard(send)

    async def cast(self, obj):
        if obj.get("type") == "topology":
            self.replay = [obj]    # new run — restart the greeting buffer
        elif self.replay:
            self.replay.append(obj)
        msg = json.dumps(obj)
        for send in list(self.clients):
            try:
                await send(msg)
            except Exception:
                self.drop(send)


# ----------------------------------------------------------------------------
# Mock backend — exercise the API wiring without loading weights.
# ----------------------------------------------------------------------------

MOCK_KINDS = "CCACCACCACACACAC"   # LFM2.5-1.2B layer layout
MOCK_REPLY = ("This is a mock reply from api_server — the OpenAI endpoint and "
              "dashboard broadcast are live; start with --model for real "
              "internals.").split()


def mock_topology() -> dict:
    layers = [{"i": i,
               "attn": ({"kind": "conv", "heads": 0} if k == "C"
                        else {"kind": "global", "heads": 32}),
               "mlp": {"kind": "dense", "experts": 0, "topk": 0}}
              for i, k in enumerate(MOCK_KINDS)]
    return {"type": "topology", "model": "mock-lfm-16L", "layers": layers}


def mock_stream(messages, max_new_tokens: int, _temperature: float):
    n = min(len(MOCK_REPLY), max_new_tokens)
    pos = [[0.0, 0.0, 0.0] for _ in MOCK_KINDS]   # per-layer 3D walk
    ctx_tokens = (["<|im_start|>", "user"]
                  + str(messages[-1].get("content", ""))[:80].split()
                  + ["<|im_end|>"])
    cw, ctx_proj = [0.0, 0.0, 0.0], []
    for _tok in ctx_tokens:
        cw = [p + random.uniform(-1, 1) for p in cw]
        ctx_proj.append([[round(p + i / 8, 3) for p in cw]
                         for i in range(len(MOCK_KINDS))])
    yield {"type": "context", "tokens": ctx_tokens, "proj": ctx_proj}
    for t in range(n):
        layers = []
        for i, k in enumerate(MOCK_KINDS):
            depth = 0.3 + 3.0 * i / len(MOCK_KINDS)
            attn = depth * random.uniform(0.5, 1.5)
            layers.append({
                "i": i,
                "attn_norm": round(attn, 4),
                "mlp_norm": round(depth * random.uniform(0.6, 1.8), 4),
                "resid_norm": round(depth * random.uniform(2.0, 3.0), 4),
                "head_norms": ([] if k == "C" else
                               [round(attn * random.uniform(0.5, 1.5), 4)
                                for _ in range(32)]),
            })
        text = ("" if t == 0 else " ") + MOCK_REPLY[t]
        filler = [" the", " a", " and", " of", " to"]
        settle = 4 + t % 8                  # lens converges partway down
        lens = [([filler[(t + i) % len(filler)], round(random.uniform(.05, .3), 3)]
                 if i < settle else [text, round(random.uniform(.4, .95), 3)])
                for i in range(len(MOCK_KINDS))]
        proj = []
        for i in range(len(MOCK_KINDS)):
            pos[i] = [p + random.uniform(-1, 1) * (0.4 + i / 6)
                      for p in pos[i]]
            proj.append([round(p, 3) for p in pos[i]])
        yield {"type": "token", "t": t, "text": text,
               "topk": [[text, 0.62], [" the", 0.11], [" a", 0.07],
                        ["\n", 0.04], [" and", 0.03]],
               "lens": lens, "proj": proj, "layers": layers}
        time.sleep(0.12)
    yield {"type": "done",
           "prompt_tokens": sum(len(str(m.get("content", "")).split())
                                for m in messages),
           "completion_tokens": n}


# ----------------------------------------------------------------------------
# Backend — one loaded model (or mock), one generation at a time.
# ----------------------------------------------------------------------------

class Backend:
    def __init__(self, args):
        self.lock = asyncio.Lock()
        if args.mock:
            self.hm = None
            self.model_id = "mock-lfm-16L"
            self.topo = mock_topology()
        else:
            from server import HookedModel
            self.hm = HookedModel(args.model, args.device, args.dtype)
            self.model_id = args.model
            self.topo = self.hm.topology(args.model)

    def stream(self, messages, max_tokens: int, temperature: float,
               tools=None):
        if self.hm:
            return self.hm.generate_stream(messages, max_tokens, temperature,
                                           tools=tools)
        return mock_stream(messages, max_tokens, temperature)


# ----------------------------------------------------------------------------
# OpenAI-compatible HTTP layer.
# ----------------------------------------------------------------------------

CORS = {"Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "content-type, authorization",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS"}


def make_app(backend: Backend, hub: Hub, cur=None):
    from aiohttp import web

    def chunk(cid, created, delta, finish=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": backend.model_id,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish}]}

    async def chat_completions(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "invalid JSON body"}},
                                     status=400, headers=CORS)
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return web.json_response({"error": {"message": "messages required"}},
                                     status=400, headers=CORS)
        max_tokens = int(body.get("max_tokens")
                         or body.get("max_completion_tokens") or 256)
        temperature = float(body.get("temperature") or 0.0)
        stream = bool(body.get("stream"))
        tools = body.get("tools")
        try:
            hf_messages = to_hf_messages(messages)
        except Exception as e:
            return web.json_response(
                {"error": {"message": f"bad message content: {e}"}},
                status=400, headers=CORS)
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        loop = asyncio.get_event_loop()

        preview = str(messages[-1].get("content", ""))[:60]
        print(f"[{time.strftime('%H:%M:%S')}] chat({len(messages)} msgs, "
              f"max={max_tokens}, T={temperature}, stream={stream}): {preview!r}")

        resp = None
        if stream:
            resp = web.StreamResponse(
                headers={"Content-Type": "text/event-stream",
                         "Cache-Control": "no-cache", **CORS})
            await resp.prepare(request)
            await resp.write(b"data: " + json.dumps(
                chunk(cid, created, {"role": "assistant", "content": ""})
            ).encode() + b"\n\n")

        text, usage, finish = [], {}, "length"
        t_start = time.time()
        rec = [backend.topo]               # curator: full run context
        async with backend.lock:           # one generation at a time
            await hub.cast(backend.topo)   # reset dashboards for this run
            gen = backend.stream(hf_messages, max_tokens, temperature,
                                 tools=tools)
            while True:
                msg = await loop.run_in_executor(None, lambda: next(gen, None))
                if msg is None:
                    break
                await hub.cast(msg)
                rec.append(msg)
                if msg["type"] == "done":
                    usage = {"prompt_tokens": msg.get("prompt_tokens", 0),
                             "completion_tokens": msg.get("completion_tokens", 0)}
                    break
                if msg["type"] != "token":  # context etc → dashboards only
                    continue
                if msg.get("eos"):
                    finish = "stop"
                    continue               # eos text stays off the API reply
                text.append(msg["text"])
                if resp:
                    await resp.write(b"data: " + json.dumps(
                        chunk(cid, created, {"content": msg["text"]})
                    ).encode() + b"\n\n")

        usage["total_tokens"] = usage.get("prompt_tokens", 0) + \
            usage.get("completion_tokens", 0)
        full = "".join(text)
        tool_calls = None
        if tools:
            tcs = server_mod.parse_tool_calls(full)
            if tcs:
                finish = "tool_calls"
                tool_calls = [{"id": "call_" + uuid.uuid4().hex[:8],
                               "type": "function",
                               "function": {"name": c["name"], "arguments":
                                            json.dumps(c["arguments"])}}
                              for c in tcs]
                full = server_mod.TOOL_CALL_RE.sub("", full).strip()
        if cur:                            # curate the run (hook-fed record)
            cur.save_run(
                {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "model": backend.model_id, "messages": messages,
                 "params": {"max_tokens": max_tokens,
                            "temperature": temperature, "stream": stream},
                 "reply": "".join(text), "finish": finish, "usage": usage,
                 "duration_s": round(time.time() - t_start, 2)}, rec)
        if resp:
            last = {"tool_calls": tool_calls} if tool_calls else {}
            await resp.write(b"data: " + json.dumps(
                chunk(cid, created, last, finish)).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        message = {"role": "assistant", "content": full or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return web.json_response(
            {"id": cid, "object": "chat.completion", "created": created,
             "model": backend.model_id,
             "choices": [{"index": 0, "finish_reason": finish,
                          "message": message}],
             "usage": usage}, headers=CORS)

    async def models(_request):
        return web.json_response(
            {"object": "list",
             "data": [{"id": backend.model_id, "object": "model",
                       "created": int(time.time()), "owned_by": "model-graph"}]},
            headers=CORS)

    async def options(_request):
        return web.Response(headers=CORS)

    async def curator_runs(_request):
        return web.json_response(cur._index() if cur else [], headers=CORS)

    async def curator_run(request):
        try:
            return web.json_response(cur.load(request.match_info["rid"]),
                                     headers=CORS)
        except (SystemExit, FileNotFoundError, AttributeError) as e:
            return web.json_response({"error": str(e) or "not found"},
                                     status=404, headers=CORS)

    async def curator_test(request):
        try:
            run = cur.load(request.match_info["rid"])
        except (SystemExit, FileNotFoundError, AttributeError) as e:
            return web.json_response({"error": str(e) or "not found"},
                                     status=404, headers=CORS)
        return web.json_response(
            {"id": run["id"],
             "results": curator_mod.run_checks(run, cur)}, headers=CORS)

    # dev: always fresh UI · deployed: cache to minimize egress
    PROD = bool(os.environ.get("RAILWAY_ENVIRONMENT")
                or os.environ.get("PROD"))
    NOCACHE = ({"Cache-Control": "public, max-age=600"} if PROD
               else {"Cache-Control": "no-cache"})

    async def ws_endpoint(request):        # same-port ws for single-port PaaS
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        send = ws.send_str
        await hub.add(send)
        try:
            async for _ in ws:             # clients only listen
                pass
        finally:
            hub.drop(send)
        return ws

    async def landing_page(_request):
        return web.FileResponse(os.path.join(WEB, "landing.html"),
                                headers=NOCACHE)

    async def chat_page(_request):
        return web.FileResponse(os.path.join(WEB, "chat.html"),
                                headers=NOCACHE)

    async def dash_page(_request):
        return web.FileResponse(os.path.join(WEB, "dashboard.html"),
                                headers=NOCACHE)

    async def sixpager_page(_request):
        return web.FileResponse(os.path.join(WEB, "six-pager.html"),
                                headers=NOCACHE)

    async def contrib_page(_request):
        return web.FileResponse(
            os.path.join(os.path.dirname(WEB), "CONTRIBUTING.md"),
            headers={**NOCACHE, "Content-Type": "text/plain; charset=utf-8"})

    @web.middleware
    async def gzip_mw(request, handler):   # egress-lean text responses
        resp = await handler(request)
        try:
            ct = resp.content_type or ""
        except Exception:
            ct = ""
        if (not resp.prepared
                and not request.path.startswith("/models/")   # big binaries
                and (ct.startswith("text/") or ct == "application/json"
                     or isinstance(resp, web.FileResponse))):
            try:
                resp.enable_compression()
            except Exception:
                pass
        return resp

    app = web.Application(client_max_size=0, middlewares=[gzip_mw])
    root = os.path.dirname(WEB)
    models_dir = os.path.join(root, "models")
    if os.path.isdir(models_dir):              # observable ONNX exports etc.
        app.router.add_static("/models/", models_dir)
    vault_dir = os.path.join(root, "vault")
    if os.path.isdir(vault_dir):               # docs, served as plain text
        app.router.add_static("/vault/", vault_dir, show_index=True)
    app.router.add_get("/", landing_page)
    app.router.add_get("/chat", chat_page)
    app.router.add_get("/dashboard", dash_page)
    app.router.add_get("/six-pager", sixpager_page)
    app.router.add_get("/ws", ws_endpoint)
    app.router.add_get("/CONTRIBUTING.md", contrib_page)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/curator/runs", curator_runs)
    app.router.add_get("/curator/runs/{rid}", curator_run)
    app.router.add_post("/curator/runs/{rid}/test", curator_test)
    app.router.add_route("OPTIONS", "/v1/{tail:.*}", options)
    return app


# ----------------------------------------------------------------------------

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--http-port", type=int,
                   default=int(os.environ.get("PORT", 8080)))
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--no-curate", action="store_true",
                   help="disable run recording under runs/")
    args = p.parse_args()

    import websockets
    from aiohttp import web

    print(f"loading backend ({'MOCK' if args.mock else args.model})...")
    backend = Backend(args)
    hub = Hub()
    hub.topology = backend.topo
    cur = None
    if not args.no_curate:
        cur = curator_mod.Curator()
        cur.on("run_saved", lambda run_id, meta, **_:
               print(f"[curator] saved {run_id} "
                     f"({meta['usage'].get('completion_tokens')} tok)"))

    async def ws_handler(ws):
        await hub.add(ws.send)
        try:
            async for _ in ws:      # dashboards only listen
                pass
        finally:
            hub.drop(ws.send)

    host = os.environ.get("HOST", "localhost")   # 0.0.0.0 in containers
    runner = web.AppRunner(make_app(backend, hub, cur))
    await runner.setup()
    await web.TCPSite(runner, host, args.http_port).start()
    async with websockets.serve(ws_handler, host, args.ws_port):
        print(f"model-graph api ({backend.model_id})\n"
              f"  chat UI:    http://localhost:{args.http_port}/\n"
              f"  dashboard:  http://localhost:{args.http_port}/dashboard\n"
              f"  OpenAI API: http://localhost:{args.http_port}/v1/chat/completions\n"
              f"  ws stream:  ws://localhost:{args.ws_port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
