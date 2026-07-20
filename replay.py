#!/usr/bin/env python3
"""
replay.py — stream a recorded/mock NDJSON trace over ws://localhost:8765
so graph.html can play it back. Works with make_mock.py output or any
trace captured from server.py.

Usage:
  python replay.py mock_dense_24L_120t.ndjson
  python replay.py mock_moe_24L_120t.ndjson --rate 8 --loop
"""

import argparse, asyncio, json


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--rate", type=float, default=4.0, help="tokens per second")
    p.add_argument("--loop", action="store_true")
    args = p.parse_args()

    msgs = [json.loads(l) for l in open(args.trace) if l.strip()]

    import websockets

    async def handler(ws):
        try:
            while True:
                for m in msgs:
                    await ws.send(json.dumps(m))
                    if m["type"] == "token":
                        await asyncio.sleep(1.0 / args.rate)
                if not args.loop:
                    break
        except websockets.ConnectionClosed:
            pass

    async with websockets.serve(handler, "localhost", args.port):
        print(f"replaying {args.trace} ({sum(1 for m in msgs if m['type']=='token')} "
              f"tokens) on ws://localhost:{args.port} — open graph.html")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
