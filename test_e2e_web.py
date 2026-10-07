"""
test_e2e_web.py -- the whole loop, live: Claude Code -> proxy -> chat.z.ai.

This is the live counterpart to test_web.py. That one proves the translation
offline against canned frames; this one spends real requests to prove the
translation is right about the real endpoint, which is a different claim.

It runs the real proxy in web mode (no API key, browser session as the
credential) and drives one full tool round trip:

    turn 1  a tool schema and a question   ->  a tool_use block named Bash
    turn 2  the tool result handed back    ->  a text answer that used it

Both turns go through the shim, because the web endpoint cannot receive a
tools array at all. If this passes, Claude Code runs on the browser session.

Two things to know before running it:

  * it costs real requests, and the endpoint IP-bans after roughly ten in a
    burst. A fresh egress exit clears the ban instantly.
  * a signed-in chat.z.ai tab has to be open in Chrome on the CDP port.

  python test_e2e_web.py
  python test_e2e_web.py -v      # every SSE event
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

os.environ["ZAI_UPSTREAM"] = "web"     # must land before proxy is imported
os.environ["ZAI_TOOLS"] = "shim"
os.environ["PROXY_PORT"] = os.environ.get("PROXY_PORT", "8791")
os.environ.pop("ZAI_API_KEY", None)

import httpx            # noqa: E402
import uvicorn          # noqa: E402

import proxy as P       # noqa: E402

PORT = int(os.environ["PROXY_PORT"])
BASE = f"http://127.0.0.1:{PORT}"
VERBOSE = "-v" in sys.argv
FAILED: list[str] = []

BASH = {
    "name": "Bash",
    "description": "Run a shell command and return its output",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string",
                                   "description": "The command to run"}},
        "required": ["command"],
    },
}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def events(text: str) -> list[tuple[str, dict]]:
    """Anthropic SSE text into (event name, payload) pairs."""
    out: list[tuple[str, dict]] = []
    name = ""
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            try:
                payload = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            out.append((name, payload))
            if VERBOSE:
                print(f"      {name}: {json.dumps(payload)[:200]}")
    return out


def blocks(evs: list[tuple[str, dict]]) -> list[dict]:
    """Reassemble content blocks out of the delta stream."""
    out: dict[int, dict] = {}
    for name, p in evs:
        if name == "content_block_start":
            out[p["index"]] = dict(p["content_block"])
        elif name == "content_block_delta":
            d = p["delta"]
            if d.get("type") == "text_delta":
                out[p["index"]]["text"] = out[p["index"]].get("text", "") + d["text"]
            elif d.get("type") == "thinking_delta":
                out[p["index"]]["thinking"] = out[p["index"]].get("thinking", "") + d["thinking"]
            elif d.get("type") == "input_json_delta":
                out[p["index"]]["_json"] = out[p["index"]].get("_json", "") + d["partial_json"]
    for b in out.values():
        if "_json" in b:
            try:
                b["input"] = json.loads(b.pop("_json"))
            except json.JSONDecodeError:
                b["input"] = {}
    return [out[k] for k in sorted(out)]


def stop_reason(evs: list[tuple[str, dict]]) -> str | None:
    for name, p in evs:
        if name == "message_delta":
            return (p.get("delta") or {}).get("stop_reason")
    return None


def ask(client: httpx.Client, body: dict) -> tuple[list[dict], str | None, str]:
    r = client.post(f"{BASE}/v1/messages", json=body)
    if r.status_code != 200:
        return [], None, f"HTTP {r.status_code}: {r.text[:300]}"
    evs = events(r.text)
    err = next((p for n, p in evs if n == "error"), None)
    if err:
        return [], None, json.dumps(err)[:300]
    return blocks(evs), stop_reason(evs), ""


def main() -> None:
    server = uvicorn.Server(uvicorn.Config(P.app, host="127.0.0.1",
                                           port=PORT, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()

    # trust_env=False throughout: Clash listens on 127.0.0.1:7897 and will
    # happily swallow a request to loopback, answering 502 for a server that
    # is running fine.
    with httpx.Client(timeout=300, trust_env=False) as client:
        for _ in range(60):
            try:
                if client.get(f"{BASE}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            raise SystemExit("proxy did not come up")

        health = client.get(f"{BASE}/health", timeout=5).json()
        print(f"proxy: {health}")
        check("web mode is on", health.get("upstream_mode") == "web")
        check("no api key is needed", health.get("key_set") is False)
        print()

        print("turn 1: a tool schema in, a tool_use block out")
        blks, stop, problem = ask(client, {
            "model": "claude-sonnet-5",
            "max_tokens": 2048,
            "stream": True,
            "system": "You are a coding agent. Use the tools you are given.",
            "tools": [BASH],
            "messages": [{"role": "user", "content":
                          "Use the Bash tool to list the files in the current "
                          "directory. Do not guess the answer."}],
        })
        if problem:
            print(f"  FAIL  turn 1: {problem}")
            FAILED.append("turn 1")
            return

        kinds = [b["type"] for b in blks]
        calls = [b for b in blks if b["type"] == "tool_use"]
        check("the stream is well formed", kinds and kinds[0] in ("thinking", "text", "tool_use"),
              f"got {kinds}")
        check("a tool_use block came back", bool(calls), f"got {kinds}")
        if not calls:
            return
        call = calls[0]
        check("the tool is the one Claude Code offered", call["name"] == "Bash",
              str(call["name"]))
        check("the arguments parsed", isinstance(call["input"], dict))
        check("the arguments are the tool's shape",
              "command" in (call["input"] or {}), json.dumps(call.get("input"))[:120])
        check("stop_reason is tool_use", stop == "tool_use", str(stop))
        print(f"      call: {call['name']} {json.dumps(call.get('input'))[:160]}")

        print("\nturn 2: the result goes back, an answer comes out")
        blks2, stop2, problem2 = ask(client, {
            "model": "claude-sonnet-5",
            "max_tokens": 2048,
            "stream": True,
            "system": "You are a coding agent. Use the tools you are given.",
            "tools": [BASH],
            "messages": [
                {"role": "user", "content": "List the files in the current directory."},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": call.get("id") or "toolu_1",
                     "name": call["name"], "input": call.get("input") or {}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": call.get("id") or "toolu_1",
                     "content": "README.md\nproxy.py\nzebra.conf"}]},
            ],
        })
        if problem2:
            print(f"  FAIL  turn 2: {problem2}")
            FAILED.append("turn 2")
            return

        text = "\n".join(b.get("text", "") for b in blks2 if b["type"] == "text")
        calls2 = [b for b in blks2 if b["type"] == "tool_use"]
        check("turn 2 answers instead of calling again", not calls2,
              json.dumps(calls2)[:200])
        check("stop_reason is end_turn", stop2 == "end_turn", str(stop2))
        check("the answer exists", bool(text.strip()), f"got {text[:120]!r}")
        check("the answer used the tool result",
              "zebra.conf" in text or "README.md" in text, f"got {text[:200]!r}")
        print(f"      answer: {text.strip()[:300]!r}")

    print()
    if FAILED:
        print(f"{len(FAILED)} failed: {', '.join(FAILED)}")
        sys.exit(1)
    print("all good - Claude Code's tool loop runs on the browser session")


if __name__ == "__main__":
    main()
