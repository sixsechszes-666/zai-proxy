"""
test_e2e_shim.py -- the whole loop over real HTTP.

Unit tests prove the pieces. This proves the wiring: a real Anthropic request
into a running proxy, a real SSE stream out of it, with a mock upstream in
between that refuses to be told about tools. If a tool_use block comes back
and the tool result then reaches the model as text, Claude Code can run on a
backend that has no tool calling of its own.

  python test_e2e_shim.py
  python test_e2e_shim.py -v
"""

import json
import os
import sys
import threading
import time

# before importing proxy: it reads its config at module level
os.environ["ZAI_TOOLS"] = "shim"
os.environ["ZAI_BASE_URL"] = "http://127.0.0.1:8899"
os.environ["ZAI_API_KEY"] = "not-needed-for-the-mock"
os.environ["ZAI_DIRECT"] = "1"          # never let a local proxy touch loopback
os.environ["PROXY_PORT"] = "8789"
os.environ["ZAI_DUMP"] = "0"

import httpx          # noqa: E402
import uvicorn        # noqa: E402

import mock_shim      # noqa: E402
import proxy          # noqa: E402

PROXY = "http://127.0.0.1:8789"
MOCK = "http://127.0.0.1:8899"
VERBOSE = "-v" in sys.argv
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def serve(app, port: int) -> None:
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()


def wait_up(url: str, seconds: float = 20.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=2, trust_env=False) as c:
                if c.get(url).status_code == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def anthropic_stream(body: dict) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    with httpx.Client(timeout=60, trust_env=False) as c:
        with c.stream("POST", f"{PROXY}/v1/messages", json=body,
                      headers={"content-type": "application/json"}) as r:
            if r.status_code != 200:
                r.read()
                raise SystemExit(f"proxy returned {r.status_code}: {r.text[:400]}")
            name = None
            for line in r.iter_lines():
                if line.startswith("event: "):
                    name = line[7:].strip()
                elif line.startswith("data: ") and name:
                    events.append((name, json.loads(line[6:])))
                    name = None
    return events


def rebuild(events: list[tuple[str, dict]]) -> tuple[list[dict], str | None, dict]:
    blocks: dict[int, dict] = {}
    order: list[int] = []
    stop: str | None = None
    usage: dict = {}
    for name, d in events:
        if name == "content_block_start":
            blocks[d["index"]] = {**d["content_block"], "_json": ""}
            order.append(d["index"])
        elif name == "content_block_delta":
            dl = d["delta"]
            b = blocks[d["index"]]
            if dl["type"] == "text_delta":
                b["text"] = b.get("text", "") + dl["text"]
            elif dl["type"] == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + dl["thinking"]
            elif dl["type"] == "input_json_delta":
                b["_json"] += dl["partial_json"]
        elif name == "message_delta":
            stop = (d.get("delta") or {}).get("stop_reason")
            usage = d.get("usage") or {}
    out = []
    for i in order:
        b = blocks[i]
        raw = b.pop("_json", "")
        if b["type"] == "tool_use":
            b["input"] = json.loads(raw) if raw else {}
        out.append(b)
    return out, stop, usage


BASH_TOOL = {
    "name": "Bash",
    "description": "Run a shell command in the project directory.",
    "input_schema": {"type": "object",
                     "properties": {"command": {"type": "string", "description": "the command"}},
                     "required": ["command"]},
}

FIRST_TURN = {
    "model": "claude-sonnet-5",
    "max_tokens": 2048,
    "stream": True,
    "system": [{"type": "text", "text": "You are a coding agent working in C:/proj."}],
    "tools": [BASH_TOOL],
    "messages": [{"role": "user", "content": "what is in this folder?"}],
}


def test_loop() -> None:
    print("\nturn 1: claude code asks, backend has no tools")
    events = anthropic_stream(FIRST_TURN)
    if VERBOSE:
        for n, d in events:
            print("       ", n, json.dumps(d, ensure_ascii=False)[:130])

    blocks, stop, usage = rebuild(events)
    kinds = [b["type"] for b in blocks]
    check("stream carries a tool_use", "tool_use" in kinds, f"got {kinds}")
    calls = [b for b in blocks if b["type"] == "tool_use"]
    check("one call, named Bash", len(calls) == 1 and calls[0]["name"] == "Bash",
          f"got {[c.get('name') for c in calls]}")
    check("arguments survived the chop", calls and calls[0]["input"] == {"command": "ls -la"},
          f"got {calls[0]['input'] if calls else None}")
    check("call has an id", calls and str(calls[0].get("id", "")).startswith("toolu_"))
    check("stop_reason is tool_use", stop == "tool_use", f"got {stop}")
    check("prose kept", any(b["type"] == "text" and "Checking the folder." in b.get("text", "")
                            for b in blocks), f"got {blocks}")
    check("usage carried through", usage.get("input_tokens") == 40 and usage.get("output_tokens") == 12,
          f"got {usage}")

    with httpx.Client(timeout=10, trust_env=False) as c:
        seen = c.get(f"{MOCK}/checks").json()
    check("backend saw no tools array", seen.get("tools_absent") is True)
    check("backend saw no tool_choice", seen.get("tool_choice_absent") is True)
    check("schemas reached it as prompt text", seen.get("schemas_in_system") is True)
    check("envelope reached it as prompt text", seen.get("envelope_taught") is True)
    if VERBOSE:
        print("        system head:", seen.get("system_head"))

    call = calls[0]
    print("\nturn 2: the tool result goes back")
    second = {
        **FIRST_TURN,
        "messages": [
            {"role": "user", "content": "what is in this folder?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Checking the folder."},
                {"type": "tool_use", "id": call["id"], "name": "Bash", "input": call["input"]},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": call["id"],
                 "content": "a.txt\nb.txt\ntotal 2"},
            ]},
        ],
    }
    events = anthropic_stream(second)
    blocks, stop, _ = rebuild(events)
    kinds = [b["type"] for b in blocks]
    check("no further call", "tool_use" not in kinds, f"got {kinds}")
    check("stop_reason is end_turn", stop == "end_turn", f"got {stop}")
    text = " ".join(b.get("text", "") for b in blocks if b["type"] == "text")
    check("model answered from the result", "a.txt" in text and "b.txt" in text, f"got {text!r}")

    with httpx.Client(timeout=10, trust_env=False) as c:
        seen = c.get(f"{MOCK}/checks").json()
    check("result reached the backend", seen.get("result_seen") is True)
    check("result body reached the backend", seen.get("result_body_seen") is True)


def test_nonstream() -> None:
    print("\nnon-streaming request")
    body = {**FIRST_TURN, "stream": False}
    with httpx.Client(timeout=60, trust_env=False) as c:
        r = c.post(f"{PROXY}/v1/messages", json=body)
    check("status 200", r.status_code == 200, f"got {r.status_code} {r.text[:200]}")
    if r.status_code != 200:
        return
    msg = r.json()
    kinds = [b["type"] for b in msg["content"]]
    check("tool_use in a single json reply", "tool_use" in kinds, f"got {kinds}")
    tu = next(b for b in msg["content"] if b["type"] == "tool_use")
    check("call parses", tu["name"] == "Bash" and tu["input"] == {"command": "ls -la"},
          f"got {tu}")
    check("stop_reason is tool_use", msg["stop_reason"] == "tool_use", f"got {msg['stop_reason']}")


def test_native_still_native() -> None:
    """Flipping ZAI_TOOLS back must put a real tools array on the wire."""
    print("\nnative mode is a different code path")
    proxy.TOOLS_MODE = "native"
    try:
        events = anthropic_stream(FIRST_TURN)
        with httpx.Client(timeout=10, trust_env=False) as c:
            seen = c.get(f"{MOCK}/checks").json()
        check("backend now sees a tools array", seen.get("tools_absent") is False,
              f"got {seen.get('tools_absent')}")
        check("and no envelope in the prompt", seen.get("envelope_taught") is False)
        _, stop, _ = rebuild(events)
        check("stream still completes", stop in ("end_turn", "tool_use"), f"got {stop}")
    finally:
        proxy.TOOLS_MODE = "shim"


def main() -> None:
    print("starting mock backend and proxy")
    serve(mock_shim.app, 8899)
    serve(proxy.app, 8789)
    if not wait_up(f"{MOCK}/checks") or not wait_up(f"{PROXY}/health"):
        print("servers did not come up")
        sys.exit(1)
    with httpx.Client(timeout=5, trust_env=False) as c:
        health = c.get(f"{PROXY}/health").json()
    print(f"proxy health: tools_mode={health.get('tools_mode')} upstream={health.get('upstream')}")
    check("proxy reports shim mode", health.get("tools_mode") == "shim")

    test_loop()
    test_nonstream()
    test_native_still_native()

    print()
    if FAILED:
        print(f"{len(FAILED)} failed: {', '.join(FAILED)}")
        sys.exit(1)
    print("end to end: claude code's tool loop runs on a backend with no tool support")


if __name__ == "__main__":
    main()
