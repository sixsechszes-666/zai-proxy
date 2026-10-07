"""
selftest for zai_proxy.

  python selftest.py --offline     # no network, no key: validates the translation layer
  python selftest.py               # hits a running proxy at 127.0.0.1:8788 end to end

Offline mode is the important one: it feeds canned OpenAI streams through the
converter and checks the Anthropic SSE that comes out, so you can prove tool
calling works before spending a single token of quota.
"""

from __future__ import annotations

import asyncio
import json
import sys

import proxy as P

PROXY = f"http://{P.LISTEN_HOST}:{P.LISTEN_PORT}"

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail and not cond else ""))


# --------------------------------------------------------------------------
# canned upstream streams
# --------------------------------------------------------------------------

def oai_chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
    body: dict = {"id": "chatcmpl-x", "object": "chat.completion.chunk", "model": "glm-4.6",
                  "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        body["usage"] = usage
    return f"data: {json.dumps(body)}\n\n"


TEXT_STREAM = [
    oai_chunk({"role": "assistant", "content": ""}),
    oai_chunk({"content": "Hello"}),
    oai_chunk({"content": " from GLM"}),
    oai_chunk({}, finish="stop", usage={"prompt_tokens": 120, "completion_tokens": 7}),
    "data: [DONE]\n\n",
]

TOOL_STREAM = [
    oai_chunk({"role": "assistant", "content": ""}),
    oai_chunk({"content": "Let me check."}),
    oai_chunk({"tool_calls": [{"index": 0, "id": "call_abc", "type": "function",
                               "function": {"name": "Bash", "arguments": ""}}]}),
    oai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"comm'}}]}),
    oai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'and":"ls -la"}'}}]}),
    oai_chunk({}, finish="tool_calls", usage={"prompt_tokens": 900, "completion_tokens": 40}),
    "data: [DONE]\n\n",
]

PARALLEL_TOOL_STREAM = [
    oai_chunk({"role": "assistant", "content": ""}),
    oai_chunk({"tool_calls": [{"index": 0, "id": "c0", "function": {"name": "Read", "arguments": '{"path":"a.txt"'}}]}),
    oai_chunk({"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}),
    oai_chunk({"tool_calls": [{"index": 1, "id": "c1", "function": {"name": "Read", "arguments": '{"path":'}}]}),
    oai_chunk({"tool_calls": [{"index": 1, "function": {"arguments": '"b.txt"}'}}]}),
    oai_chunk({}, finish="tool_calls"),
    "data: [DONE]\n\n",
]

THINKING_STREAM = [
    oai_chunk({"role": "assistant", "content": ""}),
    oai_chunk({"reasoning_content": "The user wants"}),
    oai_chunk({"reasoning_content": " a listing."}),
    oai_chunk({"content": "Here it is."}),
    oai_chunk({}, finish="stop"),
    "data: [DONE]\n\n",
]


class FakeResp:
    def __init__(self, lines: list[str]):
        self._lines = lines

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln.rstrip("\n")


async def collect(lines: list[str]) -> list[tuple[str, dict]]:
    out = []
    async for raw in P.stream_anthropic(FakeResp(lines), "glm-4.6"):
        text = raw.decode("utf-8")
        for block in text.strip().split("\n\n"):
            if not block.strip():
                continue
            ev = data = None
            for ln in block.splitlines():
                if ln.startswith("event: "):
                    ev = ln[7:]
                elif ln.startswith("data: "):
                    data = json.loads(ln[6:])
            if ev:
                out.append((ev, data))
    return out


def reassemble(events: list[tuple[str, dict]]) -> list[dict]:
    """Rebuild content blocks the way Claude Code's parser does."""
    blocks: dict[int, dict] = {}
    order: list[int] = []
    for ev, d in events:
        if ev == "content_block_start":
            i = d["index"]
            blocks[i] = dict(d["content_block"])
            blocks[i].setdefault("_text", "")
            blocks[i].setdefault("_json", "")
            order.append(i)
        elif ev == "content_block_delta":
            i, dl = d["index"], d["delta"]
            if dl["type"] == "text_delta":
                blocks[i]["_text"] += dl["text"]
            elif dl["type"] == "input_json_delta":
                blocks[i]["_json"] += dl["partial_json"]
            elif dl["type"] == "thinking_delta":
                blocks[i]["thinking"] += dl["thinking"]
    return [blocks[i] for i in order]


# --------------------------------------------------------------------------
# offline suite
# --------------------------------------------------------------------------

async def offline() -> None:
    print("\n== request translation ==")

    req = {
        "model": "claude-sonnet-5",
        "max_tokens": 8192,
        "system": [{"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "Bash", "description": "run a command",
                   "input_schema": {"type": "object", "properties": {"command": {"type": "string"}},
                                    "required": ["command"], "cache_control": {"type": "ephemeral"}}}],
        "tool_choice": {"type": "auto"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "list files", "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "sure"},
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "a.txt\nb.txt"}]},
                {"type": "text", "text": "now summarize"}]},
        ],
    }
    oai = P.anthropic_to_openai(req)

    check("model alias claude-sonnet-5 -> GLM", oai["model"] == P.DEFAULT_MODEL, oai["model"])
    check("system hoisted to role=system", oai["messages"][0]["role"] == "system")
    check("cache_control stripped from system", "cache_control" not in json.dumps(oai["messages"][0]))
    check("max_tokens preserved", oai["max_tokens"] == 8192)
    check("tools converted to function schema",
          oai["tools"][0]["function"]["parameters"]["required"] == ["command"])
    check("cache_control stripped from tool schema", "cache_control" not in json.dumps(oai["tools"]))
    check("tool_choice auto", oai["tool_choice"] == "auto")
    check("assistant tool_use -> tool_calls",
          oai["messages"][2].get("tool_calls", [{}])[0]["function"]["name"] == "Bash")
    check("tool_result -> role=tool",
          any(m["role"] == "tool" and m["tool_call_id"] == "toolu_1" for m in oai["messages"]))
    check("trailing user text kept after tool msg",
          oai["messages"][-1]["role"] == "user" and "summarize" in oai["messages"][-1]["content"])

    # teacher forcing: tool_choice variants
    for kind, want in [("any", "required"), ("none", "none")]:
        r = dict(req); r["tool_choice"] = {"type": kind}
        check(f"tool_choice {kind} -> {want}", P.anthropic_to_openai(r)["tool_choice"] == want)
    r = dict(req); r["tool_choice"] = {"type": "tool", "name": "Bash"}
    check("tool_choice tool -> pinned function",
          P.anthropic_to_openai(r)["tool_choice"]["function"]["name"] == "Bash")

    # orphan tool_result gets dropped, missing result gets stubbed
    orphan = {"messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "nope", "content": "x"}]}]}
    check("orphan tool_result dropped", P.anthropic_to_openai(orphan)["messages"] == [])

    missing = {"messages": [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
    msgs = P.anthropic_to_openai(missing)["messages"]
    check("missing tool_result gets stub", msgs[1]["role"] == "tool" and msgs[1]["tool_call_id"] == "t1")

    # images
    img = {"messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]}
    part = P.anthropic_to_openai(img)["messages"][0]["content"][0]
    check("base64 image -> image_url data uri", part["image_url"]["url"].startswith("data:image/png;base64,"))

    # thinking round trip must not go back upstream
    think = {"messages": [
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": "x"},
                                          {"type": "text", "text": "answer"}]}]}
    check("thinking blocks dropped on send",
          "hmm" not in json.dumps(P.anthropic_to_openai(think)["messages"]))

    # thinking policy -- GLM-5.3-Flash reasons unconditionally, so the default
    # must be to send nothing rather than to ask for something it ignores.
    think_req = {"messages": [], "thinking": {"type": "enabled", "budget_tokens": 4096}}
    saved = P.THINKING
    P.THINKING = "none"
    check("THINKING=none sends no thinking param",
          "thinking" not in P.anthropic_to_openai(think_req))
    P.THINKING = "auto"
    check("THINKING=auto forwards an enabled request",
          P.anthropic_to_openai(think_req).get("thinking") == {"type": "enabled"})
    check("THINKING=auto sends nothing when Claude Code did not ask",
          "thinking" not in P.anthropic_to_openai({"messages": []}))
    P.THINKING = "off"
    check("THINKING=off sends disabled",
          P.anthropic_to_openai(think_req).get("thinking") == {"type": "disabled"})
    P.THINKING = saved

    saved_effort = P.REASONING_EFFORT
    P.REASONING_EFFORT = "low"
    check("reasoning_effort forwarded",
          P.anthropic_to_openai({"messages": []}).get("reasoning_effort") == "low")
    P.REASONING_EFFORT = saved_effort
    check("reasoning_effort omitted when unset",
          "reasoning_effort" not in P.anthropic_to_openai({"messages": []}))

    # model mapping
    check("glm-5.3-flash passthrough", P.resolve_model("glm-5.3-flash") == "glm-5.3-flash")
    check("default model is glm-5.3-flash", P.DEFAULT_MODEL == "glm-5.3-flash", P.DEFAULT_MODEL)
    check("haiku -> small model", P.resolve_model("claude-haiku-4-5") == P.SMALL_MODEL)
    check("glm passthrough", P.resolve_model("glm-4.6") == "glm-4.6")
    check("unknown -> default", P.resolve_model("gpt-4o") == P.DEFAULT_MODEL)

    # json repair
    check("repair truncated json", P.repair_json('{"command":"ls"') == {"command": "ls"})
    check("repair fenced json", P.repair_json('```json\n{"a":1}\n```') == {"a": 1})
    check("repair empty -> {}", P.repair_json("") == {})

    print("\n== streaming translation ==")

    ev = await collect(TEXT_STREAM)
    names = [e for e, _ in ev]
    check("message_start first", names[0] == "message_start")
    check("message_stop last", names[-1] == "message_stop")
    blocks = reassemble(ev)
    check("single text block", len(blocks) == 1 and blocks[0]["type"] == "text")
    check("text assembled", blocks[0]["_text"] == "Hello from GLM", blocks[0]["_text"])
    md = [d for e, d in ev if e == "message_delta"][0]
    check("stop_reason end_turn", md["delta"]["stop_reason"] == "end_turn")
    check("usage carried", md["usage"]["input_tokens"] == 120 and md["usage"]["output_tokens"] == 7)

    ev = await collect(TOOL_STREAM)
    blocks = reassemble(ev)
    check("text + tool_use blocks", [b["type"] for b in blocks] == ["text", "tool_use"],
          str([b["type"] for b in blocks]))
    tu = blocks[1]
    check("tool id preserved", tu["id"] == "call_abc", tu["id"])
    check("tool name preserved", tu["name"] == "Bash")
    check("tool args reassembled as valid json", json.loads(tu["_json"]) == {"command": "ls -la"}, tu["_json"])
    md = [d for e, d in ev if e == "message_delta"][0]
    check("stop_reason tool_use", md["delta"]["stop_reason"] == "tool_use")

    ev = await collect(PARALLEL_TOOL_STREAM)
    blocks = reassemble(ev)
    tools = [b for b in blocks if b["type"] == "tool_use"]
    check("two parallel tool blocks", len(tools) == 2, str(len(tools)))
    check("indices are distinct and monotonic",
          [d["index"] for e, d in ev if e == "content_block_start"] == sorted(
              d["index"] for e, d in ev if e == "content_block_start"))
    check("parallel args correct",
          json.loads(tools[0]["_json"]) == {"path": "a.txt"} and json.loads(tools[1]["_json"]) == {"path": "b.txt"})

    ev = await collect(THINKING_STREAM)
    blocks = reassemble(ev)
    check("thinking block emitted", blocks and blocks[0]["type"] == "thinking")
    check("thinking text assembled", blocks[0]["thinking"] == "The user wants a listing.", blocks[0]["thinking"])
    check("text block after thinking", blocks[1]["type"] == "text" and blocks[1]["_text"] == "Here it is.")

    print("\n== non-streaming translation ==")
    full = {"id": "chatcmpl-1", "choices": [{"message": {
        "role": "assistant", "content": "done",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "Read", "arguments": '{"path":"x"}'}}]},
        "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}
    a = P.openai_to_anthropic(full, "glm-4.6")
    check("non-stream: tool_use block", a["content"][1]["type"] == "tool_use" and a["content"][1]["input"] == {"path": "x"})
    check("non-stream: stop_reason tool_use", a["stop_reason"] == "tool_use")
    check("non-stream: usage mapped", a["usage"]["input_tokens"] == 10 and a["usage"]["output_tokens"] == 3)


# --------------------------------------------------------------------------
# live suite
# --------------------------------------------------------------------------

def live() -> None:
    import httpx

    # localhost must never go through the system proxy (Clash et al)
    C = httpx.Client(trust_env=False, timeout=120)

    print(f"\n== live proxy at {PROXY} ==")
    try:
        h = C.get(f"{PROXY}/health").json()
    except Exception as exc:
        print(f"  proxy not reachable: {exc}")
        sys.exit(2)
    print(f"  health: {h}")
    check("upstream key configured", h.get("key_set") is True)

    H = {"x-api-key": P.PROXY_TOKEN or "local", "anthropic-version": "2023-06-01", "content-type": "application/json"}

    # 1. plain streaming
    body = {"model": "claude-sonnet-5", "max_tokens": 256, "stream": True,
            "messages": [{"role": "user", "content": "Reply with exactly: PONG"}]}
    text = ""
    with C.stream("POST", f"{PROXY}/v1/messages", json=body, headers=H) as r:
        check("200 on /v1/messages", r.status_code == 200, str(r.status_code))
        for line in r.iter_lines():
            if line.startswith("data: "):
                d = json.loads(line[6:])
                if d.get("type") == "content_block_delta" and d["delta"].get("type") == "text_delta":
                    text += d["delta"]["text"]
    check("streamed text non-empty", bool(text.strip()), repr(text[:80]))

    # 2. tool calling
    body = {
        "model": "claude-sonnet-5", "max_tokens": 512, "stream": True,
        "tools": [{"name": "get_weather", "description": "Get weather for a city",
                   "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                                    "required": ["city"]}}],
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": "What is the weather in Tokyo? Use the tool."}],
    }
    got = None
    with C.stream("POST", f"{PROXY}/v1/messages", json=body, headers=H) as r:
        cur = None
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            d = json.loads(line[6:])
            if d.get("type") == "content_block_start" and d["content_block"]["type"] == "tool_use":
                cur = {"name": d["content_block"]["name"], "json": ""}
            elif d.get("type") == "content_block_delta" and d["delta"].get("type") == "input_json_delta" and cur:
                cur["json"] += d["delta"]["partial_json"]
            elif d.get("type") == "content_block_stop" and cur:
                cur["input"] = P.repair_json(cur.pop("json"))
                got = cur
    check("model called the tool", got is not None and got["name"] == "get_weather", str(got))
    if got:
        check("tool arguments parsed", isinstance(got["input"], dict) and "city" in got["input"], str(got["input"]))

    # 3. multi-turn round trip with the tool result
    if got:
        body2 = dict(body)
        body2["tool_choice"] = {"type": "auto"}
        body2["stream"] = False
        body2["messages"] = body["messages"] + [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_live",
                                               "name": got["name"], "input": got["input"]}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_live",
                                          "content": "18C and raining"}]},
        ]
        r = C.post(f"{PROXY}/v1/messages", json=body2, headers=H)
        check("round trip accepted", r.status_code == 200, f"{r.status_code} {r.text[:200]}")
        if r.status_code == 200:
            j = r.json()
            txt = "".join(b.get("text", "") for b in j["content"] if b["type"] == "text")
            check("model used the tool result", "18" in txt or "rain" in txt.lower(), txt[:120])

    # 4. count_tokens
    r = C.post(f"{PROXY}/v1/messages/count_tokens",
               json={"model": "glm-4.6", "messages": [{"role": "user", "content": "hello there"}]},
               headers=H)
    check("count_tokens responds", r.status_code == 200 and r.json().get("input_tokens", 0) > 0, r.text[:120])


if __name__ == "__main__":
    if "--offline" in sys.argv:
        asyncio.run(offline())
    else:
        live()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1 if FAIL else 0)
