"""
test_shim.py -- proves the text-protocol tool shim without touching the network.

Two things have to hold for shim mode to be usable by Claude Code:

  1. the request going upstream carries no `tools` array and still describes
     every tool, with the tool_use / tool_result history intact as text;
  2. the reply coming back is re-framed into a well formed Anthropic SSE
     stream whose tool_use blocks Claude Code's parser will accept.

Both are checked here against canned streams, including the ugly cases: a
marker split across two deltas, several calls in one reply, a truncated
envelope, and JSON the model got slightly wrong.

  python test_shim.py
  python test_shim.py -v      # dump every event
"""

from __future__ import annotations

import asyncio
import json
import sys

import proxy as P
import toolshim as T

VERBOSE = "-v" in sys.argv
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


# --------------------------------------------------------------------------
# canned upstream
# --------------------------------------------------------------------------

class FakeResp:
    """Just enough of httpx.Response for stream_anthropic."""

    def __init__(self, text: str, piece: int = 1):
        self.lines = [f"data: {json.dumps(c)}"
                      for c in chunks_of(text, piece)] + ["data: [DONE]"]

    async def aiter_lines(self):
        for line in self.lines:
            yield line


def chunks_of(text: str, piece: int) -> list[dict]:
    """An OpenAI style delta stream carrying `text`, split every `piece` chars."""
    out = []
    for i in range(0, len(text), piece):
        out.append({"choices": [{"index": 0, "delta": {"content": text[i:i + piece]},
                                 "finish_reason": None}]})
    out.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 22}})
    return out


def replay(text: str, shim: bool = True, piece: int = 1) -> list[tuple[str, dict]]:
    """Run a canned reply through the streamer, return the Anthropic events."""
    events: list[tuple[str, dict]] = []

    async def go():
        async for raw in P.stream_anthropic(FakeResp(text, piece), "glm-5.3-flash", shim=shim):
            s = raw.decode("utf-8")
            name = s.split("\n", 1)[0][len("event: "):]
            data = json.loads(s.split("data: ", 1)[1].strip())
            events.append((name, data))

    asyncio.run(go())
    if VERBOSE:
        for name, data in events:
            print("      ", name, json.dumps(data, ensure_ascii=False)[:150])
    return events


def rebuild(events: list[tuple[str, dict]]) -> tuple[list[dict], str | None]:
    """Fold the SSE events into content blocks, the way a client would."""
    blocks: dict[int, dict] = {}
    order: list[int] = []
    stop: str | None = None
    for name, d in events:
        if name == "content_block_start":
            i = d["index"]
            cb = dict(d["content_block"])
            cb["_json"] = ""
            blocks[i] = cb
            order.append(i)
        elif name == "content_block_delta":
            i = d["index"]
            dl = d["delta"]
            if dl["type"] == "text_delta":
                blocks[i]["text"] = blocks[i].get("text", "") + dl["text"]
            elif dl["type"] == "thinking_delta":
                blocks[i]["thinking"] = blocks[i].get("thinking", "") + dl["thinking"]
            elif dl["type"] == "input_json_delta":
                blocks[i]["_json"] += dl["partial_json"]
        elif name == "message_delta":
            stop = (d.get("delta") or {}).get("stop_reason")
    out = []
    for i in order:
        b = blocks[i]
        raw = b.pop("_json", "")
        if b["type"] == "tool_use":
            b["input"] = json.loads(raw) if raw else {}
        out.append(b)
    return out, stop


def well_formed(events: list[tuple[str, dict]]) -> str:
    """Every content_block_start must close, indices must not repeat."""
    open_at: int | None = None
    seen: list[int] = []
    for name, d in events:
        if name == "content_block_start":
            if open_at is not None:
                return f"block {open_at} never closed"
            open_at = d["index"]
            seen.append(open_at)
        elif name == "content_block_stop":
            if open_at != d["index"]:
                return f"stop {d['index']} while {open_at} open"
            open_at = None
    if open_at is not None:
        return f"block {open_at} left open at end"
    if len(seen) != len(set(seen)):
        return "an index was reused"
    names = [n for n, _ in events]
    if names[0] != "message_start" or names[-1] != "message_stop":
        return f"bad envelope: {names[0]} .. {names[-1]}"
    return ""


# --------------------------------------------------------------------------
# the request direction
# --------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string"}},
    "required": ["command"],
}

ANTHROPIC_REQ = {
    "model": "claude-sonnet-5",
    "max_tokens": 4096,
    "system": "You are a coding agent.",
    "tools": [
        {"name": "Bash", "description": "Run a shell command.",
         "input_schema": SCHEMA},
        {"name": "Read", "description": "Read a file.",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
    ],
    "messages": [
        {"role": "user", "content": "list the folder"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "On it."},
            {"type": "tool_use", "id": "toolu_abc", "name": "Bash",
             "input": {"command": "ls"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_abc",
             "content": "a.txt\nb.txt"},
        ]},
    ],
}


def test_request_direction() -> None:
    print("\nrequest direction")
    req = P.shimify(json.loads(json.dumps(ANTHROPIC_REQ)))
    oai = P.anthropic_to_openai(req)

    check("no tools array upstream", "tools" not in oai,
          f"got {list(oai.get('tools', []))[:1]}")
    check("no tool_choice upstream", "tool_choice" not in oai)

    sysmsg = oai["messages"][0]
    check("system carries the schemas", sysmsg["role"] == "system"
          and "### Bash" in sysmsg["content"] and "### Read" in sysmsg["content"])
    check("system teaches the envelope", T.OPEN in sysmsg["content"])
    check("original system text kept", "coding agent" in sysmsg["content"])

    # read the block text, not the serialised json - json.dumps escapes the
    # quotes inside the envelope and the substring would never match
    texts: list[str] = []
    for m in oai["messages"]:
        c = m.get("content")
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b.get("text", ""))
    joined = "\n".join(texts)

    check("tool_use became an envelope",
          T.render_call("Bash", {"command": "ls"}) in joined)
    check("tool_result carries its tool name",
          '<tool_result tool="Bash"' in joined and "a.txt" in joined)
    check("no orphan tool role left", not any(m.get("role") == "tool" for m in oai["messages"]))

    # and the native path must still do the opposite
    native = P.anthropic_to_openai(json.loads(json.dumps(ANTHROPIC_REQ)))
    check("native still sends tools", bool(native.get("tools"))
          and native["tools"][0]["function"]["name"] == "Bash")

    # forcing a tool should be stated in words, since the wire has no field for it
    forced = json.loads(json.dumps(ANTHROPIC_REQ))
    forced["tool_choice"] = {"type": "tool", "name": "Read"}
    fsys = P.anthropic_to_openai(P.shimify(forced))["messages"][0]["content"]
    check("forced tool_choice is spelled out", "MUST call a tool" in fsys
          and "`Read`" in fsys)


# --------------------------------------------------------------------------
# the parser, on its own
# --------------------------------------------------------------------------

def test_parser() -> None:
    print("\nparser")
    call = T.render_call("Bash", {"command": "ls -la"})

    ev = T.parse_full(f"before {call} after")
    check("text around a call",
          [e[0] for e in ev] == ["text", "call", "text"]
          and ev[1][1] == {"name": "Bash", "arguments": {"command": "ls -la"}}
          and ev[0][1] == "before " and ev[2][1] == " after")

    two = call + T.render_call("Read", {"path": "x"})
    ev = T.parse_full(two)
    check("parallel calls", [e[0] for e in ev] == ["call", "call"]
          and ev[1][1]["name"] == "Read")

    # every possible split point must give the same answer
    worst = ""
    for i in range(len(call) + 1):
        p = T.ToolStreamParser()
        ev = p.feed(call[:i]) + p.feed(call[i:]) + p.finish()
        kinds = [e[0] for e in ev]
        if kinds != ["call"]:
            worst = f"split at {i} -> {kinds} {call[:i]!r}|{call[i:]!r}"
            break
    check("survives a split at any byte", not worst, worst)

    ev = parse_streamed(call, piece=3)
    check("survives three-char deltas", [e[0] for e in ev] == ["call"])

    ev = T.parse_full(f"thinking... {T.OPEN}{{\"name\":\"Bash\",\"arguments\":{{\"command\":\"ls\"")
    check("truncated envelope is salvaged",
          [e[0] for e in ev] == ["text", "call"] and ev[1][1]["arguments"] == {"command": "ls"})

    ev = T.parse_full("just prose, no tools here")
    check("plain text passes through",
          [e[0] for e in ev] == ["text"] and ev[0][1] == "just prose, no tools here")

    ev = T.parse_full(f"{call}")
    check("empty arguments become {}", T.parse_full(
        T.render_call("Bash", {}))[0][1]["arguments"] == {})

    # a half typed marker at the tail must not leak as text, and must still
    # resolve once the rest of the envelope arrives
    p = T.ToolStreamParser()
    ev = p.feed("done. <tool")
    check("partial marker held back", len(ev) == 1 and ev[0] == ("text", "done. "),
          f"got {ev}")
    body = json.dumps({"name": "Bash", "arguments": {"command": "ls"}})
    ev = p.feed("_call>" + body + T.CLOSE) + p.finish()
    check("held marker completes", [e[0] for e in ev] == ["call"]
          and ev[0][1]["name"] == "Bash", f"got {ev}")

    # a doubled marker must not swallow an otherwise good call
    ev = T.parse_full(T.OPEN + T.render_call("Bash", {"command": "ls"}))
    check("doubled marker still parses", [e[0] for e in ev] == ["call"], f"got {ev}")

    # prose smuggled inside the envelope must not lose the call either
    ev = T.parse_full(T.OPEN + "sure, calling now " + body + T.CLOSE)
    check("prose inside the envelope tolerated",
          [e[0] for e in ev] == ["call"], f"got {ev}")


def parse_streamed(text: str, piece: int) -> list:
    p = T.ToolStreamParser()
    out = []
    for i in range(0, len(text), piece):
        out += p.feed(text[i:i + piece])
    return out + p.finish()


# --------------------------------------------------------------------------
# the unwrapped call
#
# Measured on the live web endpoint: asked to use Bash, GLM wrote the sentence
# quoted in the module docstring and then the bare object, with no marker. The
# strict parser read that as an answer, so the tool loop ended in silence.
# --------------------------------------------------------------------------

BARE = '{"name": "Bash", "arguments": {"command": "ls -la"}}'
NAMES = {"Bash", "Read"}


def test_bare_calls() -> None:
    print("\nunwrapped calls")

    ev = T.parse_full(f"I'll list the files.\n\n{BARE}", NAMES)
    check("prose then a bare object is a call",
          [e[0] for e in ev] == ["text", "call"], f"got {ev}")
    check("the bare call carries its arguments",
          ev and ev[-1][1] == {"name": "Bash", "arguments": {"command": "ls -la"}},
          f"got {ev}")

    ev = T.parse_full(BARE, NAMES)
    check("a bare object alone is a call", [e[0] for e in ev] == ["call"], f"got {ev}")

    # no allowlist, no call: nothing else separates intent from a JSON object
    ev = T.parse_full(BARE)
    check("without the tool list it stays text", [e[0] for e in ev] == ["text"], f"got {ev}")

    # the allowlist is what makes this safe
    ev = T.parse_full('{"name": "config", "arguments": {"port": 8080}}', NAMES)
    check("an object naming no real tool stays text",
          [e[0] for e in ev] == ["text"], f"got {ev}")

    # a real object may still sit in a code block ahead of the call
    ev = T.parse_full('Sample config:\n\n```json\n{"name": "host", "arguments": {}}\n```\n\n'
                      + BARE, NAMES)
    check("a JSON code block does not eat the call",
          [e[0] for e in ev] == ["text", "text", "call"], f"got {ev}")

    # the marker still wins when both are present
    ev = T.parse_full(T.render_call("Read", {"path": "a"}), NAMES)
    check("a wrapped call still parses as one call", [e[0] for e in ev] == ["call"], f"got {ev}")

    # every split point, as with the marker form
    worst = ""
    for i in range(len(BARE) + 1):
        p = T.ToolStreamParser(NAMES)
        ev = p.feed(BARE[:i]) + p.feed(BARE[i:]) + p.finish()
        if [e[0] for e in ev] != ["call"]:
            worst = f"split at {i} -> {[e[0] for e in ev]} {BARE[:i]!r}|{BARE[i:]!r}"
            break
    check("survives a split at any byte", not worst, worst)

    # three characters at a time: the object is held until it closes
    p = T.ToolStreamParser(NAMES)
    ev = []
    for i in range(0, len(BARE), 3):
        ev += p.feed(BARE[i:i + 3])
    ev += p.finish()
    check("survives three-char deltas", [e[0] for e in ev] == ["call"], f"got {ev}")

    # held, not leaked: nothing the model has not finished may go out early
    p = T.ToolStreamParser(NAMES)
    ev = p.feed('sure.\n\n{"name": "Bash", "argu')
    check("an open object is held back",
          [e[0] for e in ev] == ["text"] and ev[0][1] == "sure.\n\n", f"got {ev}")

    # a stream cut mid-object is still worth a try
    ev = T.parse_full('on it\n\n{"name": "Bash", "arguments": {"command": "ls"', NAMES)
    check("a truncated bare call is salvaged",
          [e[0] for e in ev] == ["text", "call"] and ev[-1][1]["arguments"] == {"command": "ls"},
          f"got {ev}")

    # prose braces are not objects and must never be held
    ev = T.parse_full("replace {PLACEHOLDER} in the file", NAMES)
    check("prose braces are not held",
          [e[0] for e in ev] == ["text"] and ev[0][1] == "replace {PLACEHOLDER} in the file",
          f"got {ev}")

    # a brace inside an argument value is not the end of the object
    ev = T.parse_full('{"name": "Bash", "arguments": {"command": "echo \\"}\\" "}}', NAMES)
    check("a brace inside a string does not close it early",
          [e[0] for e in ev] == ["call"] and ev[0][1]["arguments"]["command"] == 'echo "}" ',
          f"got {ev}")


# --------------------------------------------------------------------------
# the response direction, end to end through the streamer
# --------------------------------------------------------------------------

def test_stream_simple_call() -> None:
    print("\none call, no prose")
    call = T.render_call("Bash", {"command": "ls -la"})
    ev = replay(call)
    bad = well_formed(ev)
    check("sse sequence well formed", not bad, bad)
    blocks, stop = rebuild(ev)
    check("one tool_use block", len(blocks) == 1 and blocks[0]["type"] == "tool_use")
    check("name and args survive", blocks[0]["name"] == "Bash"
          and blocks[0]["input"] == {"command": "ls -la"})
    check("has a tool_use id", str(blocks[0].get("id", "")).startswith("toolu_"))
    check("stop_reason is tool_use", stop == "tool_use", f"got {stop}")


def test_stream_with_prose() -> None:
    print("\nprose plus two calls")
    text = ("Let me check.\n"
            + T.render_call("Bash", {"command": "ls"}) + "\n"
            + T.render_call("Read", {"path": "a.txt"}) + "\n"
            + "Running both now.")
    ev = replay(text, piece=7)
    bad = well_formed(ev)
    check("sse sequence well formed", not bad, bad)
    blocks, stop = rebuild(ev)
    kinds = [b["type"] for b in blocks]
    check("text and both calls present", kinds == ["text", "tool_use", "tool_use", "text"],
          f"got {kinds}")
    check("both names right", [b["name"] for b in blocks if b["type"] == "tool_use"] == ["Bash", "Read"])
    check("prose preserved", blocks[0]["text"] == "Let me check.\n"
          and blocks[-1]["text"] == "\nRunning both now.")
    check("stop_reason is tool_use", stop == "tool_use")


def test_stream_text_only() -> None:
    print("\nno calls at all")
    ev = replay("Nothing to run, the folder is empty.", piece=5)
    bad = well_formed(ev)
    check("sse sequence well formed", not bad, bad)
    blocks, stop = rebuild(ev)
    check("single text block", len(blocks) == 1 and blocks[0]["type"] == "text")
    check("text intact", blocks[0]["text"] == "Nothing to run, the folder is empty.")
    check("stop_reason is end_turn", stop == "end_turn", f"got {stop}")


def test_stream_malformed_json() -> None:
    print("\nmodel mangles the json")
    text = T.OPEN + '{"name": "Bash", "arguments": {"command": "ls -la"' + T.CLOSE
    ev = replay(text)
    bad = well_formed(ev)
    check("sse sequence well formed", not bad, bad)
    blocks, _ = rebuild(ev)
    check("args repaired", blocks and blocks[0]["type"] == "tool_use"
          and blocks[0]["input"] == {"command": "ls -la"}, f"got {blocks}")


def test_stream_thinking() -> None:
    print("\nreasoning then a call")
    resp = FakeResp("")
    resp.lines = [
        f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': 'weighing options'}}]})}",
        f"data: {json.dumps({'choices': [{'delta': {'content': T.render_call('Bash', {'command': 'pwd'})}}]})}",
        f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}",
        "data: [DONE]",
    ]

    ev: list[tuple[str, dict]] = []

    async def go():
        async for raw in P.stream_anthropic(resp, "glm-5.3-flash", shim=True):
            s = raw.decode("utf-8")
            ev.append((s.split("\n", 1)[0][len("event: "):],
                       json.loads(s.split("data: ", 1)[1].strip())))

    asyncio.run(go())
    bad = well_formed(ev)
    check("sse sequence well formed", not bad, bad)
    blocks, stop = rebuild(ev)
    check("thinking block kept", blocks[0]["type"] == "thinking"
          and blocks[0]["thinking"] == "weighing options")
    check("call follows it", blocks[1]["type"] == "tool_use"
          and blocks[1]["name"] == "Bash")
    check("stop_reason is tool_use", stop == "tool_use")


def test_native_unchanged() -> None:
    print("\nnative path untouched")
    # arguments deliberately split mid-token, and carrying a quote
    part1 = '{"comm'
    part2 = 'and":"ls"}'
    resp = FakeResp("")
    resp.lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": "hi "}}]}),
        "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "Bash", "arguments": part1}}]}}]}),
        "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": part2}}]}}]}),
        "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        "data: [DONE]",
    ]

    ev: list[tuple[str, dict]] = []

    async def go():
        async for raw in P.stream_anthropic(resp, "glm-5.3-flash", shim=False):
            s = raw.decode("utf-8")
            ev.append((s.split("\n", 1)[0][len("event: "):],
                       json.loads(s.split("data: ", 1)[1].strip())))

    asyncio.run(go())
    bad = well_formed(ev)
    check("sse sequence well formed", not bad, bad)
    blocks, stop = rebuild(ev)
    check("native tool_calls still work", len(blocks) == 2
          and blocks[1]["type"] == "tool_use"
          and blocks[1]["input"] == {"command": "ls"}
          and blocks[1]["id"] == "call_1", f"got {blocks}")
    check("stop_reason is tool_use", stop == "tool_use")


def test_nonstream() -> None:
    print("\nnon-streaming path")
    data = {"choices": [{"message": {
        "content": "Here.\n" + T.render_call("Read", {"path": "a.txt"}),
        "reasoning_content": "hmm"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6}}
    out = P.openai_to_anthropic_shim(data, "glm-5.3-flash")
    kinds = [b["type"] for b in out["content"]]
    check("thinking, text, tool_use", kinds == ["thinking", "text", "tool_use"], f"got {kinds}")
    check("stop_reason is tool_use", out["stop_reason"] == "tool_use")
    check("usage carried", out["usage"]["output_tokens"] == 6)


def main() -> None:
    test_request_direction()
    test_parser()
    test_bare_calls()
    test_stream_simple_call()
    test_stream_with_prose()
    test_stream_text_only()
    test_stream_malformed_json()
    test_stream_thinking()
    test_native_unchanged()
    test_nonstream()

    print()
    if FAILED:
        print(f"{len(FAILED)} failed: {', '.join(FAILED)}")
        sys.exit(1)
    print("all good - shim mode is wire-compatible on both directions")


if __name__ == "__main__":
    main()
