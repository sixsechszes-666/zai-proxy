"""
test_web.py -- proves the web path's translation without a browser or a network.

The web upstream is not an OpenAI backend: it takes one prompt string, ignores
`messages`, and answers in `chat:completion` frames with a `phase` on each one.
Four things have to hold for it to carry a Claude Code session:

  1. the whole Anthropic conversation, tools included, survives the trip into
     a single prompt string;
  2. the frames come back out as the OpenAI delta shape the rest of the proxy
     already consumes;
  3. SSE text arriving in arbitrary chunks still parses, because the page
     hands it over a poll and can cut a frame anywhere;
  4. a non-streaming call reassembles into a well formed OpenAI response.

Nothing here touches the network, so it costs no quota -- which matters,
because the endpoint rate limits hard and a real request is one of only a few
dozen an hour.

  python test_web.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys

import webclient as W

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def frame(phase: str, **kw) -> str:
    """One `chat:completion` frame, shaped the way the endpoint sends it."""
    body = {"phase": phase}
    body.update(kw)
    return "data: " + json.dumps({"type": "chat:completion", "data": body})


# --------------------------------------------------------------------------
# the request direction
# --------------------------------------------------------------------------

def test_flatten() -> None:
    print("flatten: the transcript becomes one prompt string")
    text = W.flatten([
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
    ])
    check("system is labelled", "System: You are terse." in text)
    check("user is labelled", "User: hi" in text)
    check("assistant is labelled", "Assistant: hello" in text)
    check("turns stay in order",
          text.index("User: hi") < text.index("Assistant: hello")
          < text.index("User: again"))
    check("ends with an assistant cue", text.rstrip().endswith("Assistant:"))


def test_flatten_blocks() -> None:
    print("flatten: block content and tool traffic")
    text = W.flatten([
        {"role": "user", "content": [{"type": "text", "text": "part one"},
                                     {"type": "text", "text": "part two"}]},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "Bash",
                                      "arguments": '{"command":"ls"}'}}]},
        {"role": "tool", "content": "a.txt"},
    ])
    check("blocks are joined", "part one\npart two" in text)
    check("an OpenAI call is described",
          "[called Bash with {\"command\":\"ls\"}]" in text)
    check("a tool result is labelled", "Tool result: a.txt" in text)


def test_request_chain() -> None:
    """What Claude Code sends, all the way to the string the endpoint reads."""
    print("request: Claude Code -> shim -> flatten")
    import proxy as P

    req = {
        "model": "claude-sonnet-5",
        "max_tokens": 1024,
        "system": "Be brief.",
        "tools": [{"name": "Bash", "description": "Run a shell command",
                   "input_schema": {"type": "object",
                                    "properties": {"command": {"type": "string"}},
                                    "required": ["command"]}}],
        "messages": [
            {"role": "user", "content": "list the files"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "Bash",
                 "input": {"command": "ls -la"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": "a.txt"}]},
        ],
    }
    oai = P.anthropic_to_openai(P.shimify(json.loads(json.dumps(req))))
    text = W.flatten(oai["messages"])

    check("the schema is in the prompt", "### Bash" in text)
    check("the schema shows its arguments", '"command"' in text)
    check("a previous call is in the prompt",
          '<tool_call>{"name": "Bash", "arguments": {"command": "ls -la"}}</tool_call>' in text)
    check("a previous result is in the prompt",
          '<tool_result tool="Bash" id="toolu_1">' in text)
    check("the result body is in the prompt", "a.txt" in text)
    check("the question is still there", "list the files" in text)
    check("the turn ends on an assistant cue", text.rstrip().endswith("Assistant:"))


class FakeSession:
    """Enough of a WebSession to watch what web_stream asks it to build."""
    def __init__(self) -> None:
        self.seen: dict = {}

    async def build(self, prompt, **kw):
        self.seen["prompt"] = prompt
        self.seen["kw"] = kw
        return ("https://chat.z.ai/api/v2/chat/completions?q=1",
                {"X-Signature": "sig"},
                {"stream": True, "signature_prompt": prompt, "messages": []})

    async def captcha(self) -> str:
        return "CAPTCHA-TOKEN"


JUST_A_PROMPT = {"model": "glm-5.3-flash",
                 "messages": [{"role": "user", "content": "say PONG"}]}


def test_web_call_shape() -> None:
    print("web_stream: the call it builds")
    oai = dict(JUST_A_PROMPT, stream=True)
    session = FakeSession()
    stream = asyncio.run(W.web_stream(session, oai))

    check("the prompt is the flattened transcript",
          stream.body["signature_prompt"] == W.flatten(oai["messages"]))
    check("no tools reach the endpoint", "tools" not in stream.body)
    check("a fresh captcha rides along",
          stream.body["captcha_verify_param"] == "CAPTCHA-TOKEN")
    check("the model is the web one", session.seen["kw"].get("model") == W.WEB_MODEL)
    check("the signature header is kept",
          stream.headers.get("X-Signature") == "sig")


# --------------------------------------------------------------------------
# the reply direction
# --------------------------------------------------------------------------

def lines_of(*raw: str) -> list[str]:
    """Run raw SSE lines straight through the frame translator."""
    stream = W.WebStream("u", {}, {})
    out: list[str] = []
    for line in raw:
        out.extend(stream._line(line))
    return out


def test_frames() -> None:
    print("frames: phase -> OpenAI delta")
    got = lines_of(frame("thinking", delta_content="weighing"))
    check("thinking becomes reasoning_content",
          json.loads(got[0][5:])["choices"][0]["delta"]
          == {"reasoning_content": "weighing"})

    got = lines_of(frame("answer", delta_content="PONG"))
    check("answer becomes content",
          json.loads(got[0][5:])["choices"][0]["delta"] == {"content": "PONG"})

    got = lines_of(frame("other", usage={"prompt_tokens": 18,
                                         "completion_tokens": 3}))
    chunk = json.loads(got[0][5:])
    check("usage passes through", chunk["usage"]["prompt_tokens"] == 18)
    check("a usage frame carries no delta", chunk["choices"][0]["delta"] == {})

    got = lines_of(frame("done", done=True))
    check("done becomes finish_reason",
          json.loads(got[0][5:])["choices"][0]["finish_reason"] == "stop")


def test_frame_noise() -> None:
    print("frames: what should be ignored")
    check("a blank line", lines_of("") == [])
    check("a non-data line", lines_of("event: ping") == [])
    check("[DONE]", lines_of("data: [DONE]") == [])
    check("broken json", lines_of("data: {oops") == [])
    check("an unknown phase", lines_of(frame("unknown", delta_content="x")) == [])
    check("a frame with nothing in it", lines_of(frame("answer")) == [])


def test_frame_error() -> None:
    print("frames: an error frame")
    try:
        lines_of(frame("answer", error={"code": "INTERNAL_ERROR"}))
    except RuntimeError as exc:
        check("an error raises with the code", "INTERNAL_ERROR" in str(exc))
    else:
        check("an error raises with the code", False, "nothing raised")


# --------------------------------------------------------------------------
# the page, faked
#
# aiter_lines is exercised for real: only the CDP connection and the page's
# answers are replaced, so the drain loop, the line buffering and the status
# check all run the way they do against a live tab.

CANNED = [
    frame("thinking", delta_content="the user wants "),
    frame("thinking", delta_content="PONG"),
    frame("answer", delta_content="PO"),
    frame("answer", delta_content="NG"),
    frame("other", usage={"prompt_tokens": 18, "completion_tokens": 3}),
    frame("done", done=True),
]
CANNED_TEXT = "\n".join(CANNED) + "\ndata: [DONE]\n"


class FakeWS:
    async def close(self) -> None:
        pass


@contextlib.contextmanager
def page(chunks: list[str], status: int = 200):
    """Answer the stream's CDP calls with `chunks` instead of a browser."""
    saved = (W.websockets.connect, W._page_ws_url, W.WebStream._ask)
    state = {"i": 0}

    async def connect(url, **kw):
        return FakeWS()

    async def page_url():
        return "ws://fake"

    async def ask(self, ws, mid, js, await_promise=True):
        if mid == 1:                      # the call that starts the fetch
            return "started"
        i = state["i"]                    # every other call is a drain
        if i >= len(chunks):
            return json.dumps({"t": "", "done": True, "status": status})
        state["i"] = i + 1
        return json.dumps({"t": chunks[i], "done": False, "status": status})

    W.websockets.connect, W._page_ws_url, W.WebStream._ask = connect, page_url, ask
    try:
        yield
    finally:
        (W.websockets.connect, W._page_ws_url, W.WebStream._ask) = saved


def lines_over(chunks: list[str], status: int = 200) -> list[str]:
    out: list[str] = []

    async def go():
        async for line in W.WebStream("u", {}, {}).aiter_lines():
            out.append(line)

    with page(chunks, status):
        asyncio.run(go())
    return out


def test_stream_chunking() -> None:
    print("stream: what the page hands over, however it cuts it")
    whole = lines_over([CANNED_TEXT])
    check("a single chunk gives every frame", len(whole) == 7, f"got {len(whole)}")
    check("the last line is [DONE]", whole[-1] == "data: [DONE]")
    check("thinking came first", "reasoning_content" in whole[0])
    check("the answer came after", '"content": "PO"' in whole[2])

    per_char = lines_over(list(CANNED_TEXT))
    check("one character per chunk gives the same lines", per_char == whole,
          f"{len(per_char)} vs {len(whole)}")

    ugly = lines_over([CANNED_TEXT[:37], CANNED_TEXT[37:120], CANNED_TEXT[120:]])
    check("splits inside frames give the same lines", ugly == whole,
          f"{len(ugly)} vs {len(whole)}")


def test_stream_status() -> None:
    print("stream: a non-200 status becomes an error")
    try:
        lines_over([CANNED_TEXT], status=405)
    except RuntimeError as exc:
        check("the status is named", "405" in str(exc), str(exc)[:80])
    else:
        check("the status is named", False, "nothing raised")


def collect_from(chunks: list[str], status: int = 200) -> dict:
    with page(chunks, status):
        return asyncio.run(W.collect(FakeSession(), JUST_A_PROMPT))


def test_collect() -> None:
    print("collect: a non-streaming call reassembles")
    data = collect_from([CANNED_TEXT])
    msg = data["choices"][0]["message"]
    check("content is joined", msg["content"] == "PONG", repr(msg.get("content")))
    check("reasoning is joined", msg["reasoning_content"] == "the user wants PONG")
    check("usage is carried", data["usage"]["prompt_tokens"] == 18)
    check("finish_reason is set", data["choices"][0]["finish_reason"] == "stop")


def test_collect_through_the_proxy() -> None:
    """The whole reply direction: canned frames -> an Anthropic message."""
    print("full reply: canned frames -> Anthropic blocks")
    import proxy as P

    data = collect_from([CANNED_TEXT])
    out = P.openai_to_anthropic_shim(data, "claude-sonnet-5")
    kinds = [b["type"] for b in out["content"]]
    check("a thinking block and a text block", kinds == ["thinking", "text"],
          f"got {kinds}")
    check("the text is the answer", out["content"][1]["text"] == "PONG")
    check("stop_reason is end_turn", out["stop_reason"] == "end_turn")
    check("usage is mapped", out["usage"]["input_tokens"] == 18)


def test_collect_detects_the_wall() -> None:
    print("collect: the 405 wall surfaces as an error, not as empty text")
    try:
        collect_from([CANNED_TEXT], status=405)
    except RuntimeError as exc:
        check("the status is named", "405" in str(exc), str(exc)[:80])
    else:
        check("the status is named", False, "nothing raised")


def test_wall_message() -> None:
    """What Claude Code is told when the exit IP is blocked."""
    print("the 405 message a user actually gets")
    import proxy as P

    real = RuntimeError('chat.z.ai returned 405: <!doctypehtml><html>...')
    msg = P.web_error(real)
    check("the html is not forwarded", "doctype" not in msg)
    check("the status is named", "405" in msg)
    check("the fix is named", "egress" in msg or "exit IP" in msg, msg[:80])
    check("the account is cleared", "account" in msg)

    other = RuntimeError("chat.z.ai returned 500: boom")
    check("other failures pass through", P.web_error(other) == str(other))


def main() -> None:
    test_flatten()
    test_flatten_blocks()
    test_request_chain()
    test_web_call_shape()
    test_frames()
    test_frame_noise()
    test_frame_error()
    test_stream_chunking()
    test_stream_status()
    test_collect()
    test_collect_through_the_proxy()
    test_collect_detects_the_wall()
    test_wall_message()

    print()
    if FAILED:
        print(f"{len(FAILED)} failed: {', '.join(FAILED)}")
        sys.exit(1)
    print("all good - the web path translates on both directions")


if __name__ == "__main__":
    main()
