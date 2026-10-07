"""
mock_shim.py -- a fake backend for shim mode.

Sits where a chat-only endpoint would. It refuses to be told about tools (as
the z.ai web endpoint does), so the only way a tool call can reach it is inside
the prompt text. It answers by reading the shim envelope out of the system
prompt and replying in kind, with the reply deliberately chopped at the worst
possible places: mid-marker, mid-key, mid-object.

  python mock_shim.py                 # 127.0.0.1:8899
  GET  /checks                        # what the last request looked like
"""

import json
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

# what the last request proved about the shim, read back by the e2e test
LAST: dict = {}


def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
    body = {"id": "chatcmpl-shim", "object": "chat.completion.chunk", "model": "glm-5.3-flash",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        body["usage"] = usage
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


def split_awkward(text: str) -> list[str]:
    """Cut where a naive incremental parser would trip."""
    cuts: set[int] = set()
    for tok in ("<tool_call>", "</tool_call>", "<tool_result"):
        i = text.find(tok)
        if i >= 0:
            cuts.add(i + 4)            # in the middle of the marker
    for key in ('"name"', '"arguments"'):
        i = text.find(key)
        if i >= 0:
            cuts.add(i + 3)            # in the middle of a key
    i = text.find("{", text.find('"arguments"'))
    if i >= 0:
        cuts.add(i + 1)                # just inside the args object
    ordered = sorted(c for c in cuts if 0 < c < len(text))
    out, prev = [], 0
    for c in ordered:
        out.append(text[prev:c])
        prev = c
    out.append(text[prev:])
    return [p for p in out if p]


def _all_text(msgs: list) -> str:
    """The readable text of a conversation.

    Not json.dumps: that escapes the quotes inside the envelope and the
    markers would never match.
    """
    out: list[str] = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    out.append(b.get("text", ""))
    return "\n".join(out)


def inspect(body: dict) -> dict:
    msgs = body.get("messages") or []
    sysmsg = msgs[0].get("content", "") if msgs and msgs[0].get("role") == "system" else ""
    if not isinstance(sysmsg, str):
        sysmsg = _all_text([{"content": sysmsg}])
    dumped = _all_text(msgs)
    return {
        "tools_absent": "tools" not in body,
        "tool_choice_absent": "tool_choice" not in body,
        "schemas_in_system": "### Bash" in sysmsg,
        "envelope_taught": "<tool_call>" in sysmsg,
        "result_seen": '<tool_result tool="Bash"' in dumped,
        "result_body_seen": "a.txt" in dumped,
        "system_head": sysmsg[:160],
        "message_count": len(msgs),
    }


def decide(body: dict, checks: dict) -> str:
    """The reply text. Coming back from a tool result means the loop closed."""
    if checks["result_seen"]:
        return "The folder holds a.txt and b.txt."
    call = json.dumps({"name": "Bash", "arguments": {"command": "ls -la"}}, ensure_ascii=False)
    return "Checking the folder.\n<tool_call>" + call + "</tool_call>"


@app.get("/checks")
async def checks() -> dict:
    return LAST


@app.post("/chat/completions")
async def completions(request: Request):
    global LAST
    body = await request.json()
    checks = inspect(body)
    LAST = checks
    text = decide(body, checks)
    model = body.get("model", "glm-5.3-flash")

    if not body.get("stream"):
        return JSONResponse({
            "id": "chatcmpl-shim", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text,
                                                 "reasoning_content": "weighing it"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 12},
        })

    async def gen():
        for piece in split_awkward(text):
            yield chunk({"content": piece})
        yield chunk({"reasoning_content": ""})  # a quiet chunk, must not disturb
        yield chunk({}, finish="stop", usage={"prompt_tokens": 40, "completion_tokens": 12})
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8899, log_level="error")
