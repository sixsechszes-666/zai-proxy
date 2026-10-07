"""
mock_upstream.py -- a fake OpenAI-compatible backend for testing zai_proxy
without a z.ai key and without spending quota.

  python mock_upstream.py            # listens on 127.0.0.1:8899
  ZAI_BASE_URL=http://127.0.0.1:8899 python proxy.py
  python selftest.py

It answers /chat/completions by inspection:
  last message is a tool result -> mention its content  (round trip)
  tools present                 -> call get_weather      (tool use)
  otherwise                     -> "PONG"                (plain text)
and it splits tool arguments across chunks on purpose, to exercise the
stream buffer.
"""

import json
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()


def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
    body = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "model": "glm-4.6",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        body["usage"] = usage
    return f"data: {json.dumps(body)}\n\n"


def decide(body: dict) -> tuple[list[str], str | None, dict]:
    msgs = body.get("messages") or []
    results = [m for m in msgs if m.get("role") == "tool"]

    if results:
        text = "The weather is " + str(results[-1].get("content", ""))
        return [chunk({"content": text})], "stop", {"prompt_tokens": 200, "completion_tokens": 12}

    if body.get("tools"):
        name = body["tools"][0]["function"]["name"]
        args = json.dumps({"city": "Tokyo", "units": "metric"})
        parts = [args[:6], args[6:14], args[14:]]
        out = [
            chunk({"role": "assistant", "content": ""}),
            chunk({"tool_calls": [{"index": 0, "id": f"call_{uuid.uuid4().hex[:8]}",
                                   "type": "function", "function": {"name": name, "arguments": ""}}]}),
        ]
        for p in parts:
            out.append(chunk({"tool_calls": [{"index": 0, "function": {"arguments": p}}]}))
        out.append(chunk({}, finish="tool_calls", usage={"prompt_tokens": 300, "completion_tokens": 25}))
        return out, "tool_calls", {"prompt_tokens": 300, "completion_tokens": 25}

    out = [chunk({"role": "assistant", "content": ""}),
           chunk({"content": "PONG"}),
           chunk({}, finish="stop", usage={"prompt_tokens": 50, "completion_tokens": 2})]
    return out, "stop", {"prompt_tokens": 50, "completion_tokens": 2}


@app.post("/chat/completions")
async def completions(request: Request):
    body = await request.json()

    if not body.get("stream"):
        msgs = body.get("messages") or []
        results = [m for m in msgs if m.get("role") == "tool"]
        if results:
            msg = {"role": "assistant", "content": "The weather is " + str(results[-1].get("content", ""))}
            finish = "stop"
        elif body.get("tools"):
            name = body["tools"][0]["function"]["name"]
            msg = {"role": "assistant", "content": None,
                   "tool_calls": [{"id": "call_mock1", "type": "function",
                                   "function": {"name": name,
                                                "arguments": json.dumps({"city": "Tokyo", "units": "metric"})}}]}
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": "PONG"}
            finish = "stop"
        return JSONResponse({"id": "chatcmpl-mock", "object": "chat.completion", "model": "glm-4.6",
                             "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                             "usage": {"prompt_tokens": 50, "completion_tokens": 8}})

    events, _, usage = decide(body)

    async def gen():
        for ev in events:
            yield ev
            time.sleep(0.01)
        yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    print("mock upstream on http://127.0.0.1:8899")
    uvicorn.run(app, host="127.0.0.1", port=8899, log_level="warning")
