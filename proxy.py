"""
zai_proxy -- Anthropic Messages API  ->  OpenAI Chat Completions bridge.

Claude Code speaks the Anthropic wire protocol. z.ai speaks OpenAI. This sits
in the middle, translates both directions, and streams SSE back in Anthropic
shape so Claude Code's client-side machinery (tools, TodoWrite, skills,
subagents, MCP) keeps working untouched.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, AsyncIterator, Iterable

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

import toolshim

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

UPSTREAM_BASE = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4").rstrip("/")
UPSTREAM_KEY = os.environ.get("ZAI_API_KEY", "").strip()

# Which backend the proxy talks to.
#   api = the z.ai platform API over HTTPS, with ZAI_API_KEY. Sends a real
#         `tools` array, so tool calling is native.
#   web = chat.z.ai as the browser session, through the logged-in tab. No key
#         and no tools array: the endpoint takes a prompt string and nothing
#         else, so this mode forces the tool shim on. Needs a signed-in
#         chat.z.ai tab and Chrome on the CDP port.
UPSTREAM_MODE = os.environ.get("ZAI_UPSTREAM", "api").strip().lower()
WEB_MODE = UPSTREAM_MODE == "web"

LISTEN_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("PROXY_PORT", "8788"))
PROXY_TOKEN = os.environ.get("PROXY_TOKEN", "").strip()  # optional: require this as the client key

DEFAULT_MODEL = os.environ.get("ZAI_MODEL", "glm-5.3-flash")
SMALL_MODEL = os.environ.get("ZAI_SMALL_MODEL", "glm-5.3-flash")
MAX_TOKENS_CEILING = int(os.environ.get("ZAI_MAX_TOKENS", "131072"))

# GLM-5.3-Flash always reasons; the switch cannot be turned off, only sized.
#   none = never send a thinking param, let the model decide  (right for 5.3)
#   auto = forward thinking only when the Anthropic request asked for it
#   on   = always ask for thinking     (GLM-4.5 needs this)
#   off  = always ask for it off       (ignored by 5.3, works on 4.5/4.6)
THINKING = os.environ.get("ZAI_THINKING", "none").lower()

# low | high | max  -- GLM-5.3 only. Default upstream is "max", which is a lot
# of thinking for every Claude Code turn. "low" is usually the right call for
# the small/fast slot. Empty means send nothing and take the upstream default.
REASONING_EFFORT = os.environ.get("ZAI_REASONING_EFFORT", "").strip()
TEMPERATURE_OVERRIDE = os.environ.get("ZAI_TEMPERATURE")  # None => passthrough

# GLM's reasoning_content -> Anthropic thinking blocks (display only)
EXPOSE_REASONING = os.environ.get("ZAI_EXPOSE_REASONING", "1").lower() not in ("0", "false", "no")
STRIP_REASONING_ON_SEND = True  # never send thinking blocks back upstream

# How tool calling is carried upstream.
#   native = send a `tools` array in the request. Needs a backend that takes
#            one: the z.ai platform API does, the z.ai *web* endpoint does not.
#   shim   = render the schemas into the prompt and parse the calls back out
#            of the text stream. Works on any chat-only backend.
#   off    = drop tools and run as plain chat.
TOOLS_MODE = os.environ.get("ZAI_TOOLS", "native").strip().lower()

DUMP = os.environ.get("ZAI_DUMP", "0").lower() in ("1", "true", "yes")
DUMP_DIR = os.environ.get("ZAI_DUMP_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps"))

# Outbound proxy. z.ai is often only reachable through Clash from RU, so the
# default is to honour the system env (HTTPS_PROXY / ALL_PROXY / NO_PROXY).
#   ZAI_DIRECT=1                  ignore env entirely, go straight out
#   ZAI_HTTP_PROXY=socks5://...   force one explicit proxy
# SOCKS needs:  pip install "httpx[socks]"
ZAI_DIRECT = os.environ.get("ZAI_DIRECT", "0").lower() in ("1", "true", "yes")
ZAI_HTTP_PROXY = os.environ.get("ZAI_HTTP_PROXY", "").strip()

TIMEOUT = httpx.Timeout(connect=30.0, read=900.0, write=120.0, pool=30.0)

STOP_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def log(*a: Any) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def dump(tag: str, obj: Any) -> None:
    if not DUMP:
        return
    try:
        os.makedirs(DUMP_DIR, exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}-{uuid.uuid4().hex[:6]}.json"
        with open(os.path.join(DUMP_DIR, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
    except Exception as exc:  # dumping must never break the request
        log("dump failed:", exc)


def resolve_model(name: str | None) -> str:
    """Claude Code sends claude-* names. Map them onto GLM."""
    if not name:
        return DEFAULT_MODEL
    n = name.lower()
    if n.startswith(("glm", "zai/", "z-ai")):
        return name
    if "haiku" in n or "small" in n or "flash" in n:
        return SMALL_MODEL
    return DEFAULT_MODEL


def repair_json(raw: str | None) -> Any:
    """Models emit almost-JSON for tool arguments. Salvage what we can."""
    if raw is None or not str(raw).strip():
        return {}
    text = str(raw).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for _ in range(8):
        opens = text.count("{") - text.count("}")
        brks = text.count("[") - text.count("]")
        if opens <= 0 and brks <= 0:
            break
        text += "]" * max(0, brks) + "}" * max(0, opens)
        try:
            return json.loads(text)
        except Exception:
            continue
    log("tool args unrecoverable, sending {}:", raw[:200])
    return {}


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


# --------------------------------------------------------------------------
# Anthropic request -> OpenAI request
# --------------------------------------------------------------------------

def anthropic_to_openai(req: dict) -> dict:
    messages: list[dict] = []

    system = req.get("system")
    if system:
        sys_text = system if isinstance(system, str) else text_of(system)
        if sys_text.strip():
            messages.append({"role": "system", "content": sys_text})

    for m in req.get("messages") or []:
        role = m.get("role")
        content = m.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue

        if role == "assistant":
            texts: list[str] = []
            calls: list[dict] = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    texts.append(b.get("text", ""))
                elif bt == "tool_use":
                    calls.append({
                        "id": b.get("id") or f"call_{uuid.uuid4().hex[:16]}",
                        "type": "function",
                        "function": {
                            "name": b.get("name") or "unknown",
                            "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False),
                        },
                    })
                # thinking/redacted_thinking dropped -- never round-trip them upstream
            msg: dict = {"role": "assistant", "content": "\n".join(t for t in texts if t) or None}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)

        elif role == "user":
            parts: list[dict] = []
            tool_msgs: list[dict] = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    parts.append({"type": "text", "text": b.get("text", "")})
                elif bt == "image":
                    src = b.get("source") or {}
                    if src.get("type") == "base64":
                        parts.append({"type": "image_url", "image_url": {
                            "url": f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"}})
                    elif src.get("type") == "url":
                        parts.append({"type": "image_url", "image_url": {"url": src.get("url", "")}})
                elif bt == "tool_result":
                    body = b.get("content")
                    txt = body if isinstance(body, str) else text_of(body)
                    if isinstance(body, list):
                        # non-text tool results (images etc.) degrade to a marker
                        for sub in body:
                            if isinstance(sub, dict) and sub.get("type") == "image":
                                txt += "\n[image result omitted]"
                    tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id") or "",
                        "content": txt.strip() or "(empty result)",
                    })

            messages.extend(tool_msgs)
            if parts:
                if len(parts) == 1 and parts[0]["type"] == "text":
                    messages.append({"role": "user", "content": parts[0]["text"]})
                else:
                    messages.append({"role": "user", "content": parts})

    messages = fix_tool_pairs(messages)
    messages = merge_consecutive(messages)

    out: dict = {
        "model": resolve_model(req.get("model")),
        "messages": messages,
        "stream": bool(req.get("stream")),
    }

    max_tokens = req.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        out["max_tokens"] = min(max_tokens, MAX_TOKENS_CEILING)

    if TEMPERATURE_OVERRIDE is not None:
        try:
            out["temperature"] = float(TEMPERATURE_OVERRIDE)
        except ValueError:
            pass
    elif isinstance(req.get("temperature"), (int, float)):
        out["temperature"] = req["temperature"]

    if isinstance(req.get("top_p"), (int, float)):
        out["top_p"] = req["top_p"]
    # top_k has no OpenAI equivalent -- dropped

    stops = req.get("stop_sequences")
    if stops:
        out["stop"] = stops[:4]

    tools = req.get("tools")
    if tools:
        converted = []
        for t in tools:
            if not isinstance(t, dict) or not t.get("name"):
                continue
            schema = t.get("input_schema") or {"type": "object", "properties": {}}
            schema = {k: v for k, v in schema.items() if k != "cache_control"}
            converted.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": (t.get("description") or "")[:4096],
                    "parameters": schema,
                },
            })
        if converted:
            out["tools"] = converted
            tc = req.get("tool_choice") or {}
            kind = tc.get("type")
            if kind == "auto":
                out["tool_choice"] = "auto"
            elif kind == "any":
                out["tool_choice"] = "required"
            elif kind == "none":
                out["tool_choice"] = "none"
            elif kind == "tool" and tc.get("name"):
                out["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
            else:
                out["tool_choice"] = "auto"

    # GLM thinking switch. "none" is right for GLM-5.3-Flash: its reasoning is
    # always on and cannot be disabled, so asking is pointless and asking for
    # "disabled" can be rejected outright.
    wants_thinking = isinstance(req.get("thinking"), dict) and req["thinking"].get("type") == "enabled"
    if THINKING == "on" or (THINKING == "auto" and wants_thinking):
        out["thinking"] = {"type": "enabled"}
    elif THINKING == "off":
        out["thinking"] = {"type": "disabled"}

    if REASONING_EFFORT:
        out["reasoning_effort"] = REASONING_EFFORT

    return out


def shimify(req: dict) -> dict:
    """Anthropic request -> one that carries no tool schema at all.

    The schemas move into the system prompt, each `tool_use` becomes the text
    envelope the model was taught to write, each `tool_result` comes back as a
    <tool_result> block. What remains is an ordinary chat request, which is the
    only kind a chat-only backend will accept.
    """
    tools = req.get("tools") or []
    out = dict(req)
    out.pop("tools", None)

    # tool_result carries only an id, so names have to be recovered first
    names: dict[str, str] = {}
    for m in req.get("messages") or []:
        c = m.get("content")
        if m.get("role") == "assistant" and isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id"):
                    names[b["id"]] = b.get("name") or "unknown"

    msgs: list[dict] = []
    for m in req.get("messages") or []:
        c = m.get("content")
        if not isinstance(c, list):
            msgs.append(m)
            continue
        blocks: list[dict] = []
        for b in c:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "tool_use":
                blocks.append({"type": "text", "text": toolshim.render_call(
                    b.get("name") or "unknown", b.get("input") or {})})
            elif bt == "tool_result":
                tid = b.get("tool_use_id") or ""
                body = b.get("content")
                txt = body if isinstance(body, str) else text_of(body)
                if isinstance(body, list):
                    for sub in body:
                        if isinstance(sub, dict) and sub.get("type") == "image":
                            txt += "\n[image result omitted]"
                blocks.append({"type": "text", "text": toolshim.render_result(
                    names.get(tid, "tool"), tid, txt)})
            elif bt == "thinking":
                continue  # never round-trip reasoning
            else:
                blocks.append(b)
        msgs.append({**m, "content": blocks})
    out["messages"] = msgs

    addendum = toolshim.render_tools(tools)
    if addendum:
        tc = req.get("tool_choice") or {}
        if tc.get("type") in ("any", "tool"):
            want = tc.get("name")
            addendum += ("\n\nThis turn you MUST call a tool"
                         + (f", specifically `{want}`" if want else "")
                         + f". Reply with {toolshim.OPEN} block(s) and nothing else.\n")
        sys_text = text_of(req.get("system")) if req.get("system") else ""
        out["system"] = f"{sys_text}\n\n{addendum}".strip() if sys_text.strip() else addendum

    return out


def fix_tool_pairs(msgs: list[dict]) -> list[dict]:
    """Every tool_call needs a matching tool message, or providers 400."""
    out: list[dict] = []
    i, n = 0, len(msgs)
    while i < n:
        m = msgs[i]
        if m.get("role") == "tool":
            i += 1  # orphan result with no preceding call -- drop
            continue
        out.append(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids = []
            for k, tc in enumerate(m["tool_calls"]):
                if not tc.get("id"):
                    tc["id"] = f"call_{uuid.uuid4().hex[:16]}"
                ids.append(tc["id"])
            seen: dict[str, dict] = {}
            j = i + 1
            while j < n and msgs[j].get("role") == "tool":
                seen[msgs[j].get("tool_call_id")] = msgs[j]
                j += 1
            for tid in ids:
                out.append(seen.get(tid) or {"role": "tool", "tool_call_id": tid, "content": "(no result)"})
            i = j
            continue
        i += 1
    return out


def merge_consecutive(msgs: list[dict]) -> list[dict]:
    """Some backends dislike two user turns in a row."""
    out: list[dict] = []
    for m in msgs:
        if (out and out[-1].get("role") == m.get("role") == "user"
                and isinstance(out[-1].get("content"), str) and isinstance(m.get("content"), str)):
            out[-1]["content"] += "\n\n" + m["content"]
        else:
            out.append(m)
    return out


# --------------------------------------------------------------------------
# OpenAI response -> Anthropic response
# --------------------------------------------------------------------------

def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def openai_to_anthropic(data: dict, model: str) -> dict:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    blocks: list[dict] = []

    rc = msg.get("reasoning_content") or msg.get("reasoning")
    if rc and EXPOSE_REASONING:
        blocks.append({"type": "thinking", "thinking": rc, "signature": ""})
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:20]}",
            "name": fn.get("name") or "unknown",
            "input": repair_json(fn.get("arguments")),
        })

    has_tool = any(b["type"] == "tool_use" for b in blocks)
    stop = "tool_use" if has_tool else STOP_MAP.get(choice.get("finish_reason"), "end_turn")
    usage = data.get("usage") or {}

    return {
        "id": data.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks or [{"type": "text", "text": ""}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("completion_tokens") or 0,
        },
    }


def openai_to_anthropic_shim(data: dict, model: str,
                             names: Iterable[str] | None = None) -> dict:
    """The non-streaming path for shim mode: calls parsed out of the text."""
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    blocks: list[dict] = []

    rc = msg.get("reasoning_content") or msg.get("reasoning")
    if rc and EXPOSE_REASONING:
        blocks.append({"type": "thinking", "thinking": rc, "signature": ""})

    for kind, val in toolshim.parse_full(msg.get("content") or "", names):
        if kind == "text":
            if val.strip():
                blocks.append({"type": "text", "text": val})
        else:
            blocks.append({"type": "tool_use", "id": toolshim.new_tool_id(),
                           "name": val["name"], "input": val["arguments"]})

    has_tool = any(b["type"] == "tool_use" for b in blocks)
    stop = "tool_use" if has_tool else STOP_MAP.get(choice.get("finish_reason"), "end_turn")
    usage = data.get("usage") or {}

    return {
        "id": data.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks or [{"type": "text", "text": ""}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("completion_tokens") or 0,
        },
    }


def web_error(exc: Exception) -> str:
    """A web upstream failure, in the one form that is actionable.

    The 405 is not a rate limit on the account. Nine requests in a burst trip
    it, the app's own UI request gets it too while it lasts, and a fresh egress
    node clears it immediately. The raw message is 300 characters of Aliyun WAF
    HTML with a traceid in it, which tells whoever reads it nothing, so it is
    replaced with the two facts that matter: what happened and what fixes it.
    """
    text = str(exc)
    if " 405" in text or text.startswith("405"):
        return ("chat.z.ai answered 405 - the exit IP is blocked, on the "
                "completions path only. A different egress node clears it at "
                "once and the account is unaffected. See WEB_MODE.md.")
    return text


async def stream_anthropic(resp: Any, model: str, shim: bool = False,
                           names: Iterable[str] | None = None) -> AsyncIterator[bytes]:
    """OpenAI SSE in, Anthropic SSE out.

    `resp` is anything with an async `aiter_lines()`: an httpx response on the
    api path, a `webclient.WebStream` on the web path, which has already
    reshaped z.ai's frames into the same delta shape.

    `names` are the tools offered this turn, handed to the shim parser so it
    can tell an unwrapped call from an object the model wrote in prose.
    """
    parser = toolshim.ToolStreamParser(names) if shim else None
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield sse("ping", {"type": "ping"})

    next_index = 0
    open_kind: str | None = None
    open_index = -1

    # tool args are buffered per OpenAI tool index and flushed whole, so
    # interleaved tool_call deltas can't produce a reopened content block
    buf_idx: int | None = None
    buf_id = ""
    buf_name = ""
    buf_args: list[str] = []
    emitted_tool = False

    def flush() -> list[bytes]:
        nonlocal buf_idx, buf_id, buf_name, buf_args, open_kind, open_index, next_index, emitted_tool
        if buf_idx is None:
            return []
        out: list[bytes] = []
        if open_kind is not None:
            out.append(sse("content_block_stop", {"type": "content_block_stop", "index": open_index}))
        block_index = next_index
        next_index += 1
        open_kind, open_index = "tool", block_index
        emitted_tool = True
        out.append(sse("content_block_start", {
            "type": "content_block_start", "index": block_index,
            "content_block": {"type": "tool_use", "id": buf_id, "name": buf_name, "input": {}},
        }))
        args = "".join(buf_args).strip() or "{}"
        out.append(sse("content_block_delta", {
            "type": "content_block_delta", "index": block_index,
            "delta": {"type": "input_json_delta", "partial_json": args},
        }))
        buf_idx, buf_id, buf_name, buf_args = None, "", "", []
        return out

    def emit_call(name: str, args: dict) -> list[bytes]:
        """A completed shim call becomes one whole tool_use block.

        The arguments are already whole, so they go out as a single
        input_json_delta rather than being streamed piecemeal.
        """
        nonlocal open_kind, open_index, next_index, emitted_tool
        out: list[bytes] = []
        if open_kind is not None:
            out.append(sse("content_block_stop", {"type": "content_block_stop", "index": open_index}))
            open_kind = None
        block_index = next_index
        next_index += 1
        emitted_tool = True
        out.append(sse("content_block_start", {
            "type": "content_block_start", "index": block_index,
            "content_block": {"type": "tool_use", "id": toolshim.new_tool_id(),
                              "name": name, "input": {}},
        }))
        out.append(sse("content_block_delta", {
            "type": "content_block_delta", "index": block_index,
            "delta": {"type": "input_json_delta",
                      "partial_json": json.dumps(args or {}, ensure_ascii=False)},
        }))
        out.append(sse("content_block_stop", {"type": "content_block_stop", "index": block_index}))
        return out

    def switch(kind: str) -> list[bytes]:
        nonlocal open_kind, open_index, next_index
        out: list[bytes] = []
        if open_kind == kind:
            return out
        if open_kind is not None:
            out.append(sse("content_block_stop", {"type": "content_block_stop", "index": open_index}))
        open_kind, open_index = kind, next_index
        next_index += 1
        if kind == "text":
            out.append(sse("content_block_start", {
                "type": "content_block_start", "index": open_index,
                "content_block": {"type": "text", "text": ""},
            }))
        elif kind == "thinking":
            out.append(sse("content_block_start", {
                "type": "content_block_start", "index": open_index,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            }))
        return out

    def text_block(chunk: str) -> list[bytes]:
        return list(switch("text")) + [sse("content_block_delta", {
            "type": "content_block_delta", "index": open_index,
            "delta": {"type": "text_delta", "text": chunk},
        })]

    input_tokens = 0
    output_tokens = 0
    finish_reason: str | None = None
    # shim mode holds text back until it knows whether it is content or just
    # whitespace sitting between two calls
    pending_text = ""

    async for line in resp.aiter_lines():
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        u = chunk.get("usage")
        if isinstance(u, dict):
            input_tokens = u.get("prompt_tokens") or input_tokens
            output_tokens = u.get("completion_tokens") or output_tokens

        choices = chunk.get("choices") or []
        if not choices:
            continue
        ch = choices[0]
        delta = ch.get("delta") or {}
        if ch.get("finish_reason"):
            finish_reason = ch["finish_reason"]

        rc = delta.get("reasoning_content") or delta.get("reasoning")
        if rc and EXPOSE_REASONING:
            for b in switch("thinking"):
                yield b
            yield sse("content_block_delta", {
                "type": "content_block_delta", "index": open_index,
                "delta": {"type": "thinking_delta", "thinking": rc},
            })

        text = delta.get("content")
        if text and shim:
            for kind, val in parser.feed(text):
                if kind == "call":
                    # whitespace the model wrote between two calls is layout,
                    # not content, so it never earns a block of its own
                    if pending_text.strip():
                        for b in text_block(pending_text):
                            yield b
                    pending_text = ""
                    for b in emit_call(val["name"], val["arguments"]):
                        yield b
                else:
                    pending_text += val
                    if pending_text.strip():
                        for b in text_block(pending_text):
                            yield b
                        pending_text = ""
        elif text:
            if buf_idx is not None:
                for b in flush():
                    yield b
                open_kind = None  # force switch() to reopen a text block
            for b in switch("text"):
                yield b
            yield sse("content_block_delta", {
                "type": "content_block_delta", "index": open_index,
                "delta": {"type": "text_delta", "text": text},
            })

        for tc in (() if shim else delta.get("tool_calls") or []):
            idx = tc.get("index", 0)
            fn = tc.get("function") or {}
            if buf_idx != idx:
                for b in flush():
                    yield b
                buf_idx = idx
                buf_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:20]}"
                buf_name = fn.get("name") or ""
            else:
                if tc.get("id") and not buf_id:
                    buf_id = tc["id"]
                if fn.get("name") and not buf_name:
                    buf_name = fn["name"]
            if fn.get("arguments"):
                buf_args.append(fn["arguments"])

    if shim:
        for kind, val in parser.finish():
            if kind == "call":
                if pending_text.strip():
                    for b in text_block(pending_text):
                        yield b
                pending_text = ""
                for b in emit_call(val["name"], val["arguments"]):
                    yield b
            else:
                pending_text += val
        if pending_text.strip():
            for b in text_block(pending_text):
                yield b
        pending_text = ""

    for b in flush():
        yield b
    if open_kind is not None:
        yield sse("content_block_stop", {"type": "content_block_stop", "index": open_index})

    stop = "tool_use" if emitted_tool else STOP_MAP.get(finish_reason, "end_turn")
    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    })
    yield sse("message_stop", {"type": "message_stop"})


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------

app = FastAPI(title="zai_proxy", docs_url=None, redoc_url=None)

_client: httpx.AsyncClient | None = None


def http() -> httpx.AsyncClient:
    """Lazy so importing this module never touches proxy/socks config."""
    global _client
    if _client is None:
        kwargs: dict = {
            "timeout": TIMEOUT,
            "limits": httpx.Limits(max_connections=64, max_keepalive_connections=32),
            "trust_env": not ZAI_DIRECT,
        }
        if ZAI_HTTP_PROXY:
            kwargs["proxy"] = ZAI_HTTP_PROXY
            kwargs["trust_env"] = False
        _client = httpx.AsyncClient(**kwargs)
    return _client


_web_session: Any = None


def web_client():
    """The webclient module, imported on first use.

    Kept out of module scope because importing it clears the proxy env vars
    that have to be clear to reach CDP on loopback, and doing that at import
    time would break the api path's own proxy handling. Web mode owns those
    variables; the two modes are not meant to be mixed.
    """
    import webclient
    return webclient


async def web_session():
    """The browser session, refreshed when its token is close to expiring."""
    global _web_session
    if _web_session is None:
        _web_session = web_client().WebSession()
    if not _web_session.token or _web_session.expires_in() < 300:
        await _web_session.arefresh()
        log("web session refreshed, token expires in "
            f"{_web_session.expires_in():.0f}s")
    return _web_session


def err(message: str, kind: str = "api_error", status: int = 502) -> JSONResponse:
    return JSONResponse(status_code=status, content={"type": "error", "error": {"type": kind, "message": message}})


def authorized(request: Request) -> bool:
    if not PROXY_TOKEN:
        return True
    supplied = request.headers.get("x-api-key") or ""
    auth = request.headers.get("authorization") or ""
    if not supplied and auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    return supplied == PROXY_TOKEN


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "upstream_mode": UPSTREAM_MODE, "upstream": UPSTREAM_BASE,
            "key_set": bool(UPSTREAM_KEY), "default_model": DEFAULT_MODEL,
            "small_model": SMALL_MODEL, "tools_mode": TOOLS_MODE}


@app.get("/v1/models")
async def models() -> dict:
    now = int(time.time())
    return {"data": [
        {"id": DEFAULT_MODEL, "type": "model", "display_name": DEFAULT_MODEL, "created_at": now},
        {"id": SMALL_MODEL, "type": "model", "display_name": SMALL_MODEL, "created_at": now},
    ], "has_more": False, "first_id": DEFAULT_MODEL, "last_id": SMALL_MODEL}


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> dict:
    body = await request.json()
    chars = len(json.dumps(body.get("messages") or [], ensure_ascii=False)) + len(text_of(body.get("system")))
    return {"input_tokens": max(1, chars // 4)}


@app.post("/v1/messages")
async def messages(request: Request):
    if not authorized(request):
        return err("invalid proxy token", "authentication_error", 401)
    if not WEB_MODE and not UPSTREAM_KEY:
        return err("ZAI_API_KEY is not set on the proxy", "authentication_error", 500)

    try:
        body = await request.json()
    except Exception:
        return err("request body is not JSON", "invalid_request_error", 400)

    # The web endpoint takes a prompt string, so a tools array can never reach
    # it. Shim is the only way tools work there, and it is forced rather than
    # trusted to the env, because a silent "native" here would drop them.
    shim_on = WEB_MODE or TOOLS_MODE == "shim"
    # The offered names are what lets the parser accept an unmarked call: an
    # object naming a real tool is a call, an object naming anything else is
    # prose. Collected before shimify, which is where the array disappears.
    tool_names = {t.get("name") for t in (body.get("tools") or [])
                  if isinstance(t, dict) and t.get("name")}
    if shim_on:
        body = shimify(body)
    elif TOOLS_MODE == "off":
        body = {k: v for k, v in body.items() if k != "tools"}

    oai = anthropic_to_openai(body)
    model = oai["model"]
    want_stream = oai.pop("stream", False)
    dump("request", {"anthropic": body, "openai": oai, "tools_mode": TOOLS_MODE,
                     "upstream_mode": UPSTREAM_MODE})

    if WEB_MODE:
        wc = web_client()
        if not want_stream:
            try:
                data = await wc.collect(await web_session(), oai)
            except Exception as exc:
                log("web upstream failed:", exc)
                return err(web_error(exc))
            dump("response", data)
            convert = openai_to_anthropic_shim if shim_on else openai_to_anthropic
            return JSONResponse(content=convert(data, model, tool_names) if shim_on
                                else convert(data, model))
        try:
            stream = await wc.web_stream(await web_session(), oai)
        except Exception as exc:
            log("web upstream failed:", exc)
            return err(web_error(exc))

        async def gen_web() -> AsyncIterator[bytes]:
            try:
                async for chunk in stream_anthropic(stream, model, shim=shim_on,
                                                    names=tool_names):
                    yield chunk
            except Exception as exc:
                # the WAF refusal lands here: a 405 with an HTML body, which
                # the frame parser has nothing to say about
                log("web stream aborted:", exc)
                yield sse("error", {"type": "error",
                                    "error": {"type": "api_error",
                                              "message": web_error(exc)}})

        return StreamingResponse(gen_web(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })

    headers = {"Authorization": f"Bearer {UPSTREAM_KEY}", "Content-Type": "application/json",
               "Accept": "text/event-stream" if want_stream else "application/json"}

    if not want_stream:
        try:
            r = await http().post(f"{UPSTREAM_BASE}/chat/completions", json=oai, headers=headers)
        except httpx.HTTPError as exc:
            return err(f"upstream connection failed: {exc}")
        if r.status_code >= 400:
            log("upstream", r.status_code, r.text[:400])
            return err(f"upstream {r.status_code}: {r.text[:800]}", "api_error", r.status_code)
        data = r.json()
        dump("response", data)
        return JSONResponse(content=openai_to_anthropic_shim(data, model, tool_names) if shim_on
                            else openai_to_anthropic(data, model))

    oai["stream"] = True
    oai["stream_options"] = {"include_usage": True}

    upstream = http().stream("POST", f"{UPSTREAM_BASE}/chat/completions",
                             json=oai, headers=headers, timeout=TIMEOUT)
    try:
        resp = await upstream.__aenter__()
    except httpx.HTTPError as exc:
        return err(f"upstream connection failed: {exc}")

    if resp.status_code >= 400:
        raw = (await resp.aread()).decode("utf-8", "replace")
        await upstream.__aexit__(None, None, None)
        log("upstream", resp.status_code, raw[:400])
        return err(f"upstream {resp.status_code}: {raw[:800]}", "api_error", resp.status_code)

    async def gen() -> AsyncIterator[bytes]:
        try:
            async for chunk in stream_anthropic(resp, model, shim=shim_on,
                                                names=tool_names):
                yield chunk
        except httpx.HTTPError as exc:
            log("stream aborted:", exc)
            yield sse("error", {"type": "error", "error": {"type": "api_error", "message": str(exc)}})
        finally:
            await upstream.__aexit__(None, None, None)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(path: str, request: Request):
    log("unhandled", request.method, path)
    return err(f"zai_proxy does not implement /{path}", "not_found_error", 404)


if __name__ == "__main__":
    if not UPSTREAM_KEY:
        log("!! ZAI_API_KEY is empty -- every request will fail with 500")
    log(f"upstream   : {UPSTREAM_BASE}")
    log(f"default    : {DEFAULT_MODEL}   small: {SMALL_MODEL}")
    log(f"listening  : http://{LISTEN_HOST}:{LISTEN_PORT}")
    log(f"dump       : {'on -> ' + DUMP_DIR if DUMP else 'off'}")
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT, log_level="warning", access_log=False)
