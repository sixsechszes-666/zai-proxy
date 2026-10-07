"""
toolshim -- tool calling over an endpoint that has no tool calling.

Some backends take a chat completion and nothing else. z.ai's web endpoint is
one: a client-supplied `tools` array comes back INTERNAL_ERROR, and the app
itself never sends one. Claude Code, meanwhile, is almost entirely tool calls,
so a plain chat pipe is useless to it.

This fixes that without touching the backend. The tool schemas are rendered
into the prompt as text, the model is taught a tiny envelope for calling one,
and the reply text is parsed back into Anthropic `tool_use` blocks on the way
out. The whole tool loop stays plain text on the wire.

    Claude Code                 proxy.py                  upstream
      tools[]     ----------->  render into prompt  ---->  plain chat
      tool_use    <-----------  parse text stream    <----  <tool_call>{...}

The envelope:

    <tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>

Chosen because it is rare in prose, unambiguous to split on, and survives
being streamed token by token. `ToolStreamParser` is incremental: it holds
back any trailing text that could still turn out to be the start of a marker,
so a marker split across two deltas is not leaked to the client as text.

The wrapper is the part models drop. GLM produced this, unprompted, on the
first live run:

    I'll list the files in the current directory using the Bash tool.

    {"name": "Bash", "arguments": {"command": "ls -la"}}

which the strict parser read as an answer, so the tool loop died quietly -- the
worst failure mode available, since nothing errors. So the bare object is
accepted too, under two guards: the object must be a well formed JSON object
carrying a `name` key, and that name must be one of the tools actually offered
this turn. A stray `{"name": "config", "arguments": ...}` in a code block stays
text, and so does the same object with no allowlist to vouch for it.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Iterable

OPEN = "<tool_call>"
CLOSE = "</tool_call>"

# The start of an object whose first key is a quoted string: `{"name"`, `{ "x"`.
# Plain prose braces (`{NAME}`) do not match, so they are never held back.
CALL_START = re.compile(r'\{\s*"')

# A malformed reply must not grow the buffer forever. Past this the parser
# gives up on the block and releases what it has as ordinary text.
MAX_BLOCK = 262144


# --------------------------------------------------------------------------
# json salvage
# --------------------------------------------------------------------------

def coerce_json(raw: Any) -> Any:
    """Best effort at an object out of whatever the model produced."""
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # an unterminated object from a truncated stream
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
    return {}


# --------------------------------------------------------------------------
# rendering: Anthropic -> prompt text
# --------------------------------------------------------------------------

def _schema_lines(schema: dict) -> str:
    return json.dumps(schema or {"type": "object", "properties": {}}, ensure_ascii=False)


def render_tools(tools: Iterable[dict]) -> str:
    """The system-prompt block that teaches the envelope and lists the tools."""
    seen: dict[str, dict] = {}
    for t in tools or []:
        if isinstance(t, dict) and t.get("name"):
            seen[t["name"]] = t
    if not seen:
        return ""

    parts = [
        "# Tools",
        "",
        "You can call tools. A call is one line, wrapped in the marker below:",
        "",
        f'{OPEN}{{"name": "<tool name>", "arguments": {{<json>}}}}{CLOSE}',
        "",
        "To list the files in the current directory with a `Bash` tool, the whole",
        "reply is exactly this:",
        "",
        f'{OPEN}{{"name": "Bash", "arguments": {{"command": "ls -la"}}}}{CLOSE}',
        "",
        "The marker is the whole difference between calling a tool and talking",
        "about calling one. Emitting the JSON object on its own, without the",
        f"{OPEN} and {CLOSE} around it, is read as prose that nobody executes.",
        "",
        "Rules:",
        "- `arguments` is one valid JSON object matching that tool's schema. Use `{}` when it takes none.",
        "- Several calls in one reply run in parallel. Emit several blocks, one per call.",
        "- Prose you write alongside a call is shown to the user, so keep it short.",
        "- After a call, stop and wait. Results come back as <tool_result> blocks.",
        "- Never invent a tool name and never call one that is not listed here.",
        f"- When no tool is needed, answer in plain text with no {OPEN} block.",
        f"- Never write {OPEN} except to make a call. To quote the format, describe it in words.",
        "",
        "## Available tools",
        "",
    ]
    for name, t in seen.items():
        desc = (t.get("description") or "").strip()
        parts.append(f"### {name}")
        if desc:
            parts.append(desc)
        parts.append(f"Input schema: {_schema_lines(t.get('input_schema'))}")
        parts.append("")
    return "\n".join(parts)


def render_call(name: str, args: Any) -> str:
    """A tool call as the assistant would have written it."""
    payload = json.dumps({"name": name, "arguments": args if isinstance(args, dict) else {}},
                         ensure_ascii=False)
    return f"{OPEN}{payload}{CLOSE}"


def render_result(name: str, call_id: str, content: str) -> str:
    """A tool result, wrapped so the model can tell it from user prose."""
    body = content if content else "(no output)"
    return (f'<tool_result tool="{name}" id="{call_id}">\n'
            f"{body}\n"
            f"</tool_result>")


# --------------------------------------------------------------------------
# parsing: prompt text -> Anthropic tool_use
# --------------------------------------------------------------------------

def parse_call(raw: str, names: Iterable[str] | None = None,
               strict: bool = False) -> dict | None:
    """One envelope's innards into {"name","arguments"}.

    `strict` is the guard for the unmarked form: there no marker vouched for
    the intent, so the object has to name a tool that was actually offered.
    An envelope that got the name wrong still becomes a call, because the
    marker already said the model meant to call something, and a wrong name is
    worth surfacing to the client rather than silently demoting to prose.
    """
    text = str(raw).strip()
    # Cut to the outermost object. A doubled marker, or prose the model put
    # inside the envelope, would otherwise lose an otherwise good call.
    k = text.find("{")
    if k >= 0:
        text = text[k:]
        if "}" in text:
            text = text[:text.rfind("}") + 1]
    obj = coerce_json(text)
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if strict:
        allowed = {n for n in (names or []) if n}
        if name not in allowed:
            return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters") or obj.get("input") or obj.get("args") or {}
    if not isinstance(args, dict):
        args = coerce_json(args)
    return {"name": name, "arguments": args if isinstance(args, dict) else {}}


def _first_call_start(buf: str) -> int:
    """Index of the first `{"` in `buf`, or -1."""
    m = CALL_START.search(buf)
    return m.start() if m else -1


def _object_end(buf: str, start: int) -> int:
    """Index just past the object opening at `start`, or -1 if it never closes.

    String aware, because a `}` inside an argument value is not the end of
    anything. A string left open at the end of the buffer is the normal state
    mid-stream, and returns -1, which is what makes the caller hold it.
    """
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(buf)):
        c = buf[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _partial_tail(buf: str, marker: str) -> int:
    """Length of the longest suffix of `buf` that could still become `marker`."""
    for k in range(min(len(marker) - 1, len(buf)), 0, -1):
        if buf.endswith(marker[:k]):
            return k
    return 0


def _partial_call_start(buf: str) -> int:
    """Length of the longest suffix that could still become the start of `{"`.

    A stream can cut between the brace and the quote, and without this the
    brace goes out as text and the call it was opening is lost.
    """
    i = len(buf)
    while i > 0 and buf[i - 1] in " \t\r\n":
        i -= 1
    if i > 0 and buf[i - 1] == "{":
        return len(buf) - (i - 1)
    return 0


class ToolStreamParser:
    """Incremental splitter. Feed it text, get back ordered events.

    Events are ("text", str) or ("call", {"name","arguments"}). Text that
    might be the prefix of a marker is held until the next feed decides it,
    and the same goes for an object that might still turn out to be a call.

    `names` is the set of tools offered this turn. Without it the unmarked
    form is not accepted at all, since nothing else distinguishes a call from
    an object the model happened to write.
    """

    def __init__(self, names: Iterable[str] | None = None) -> None:
        self.buf = ""
        self.inside = False
        self.names = {n for n in (names or []) if n}

    def feed(self, text: str) -> list[tuple[str, Any]]:
        if not text:
            return []
        self.buf += text
        out: list[tuple[str, Any]] = []

        while True:
            if self.inside:
                j = self.buf.find(CLOSE)
                if j >= 0:
                    raw, self.buf = self.buf[:j], self.buf[j + len(CLOSE):]
                    self.inside = False
                    call = parse_call(raw)
                    if call:
                        out.append(("call", call))
                    continue
                if len(self.buf) > MAX_BLOCK:
                    # runaway block: give up and release it as text
                    out.append(("text", OPEN + self.buf))
                    self.buf, self.inside = "", False
                    break
                break  # need more input before the block closes

            i = self.buf.find(OPEN)
            if i >= 0:
                if i:
                    out.append(("text", self.buf[:i]))
                self.buf = self.buf[i + len(OPEN):]
                self.inside = True
                continue

            # no marker. an object that opens with a quoted key might be an
            # unwrapped call, so it is held until it closes or the stream ends
            m = _first_call_start(self.buf)
            if m >= 0:
                end = _object_end(self.buf, m)
                if end < 0:
                    if len(self.buf) - m > MAX_BLOCK:
                        out.append(("text", self.buf[:m + 1]))
                        self.buf = self.buf[m + 1:]
                        continue
                    if m:
                        out.append(("text", self.buf[:m]))
                    self.buf = self.buf[m:]
                    break
                call = parse_call(self.buf[m:end], self.names, strict=True)
                if call:
                    if m:
                        out.append(("text", self.buf[:m]))
                    out.append(("call", call))
                else:
                    # just an object in the reply: release it and read on, in
                    # case a real call sits further along
                    out.append(("text", self.buf[:end]))
                self.buf = self.buf[end:]
                continue

            keep = max(_partial_tail(self.buf, OPEN), _partial_call_start(self.buf))
            head = self.buf[:len(self.buf) - keep] if keep else self.buf
            if head:
                out.append(("text", head))
            self.buf = self.buf[len(self.buf) - keep:] if keep else ""
            break

        return out

    def finish(self) -> list[tuple[str, Any]]:
        """Drain at end of stream. A truncated block is still worth a try."""
        out: list[tuple[str, Any]] = []
        if not self.buf:
            return out
        if self.inside:
            call = parse_call(self.buf)
            if call:
                out.append(("call", call))
            else:
                out.append(("text", OPEN + self.buf))
        else:
            m = _first_call_start(self.buf)
            call = parse_call(self.buf[m:], self.names, strict=True) if m >= 0 else None
            if call:
                if m:
                    out.append(("text", self.buf[:m]))
                out.append(("call", call))
            else:
                out.append(("text", self.buf))
        self.buf, self.inside = "", False
        return out


def parse_full(text: str, names: Iterable[str] | None = None) -> list[tuple[str, Any]]:
    """Non-streaming convenience: the whole reply at once."""
    p = ToolStreamParser(names)
    return p.feed(text) + p.finish()


def new_tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:20]}"
