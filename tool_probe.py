"""
tool_probe.py -- does /api/v2/chat/completions accept client-declared tools
under any encoding at all?

`WEB_MODE.md` records the headline result: a populated `tools` array comes back
INTERNAL_ERROR on every model, an empty one is accepted. That leaves the
question of whether some *other* encoding is the one the backend actually
wants. This walks the plausible ones, each against the app's own captured body
so nothing else varies between runs.

The baseline body is the one the fetch hook caught the app sending. Get it by
installing the hook and sending a message from the UI (see WEB_MODE.md).

  python tool_probe.py              # every variant
  python tool_probe.py --only 3     # just one
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid

import webclient as W
import websig

MIN_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Ping",
        "parameters": {"type": "object", "properties": {}},
    },
}

# Claude Code's shape, so a success here would be directly usable.
REAL_TOOL = W.TOOL


async def capture_base() -> dict:
    """The last request the app itself sent, tools-free."""
    raw = await W._eval_tab(
        "(() => {"
        " const h = window.__zai_hook || [];"
        " for (let i = h.length - 1; i >= 0; i--) {"
        "   const b = h[i].body || '';"
        "   if (b.indexOf('\"tools\"') === -1 && b.indexOf('signature_prompt') !== -1)"
        "     return b;"
        " }"
        " return null;"
        "})()")
    if not raw:
        raise SystemExit("no captured app request in window.__zai_hook")
    return json.loads(raw)


def variants(base: dict):
    """(label, mutate) pairs. Each mutate gets a fresh copy of the base body."""
    def v_tools(b):
        b["tools"] = [MIN_TOOL]

    def v_vlm_flag(b):
        b["tools"] = [MIN_TOOL]
        b["features"]["vlm_tools_enable"] = True

    def v_vlm_all(b):
        b["tools"] = [MIN_TOOL]
        b["features"].update({"vlm_tools_enable": True,
                              "vlm_web_search_enable": True,
                              "vlm_website_mode": True})

    def v_real_tool(b):
        b["tools"] = [REAL_TOOL]

    def v_dict(b):
        b["tools"] = {"ping": MIN_TOOL["function"]}

    def v_choice(b):
        b["tools"] = [MIN_TOOL]
        b["tool_choice"] = "auto"

    def v_functions(b):
        # the older OpenAI name for the same thing
        b.pop("tools", None)
        b["functions"] = [MIN_TOOL["function"]]

    return [
        ("tools[] minimal", v_tools),
        ("tools[] + vlm_tools_enable", v_vlm_flag),
        ("tools[] + all vlm flags", v_vlm_all),
        ("tools[] Claude-Code-shaped", v_real_tool),
        ("tools as dict", v_dict),
        ("tools[] + tool_choice=auto", v_choice),
        ("functions[] legacy", v_functions),
    ]


async def run_one(s: W.WebSession, base: dict, label: str, mutate) -> str:
    body = json.loads(json.dumps(base))
    body.pop("captcha_verify_param", None)
    mutate(body)

    prompt = body.get("signature_prompt", "")
    rid = str(uuid.uuid4())
    ts = int(time.time() * 1000)
    url = f"{W.WEB}/api/v2/chat/completions?{s._query(rid, ts)}"
    body["captcha_verify_param"] = await s.captcha()

    res = await s.post_in_page(url, s._headers(websig.sign(prompt, ts, rid, s.user_id)), body)
    fr = W.frames(res.get("body") or "")

    err = next((W.dig(f).get("error") for f in fr
                if isinstance(W.dig(f), dict) and W.dig(f).get("error")), None)
    if err:
        return f"{label:32} -> {err.get('code')}  {str(err.get('detail'))[:70]}"

    text, tc, _ = W.harvest(fr)
    seen = " ".join(sorted({str(W.dig(f).get("phase")) for f in fr
                            if isinstance(W.dig(f), dict)}))
    if not text and not tc:
        raw = (res.get("body") or "")[:400].replace("\n", " ")
        return f"{label:32} -> ?  {len(fr)} frame(s)  raw={raw!r}"
    return (f"{label:32} -> OK  phases=[{seen}]  "
            f"text={str(text.get('content'))[:50]!r}  tool_calls={tc}")


async def main() -> None:
    s = W.WebSession()
    await s.arefresh()
    print(f"token {len(s.token)} chars, device={s.device_id or '(none)'}")
    base = await capture_base()
    print(f"baseline: model={base.get('model')}, "
          f"{len(base)} fields, prompt={base.get('signature_prompt')!r}\n")

    only = None
    if "--only" in sys.argv:
        only = int(sys.argv[sys.argv.index("--only") + 1])

    for i, (label, mutate) in enumerate(variants(base), 1):
        if only is not None and i != only:
            continue
        try:
            print(await run_one(s, base, label, mutate))
        except SystemExit as exc:
            print(f"{label:32} -> aborted: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
