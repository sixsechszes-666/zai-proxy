"""
probe_web.py -- recon against the chat.z.ai web backend, using the JWT that the
logged-in (or guest) browser holds in localStorage.

The app's own request log says `prod-open-webui-frontend`, and its API surface
(`/api/models`, `/api/v1/auths/`, `/api/v1/chats/`) is the Open WebUI one. If
that holds, the completion endpoint is OpenAI-shaped and the existing
anthropic_to_openai converter applies unchanged -- only the base URL and the
auth header differ from the z.ai platform API.

  python probe_web.py                 # who am I, what models are offered
  python probe_web.py --tools         # does it accept tools and return tool_calls?
  python probe_web.py --stream        # what does the SSE actually look like
  python probe_web.py --chat          # non-streaming completion

Nothing here writes to His account: /api/models and /api/v1/auths are reads,
and the completions call is one short throwaway message.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# Capture the outbound proxy before clearing it. Two different routes are in
# play: chat.z.ai is unreachable directly from here (TLS dies with
# WRONG_VERSION_NUMBER, which is DPI, not a cert problem) and must go through
# Clash, while 127.0.0.1:9222 must never go near it.
_OUT_PROXY = (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
              or os.environ.get("ALL_PROXY") or os.environ.get("all_proxy") or "")

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_v, None)
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost,::1"

import httpx          # noqa: E402
import websockets     # noqa: E402

CDP = f"http://127.0.0.1:{os.environ.get('ZAI_CDP_PORT', '9222')}"
WEB = "https://chat.z.ai"
TAB_MATCH = "chat.z.ai"


async def grab_token() -> str:
    with httpx.Client(timeout=10, trust_env=False) as c:
        targets = c.get(f"{CDP}/json/list").json()
    page = next((t for t in targets
                 if t.get("type") == "page" and TAB_MATCH in t.get("url", "")), None)
    if page is None:
        raise SystemExit(f"no {TAB_MATCH} tab open. run: python cdp.py open https://chat.z.ai/")

    async with websockets.connect(page["webSocketDebuggerUrl"],
                                  max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {
            "expression": 'localStorage.getItem("token")||""', "returnByValue": True}}))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("id") == 1:
                token = (msg.get("result", {}).get("result", {}) or {}).get("value") or ""
                if not token:
                    raise SystemExit("no token in localStorage - the tab is not signed in")
                return token


def claims(token: str) -> dict:
    import base64
    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        return json.loads(base64.urlsafe_b64decode(part))
    except Exception as exc:
        return {"_decode_error": str(exc)}


def client() -> httpx.Client:
    kw: dict = {"timeout": 300, "follow_redirects": True, "trust_env": False}
    if _OUT_PROXY:
        kw["proxy"] = _OUT_PROXY
    return httpx.Client(**kw)


def headers(token: str, stream: bool = False) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "Origin": WEB,
        "Referer": f"{WEB}/",
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"),
    }


TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["metric", "imperial"]},
            },
            "required": ["city"],
        },
    },
}


def show(label: str, obj, limit: int = 2500) -> None:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2)
    if len(text) > limit:
        text = text[:limit] + f"\n... [{len(text) - limit} more]"
    print(f"{label}: {text}")


def main() -> None:
    token = asyncio.run(grab_token())
    print(f"token: {len(token)} chars, claims={json.dumps(claims(token), ensure_ascii=False)}")
    print(f"prefix: {token[:16]}...\n")

    with client() as c:
        r = c.get(f"{WEB}/api/v1/auths/", headers=headers(token))
        print(f"GET /api/v1/auths/ -> {r.status_code}")
        if r.status_code == 200:
            me = r.json()
            show("  me", {k: me.get(k) for k in
                          ("id", "email", "name", "role", "profile_image_url") if k in me})
        else:
            show("  body", r.text[:400])

        r = c.get(f"{WEB}/api/models", headers=headers(token))
        print(f"\nGET /api/models -> {r.status_code}")
        if r.status_code == 200:
            data = r.json().get("data") or r.json()
            ids = [m.get("id") for m in data] if isinstance(data, list) else data
            print(f"  {len(ids)} models")
            for mid in ids:
                mark = "  <<<" if "5.3" in str(mid) or "flash" in str(mid).lower() else ""
                print(f"    {mid}{mark}")
        else:
            show("  body", r.text[:400])

        if "--tools" in sys.argv or "--chat" in sys.argv or "--stream" in sys.argv:
            # The web app does NOT offer glm-5.3-flash. Its id for the same
            # family is plain `glm-5.3`; -flash is the platform-API spelling.
            model = "glm-5.3"
            if "--model" in sys.argv:
                model = sys.argv[sys.argv.index("--model") + 1]
            body = {
                "model": model,
                "messages": [{"role": "user",
                              "content": "What is the weather in Tokyo? Use the tool."}],
                "stream": "--stream" in sys.argv,
            }
            if "--tools" in sys.argv:
                body["tools"] = [TOOL]
                body["tool_choice"] = "auto"
                body["messages"][0]["content"] = (
                    "Use the get_weather tool to answer. Do not answer from memory.")

            print(f"\nPOST /api/chat/completions  model={body['model']} "
                  f"stream={body['stream']} tools={'tools' in body}")
            if body["stream"]:
                got_any = False
                with c.stream("POST", f"{WEB}/api/chat/completions",
                              json=body, headers=headers(token, stream=True)) as r:
                    print(f"  status {r.status_code}")
                    if r.status_code >= 400:
                        print("  body:", r.read().decode("utf-8", "replace")[:800])
                        return
                    for line in r.iter_lines():
                        if not line:
                            continue
                        if not got_any:
                            print("  --- first 25 lines of the raw stream ---")
                            got_any = True
                        if line.startswith("data: "):
                            print("  " + line[:220])
                        else:
                            print("  RAW " + line[:220])
                        if line.strip() == "data: [DONE]":
                            break
            else:
                r = c.post(f"{WEB}/api/chat/completions", json=body, headers=headers(token))
                print(f"  status {r.status_code}")
                if r.status_code >= 400:
                    show("  body", r.text[:800])
                    return
                j = r.json()
                show("  response", j, 3000)
                msg = ((j.get("choices") or [{}])[0].get("message") or {})
                print(f"\n  content    : {str(msg.get('content'))[:200]!r}")
                print(f"  tool_calls : {json.dumps(msg.get('tool_calls'), ensure_ascii=False)[:400]}")

    print("\nkeep the token out of scrollback: it is a live credential.")


if __name__ == "__main__":
    main()
