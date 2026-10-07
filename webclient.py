"""
webclient.py -- talk to the chat.z.ai web backend as the browser session.

This is the "browser session as a token dispenser" path: no platform API key,
just the JWT that the logged-in tab already holds in localStorage, plus a
request signature that `websig.py` reconstructs from the frontend bundle.

The web app is Open WebUI underneath, but the completion call is NOT the
OpenAI shape -- it is `/api/v2/chat/completions` with a body that carries the
signature prompt, a browser fingerprint and a captcha token. Only the last
part is a real obstacle; see `WEB_MODE.md`.

Two things were learned the hard way and shape everything here:

  * the endpoint reads `signature_prompt` and nothing else. `messages` is
    ignored and the stored chat never becomes context, so `flatten()` writes
    the whole transcript into the prompt by hand.
  * `chat_id` has to name a chat the server already knows. A fresh uuid dies
    with INTERNAL_ERROR, so `chat()` opens one through `/api/v1/chats/new`
    first, exactly as the app does.

  python webclient.py --whoami
  python webclient.py --chat "say hi"
  python webclient.py --chat "say hi" --inpage    # buffered, frames readable
  python webclient.py --new-chat
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
import uuid

# chat.z.ai is unreachable straight from this machine (TLS gets reset by DPI),
# so the outbound proxy is captured before the loopback vars are stripped.
_OUT_PROXY = (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
              or os.environ.get("ALL_PROXY") or os.environ.get("all_proxy") or "")

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_v, None)
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost,::1"

import httpx          # noqa: E402
import websockets     # noqa: E402

import websig         # noqa: E402

CDP = f"http://127.0.0.1:{os.environ.get('ZAI_CDP_PORT', '9222')}"
WEB = "https://chat.z.ai"
TAB = "chat.z.ai"
FE_VERSION = os.environ.get("ZAI_FE_VERSION", "prod-fe-1.1.96")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36")

# Every request the app makes goes through `wy(headers)`, which reads the
# device id out of localStorage and sets X-Device-ID. It is not the `device_id`
# key -- it is `_arms_uid`, or failing that the first value anywhere in
# localStorage that matches /^uid_[A-Za-z0-9]{7,}$/. The completion request
# carries it too, so a request without it is not the request the app makes.
_DEVICE_JS = r"""
JSON.stringify((() => {
  const RX = /^uid_[A-Za-z0-9]{7,}$/;
  const tok = localStorage.getItem("token") || "";
  try {
    const direct = localStorage.getItem("_arms_uid");
    if (direct && RX.test(direct)) return { token: tok, device: direct };
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (!k) continue;
      const v = localStorage.getItem(k);
      if (v && RX.test(v)) return { token: tok, device: v };
    }
  } catch (e) {}
  return { token: tok, device: "" };
})())
"""


def _jwt_claims(token: str) -> dict:
    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        return json.loads(base64.urlsafe_b64decode(part))
    except Exception:
        return {}


async def _page_ws_url() -> str:
    async with httpx.AsyncClient(timeout=10, trust_env=False) as c:
        targets = (await c.get(f"{CDP}/json/list")).json()
    page = next((t for t in targets
                 if t.get("type") == "page" and TAB in t.get("url", "")), None)
    if page is None:
        raise SystemExit(f"no {TAB} tab. run: python cdp.py open https://chat.z.ai/")
    return page["webSocketDebuggerUrl"]


async def _eval_tab(js: str) -> object:
    """Run one expression in the tab over a connection that lives for the call."""
    async with websockets.connect(await _page_ws_url(),
                                  max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {
            "expression": js, "returnByValue": True, "awaitPromise": True}}))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("id") == 1:
                res = msg.get("result", {})
                if "exceptionDetails" in res:
                    raise SystemExit("page threw: " +
                                     json.dumps(res["exceptionDetails"])[:300])
                return (res.get("result") or {}).get("value")


def _eval_in_tab(js: str) -> object:
    """Sync view of the same thing, for the CLI."""
    return asyncio.run(_eval_tab(js))


# Scene config, lifted from the bundle (`zh`): sgp region, no8xfe prefix, and
# the scene id swaps on hostname, so chat.z.ai gets didk33e0.
#
# This mirrors `BN()` from the bundle, which is the app's real path to a param:
#
#   BN = async () => {
#     i2 = null; tE = false; Fh = null;          // drop the cached init
#     await ohe();                               // re-init, no destroyCaptcha
#     return new Promise((res, rej) => { J2 = {resolve: res, reject: rej}; OU(); });
#   }
#
# and `OU()` is nothing but `document.getElementById("chat-captcha-trigger")
# .click()`. The app never calls startTracelessVerification itself -- the click
# is the whole trigger, and the SDK decides internally whether the check can be
# silent. Notably it never destroys the old instance either; it just re-inits,
# which rebinds the button. Both of those are reproduced here.
CAPTCHA_JS = r"""
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const EL = "chat-captcha-element", BTN = "chat-captcha-trigger";
  try {
    if (!window.initAliyunCaptcha) {
      await new Promise((res, rej) => {
        const s = document.createElement("script");
        s.src = "https://o.alicdn.com/captcha-frontend/aliyunCaptcha/AliyunCaptcha.js";
        s.onload = res;
        s.onerror = () => rej(new Error("captcha sdk failed to load"));
        document.head.appendChild(s);
      });
    }
    window.AliyunCaptchaConfig = { region: "sgp", prefix: "no8xfe" };

    // The app's own offscreen pair, with the app's own styles. Both already
    // exist in a live tab; reusing them hands the SDK exactly what it is
    // handed when the app asks for a captcha itself.
    const mk = (id, tag, css, attrs) => {
      if (document.getElementById(id)) return;
      const e = document.createElement(tag);
      e.id = id;
      e.style.cssText = css;
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      document.body.appendChild(e);
    };
    mk(EL, "div",
       "position:absolute;left:-99999px;top:-99999px;width:0;height:0;overflow:hidden;pointer-events:none;",
       {});
    mk(BTN, "button",
       "position:absolute;left:-99999px;top:-99999px;width:1px;height:1px;opacity:0;",
       {"type": "button", "aria-hidden": "true", "tabindex": "-1"});

    window.__zcap = null; window.__zfail = null; window.__zerr = null;
    window.__zinst = null; window.__zres = null;

    await new Promise(resolve => {
      window.initAliyunCaptcha({
        SceneId: "didk33e0", mode: "popup",
        element: "#" + EL, button: "#" + BTN,
        captchaLogoImg: "https://z-cdn.chatglm.cn/z-ai/static/logo.svg",
        language: "en", timeout: 10000, delayBeforeSuccess: false,
        success: p => { window.__zcap = p; if (window.__zres) window.__zres(); },
        // the app's fail handler is OU() -- click the trigger again -- so a
        // fail is a retry, not a terminal state
        fail: e => { window.__zfail = String(e); },
        onError: e => { window.__zerr = "error:" + String(e); if (window.__zres) window.__zres(); },
        onClose: () => { window.__zerr = "close"; if (window.__zres) window.__zres(); },
        getInstance: i => { window.__zinst = i; resolve(); },
      });
    });
    await sleep(300);

    for (let round = 0; round < 4 && !window.__zcap && !window.__zerr; round++) {
      await new Promise(res => {
        window.__zres = res;
        document.getElementById(BTN).click();
        setTimeout(res, 4000);
      });
    }

    // If the click path produced nothing at all, fall back to asking the SDK
    // directly -- same verification, just without the button in the way.
    if (!window.__zcap && !window.__zerr && window.__zinst
        && window.__zinst.startTracelessVerification) {
      try { window.__zinst.startTracelessVerification(); } catch (e) {}
      for (let k = 0; k < 40; k++) {
        await sleep(250);
        if (window.__zcap || window.__zerr) break;
      }
    }

    let err = window.__zerr;
    if (!err && !window.__zcap) err = "fail:" + window.__zfail;
    return JSON.stringify({ cap: window.__zcap || null, err: err });
  } catch (e) {
    return JSON.stringify({ cap: null, err: String(e) });
  }
})()
"""


def _variables(user_name: str = "User") -> dict:
    """The `variables` block the app sends alongside every message.

    Captured off a live send: the server fills these {{...}} placeholders into
    the system prompt server-side, and it wants the block present and shaped
    exactly like this. Values come from the browser in the app; here they are
    computed locally, which is the same thing for every field that is not the
    user's own name.
    """
    import datetime
    now = datetime.datetime.now()
    return {
        "{{USER_NAME}}": user_name,
        "{{USER_LOCATION}}": "Unknown",
        "{{CURRENT_DATETIME}}": now.strftime("%Y-%m-%d %H:%M:%S"),
        "{{CURRENT_DATE}}": now.strftime("%Y-%m-%d"),
        "{{CURRENT_TIME}}": now.strftime("%H:%M:%S"),
        "{{CURRENT_WEEKDAY}}": now.strftime("%A"),
        "{{CURRENT_TIMEZONE}}": _tz_name(),
        "{{USER_LANGUAGE}}": "en-US",
    }


def _tz_name() -> str:
    import datetime
    try:
        return datetime.datetime.now().astimezone().tzinfo.tzname(None) or "UTC"
    except Exception:
        return "UTC"


class WebSession:
    """Holds the browser-derived credentials and signs requests.

    Async-first, because the proxy drives this from inside a running event
    loop where `asyncio.run` is not available. The CLI wraps each call.
    """
    def __init__(self) -> None:
        self.token = ""
        self.user_id = ""
        self.device_id = ""
        self.chat_id = ""
        self._tab = None

    def refresh(self) -> None:
        asyncio.run(self.arefresh())

    async def arefresh(self) -> None:
        data = json.loads(await _eval_tab(_DEVICE_JS) or "{}")
        self.token = data.get("token", "")
        if not self.token:
            raise SystemExit("no token in localStorage - the tab is not signed in")
        self.user_id = _jwt_claims(self.token).get("id", "")
        self.device_id = data.get("device", "")

    def expires_in(self) -> float:
        """Seconds until the JWT expires. Huge when it carries no exp."""
        exp = _jwt_claims(self.token).get("exp")
        return float(exp) - time.time() if exp else 1e9

    async def captcha(self) -> str:
        """A fresh captcha_verify_param, minted by the page with no user input.

        The Aliyun scene on chat.z.ai is set up so the SDK exposes
        `startTracelessVerification`, which runs the risk check silently and
        hands back a param through the `success` callback. The app itself only
        ever invokes it by clicking an off-screen button, so nothing about this
        needs a human. The param is single-use, so it is fetched per request.
        """
        raw = await _eval_tab(CAPTCHA_JS)
        data = json.loads(raw or "{}")
        if not data.get("cap"):
            raise SystemExit(f"captcha failed: {data.get('err') or data or 'no param'}")
        return data["cap"]

    # -- chats -------------------------------------------------------------

    async def create_chat(self, model: str = "x-preview-l",
                          title: str = "New Chat") -> str:
        """Open a chat server-side and return its id.

        The completion endpoint will not accept a chat_id it has never seen.
        A client-invented uuid passes the captcha, the signature and every
        other field, then dies on the second frame with INTERNAL_ERROR; the
        only field that matters is this one. The app does not invent one
        either -- it POSTs here first with an empty id and the server assigns
        the real one. Same-origin fetch is enough, no signature needed.
        """
        payload = {"chat": {
            "id": "", "title": title, "models": [model], "params": {},
            "history": {"messages": {}, "currentId": ""},
            "tags": [], "flags": [], "features": [], "mcp_servers": [],
            "enable_thinking": True, "reasoning_effort": "max",
            "auto_web_search": False, "message_version": 1, "extra": {},
        }}
        js = ("(async () => {"
              "try {"
              " const r = await fetch('/api/v1/chats/new', {method:'POST',"
              " headers:{'Content-Type':'application/json'},"
              " credentials:'include',"
              f" body: {json.dumps(json.dumps(payload))}}});"
              " const t = await r.text();"
              " return JSON.stringify({status: r.status, body: t});"
              "} catch (e) { return JSON.stringify({status: 0, body: String(e)}); }"
              "})()")
        data = json.loads(await _eval_tab(js) or "{}")
        if data.get("status") != 200:
            raise SystemExit(f"chats/new -> {data.get('status')}: "
                             f"{str(data.get('body'))[:300]}")
        try:
            cid = (json.loads(data["body"]) or {}).get("id", "")
        except Exception:
            cid = ""
        if not cid:
            raise SystemExit(f"chats/new returned no id: {str(data.get('body'))[:300]}")
        return cid

    async def chat(self, model: str = "x-preview-l", *, fresh: bool = False) -> str:
        """The chat this session writes into, created on first use.

        One chat is reused for the whole session. That is safe because the
        endpoint turns out to be stateless: it reads the prompt and nothing
        else, so the stored chat never becomes context (see `flatten`).
        """
        if fresh or not self.chat_id:
            self.chat_id = await self.create_chat(model)
        return self.chat_id

    # -- transport ---------------------------------------------------------

    def _client(self) -> httpx.Client:
        kw: dict = {"timeout": 300, "follow_redirects": True, "trust_env": False}
        if _OUT_PROXY:
            kw["proxy"] = _OUT_PROXY
        return httpx.Client(**kw)

    def _query(self, request_id: str, ts: int) -> str:
        return "&".join([
            f"timestamp={ts}",
            f"requestId={request_id}",
            f"user_id={self.user_id}",
            "version=0.0.1",
            "platform=web",
            f"token={self.token}",
            f"language=en-US",
            f"signature_timestamp={ts}",
        ])

    def _headers(self, signature: str) -> dict:
        h = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept-Language": "en-US",
            "X-FE-Version": FE_VERSION,
            "X-Signature": signature,
            "User-Agent": UA,
            "Origin": WEB,
            "Referer": f"{WEB}/",
        }
        if self.device_id:
            h["X-Device-ID"] = self.device_id
        return h

    async def build(self, prompt: str, *, model: str = "x-preview-l",
                    stream: bool = True, messages: list | None = None,
                    tools: list | None = None, params: dict | None = None,
                    captcha: str = "", chat_id: str = "",
                    effort: str = "max") -> tuple[str, dict, dict]:
        """Return (url, headers, body) for one completion call."""
        request_id = str(uuid.uuid4())
        ts = int(__import__("time").time() * 1000)
        signature = websig.sign(prompt, ts, request_id, self.user_id)

        body = {
            "stream": stream,
            "model": model,
            "messages": messages or [{"role": "user", "content": prompt}],
            "signature_prompt": prompt,
            "params": params or {},
            "extra": {},
            "features": {
                "image_generation": False,
                "web_search": False,
                "auto_web_search": False,
                "preview_mode": True,
                "flags": [],
                "vlm_tools_enable": False,
                "vlm_web_search_enable": False,
                "vlm_website_mode": False,
                "enable_thinking": True,
                "reasoning_effort": effort,
            },
            "variables": _variables(),
            "chat_id": chat_id or await self.chat(model),
            "id": str(uuid.uuid4()),
            "current_user_message_id": str(uuid.uuid4()),
            "current_user_message_parent_id": None,
            "background_tasks": {"title_generation": True,
                                 "tags_generation": True},
        }
        if captcha:
            body["captcha_verify_param"] = captcha
        if tools:
            body["tools"] = tools

        url = f"{WEB}/api/v2/chat/completions?{self._query(request_id, ts)}"
        return url, self._headers(signature), body

    def post(self, url: str, headers: dict, body: dict):
        with self._client() as c:
            return c.post(url, headers=headers, json=body)

    async def post_in_page(self, url: str, headers: dict, body: dict,
                           limit: int = 1_000_000) -> dict:
        """Fire the same request from inside the tab, over the browser's stack.

        The app calls plain `fetch` with no `credentials` option, so the
        browser applies the default (same-origin) and sends cookies along with
        the request. This exists to tell those two things apart: if the request
        succeeds here but fails from httpx, the gap is the browser's network
        path or its cookies, not the body or the signature.

        The reply is buffered whole, so it is capped -- but generously, because
        a reasoning-heavy turn spends thousands of characters on `thinking`
        frames before the `answer` phase starts and a tight cap silently
        truncates the part that matters.
        """
        js = ("(async () => {"
              f"const u={json.dumps(url)};"
              f"const h={json.dumps(headers)};"
              f"const b={json.dumps(body)};"
              "const r = await fetch(u, {method:'POST', headers:h,"
              " body: JSON.stringify(b)});"
              "const t = await r.text();"
              "return JSON.stringify({status: r.status, statusText: r.statusText,"
              f" len: t.length, body: t.slice(0, {limit})}});"
              "})()")
        raw = await _eval_tab(js)
        return json.loads(raw or "{}")

    def stream(self, url: str, headers: dict, body: dict):
        c = self._client()
        return c, c.stream("POST", url, headers=headers, json=body)


# --------------------------------------------------------------------------
# the completion, read as a stream
#
# `post_in_page` buffers: it awaits the whole reply, which can answer "did that
# work" and nothing more. This runs the same in-page fetch and reads the body
# with a reader inside the page, draining chunks to Python over one CDP
# connection that stays open for the call. Latency is the poll interval, not
# the length of the reply, so a long agent turn arrives as it is written.

STREAM_START_JS = r"""
(() => {
  const S = window.__zs = { chunks: [], done: false, err: null, status: 0 };
  (async () => {
    try {
      const r = await fetch(window.__zs_url, { method: 'POST',
        headers: window.__zs_headers, body: window.__zs_body });
      S.status = r.status;
      if (!r.body) { S.chunks.push(await r.text()); S.done = true; return; }
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      for (;;) {
        const step = await reader.read();
        if (step.done) break;
        S.chunks.push(dec.decode(step.value, { stream: true }));
      }
    } catch (e) { S.err = String(e); }
    S.done = true;
  })();
  return 'started';
})()
"""

STREAM_DRAIN_JS = r"""
(() => {
  const S = window.__zs;
  if (!S) return JSON.stringify({ missing: true });
  const t = S.chunks.join('');
  S.chunks.length = 0;
  return JSON.stringify({ t: t, done: S.done, err: S.err, status: S.status });
})()
"""

IDLE_LIMIT = float(os.environ.get("ZAI_WEB_IDLE", "600"))

# The web endpoint serves this account one model whatever name is asked for,
# and `x-preview-l` is the one the app itself sends for GLM-5.3-Flash.
WEB_MODEL = os.environ.get("ZAI_WEB_MODEL", "x-preview-l")
WEB_EFFORT = os.environ.get("ZAI_WEB_EFFORT", "max")


class WebStream:
    """z.ai `chat:completion` frames, reshaped into OpenAI SSE lines.

    Everything above this already speaks the OpenAI delta shape, so the whole
    translation lives here and nowhere else: `phase: thinking` becomes
    reasoning_content, `phase: answer` becomes content, and the usage frame
    rides along untouched because the field names happen to match.
    """

    def __init__(self, url: str, headers: dict, body: dict,
                 poll: float = 0.08) -> None:
        self.url = url
        self.headers = headers
        self.body = body
        self.poll = poll
        self.status = 0
        self.raw = ""       # what the page sent back, for error reporting

    async def _ask(self, ws, mid: int, js: str, await_promise: bool = True) -> object:
        await ws.send(json.dumps({"id": mid, "method": "Runtime.evaluate", "params": {
            "expression": js, "returnByValue": True, "awaitPromise": await_promise}}))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("id") == mid:
                res = msg.get("result", {})
                if "exceptionDetails" in res:
                    raise RuntimeError("page threw: " +
                                       json.dumps(res["exceptionDetails"])[:300])
                return (res.get("result") or {}).get("value")

    def _line(self, line: str) -> list[str]:
        """One raw SSE line in, zero or more OpenAI-shaped lines out."""
        line = line.strip()
        if not line.startswith("data:"):
            return []
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            return []
        try:
            frame = json.loads(payload)
        except json.JSONDecodeError:
            return []
        d = dig(frame)
        if not isinstance(d, dict):
            return []
        if d.get("error"):
            raise RuntimeError("z.ai: " +
                               json.dumps(d["error"], ensure_ascii=False)[:300])
        delta: dict = {}
        if d.get("delta_content"):
            if d.get("phase") == "thinking":
                delta["reasoning_content"] = d["delta_content"]
            elif d.get("phase") == "answer":
                delta["content"] = d["delta_content"]
        chunk: dict = {"choices": [{"index": 0, "delta": delta}]}
        if d.get("usage"):
            chunk["usage"] = d["usage"]
        if d.get("done"):
            chunk["choices"][0]["finish_reason"] = "stop"
        if not delta and "usage" not in chunk and not d.get("done"):
            return []
        return ["data: " + json.dumps(chunk, ensure_ascii=False)]

    async def aiter_lines(self):
        """The stream as SSE text lines, which is what the proxy consumes."""
        ws = await websockets.connect(await _page_ws_url(),
                                      max_size=32 * 1024 * 1024)
        setup = ("window.__zs_url=" + json.dumps(self.url) + ";"
                 "window.__zs_headers=" + json.dumps(self.headers) + ";"
                 "window.__zs_body=" + json.dumps(json.dumps(self.body)) + ";")
        try:
            await self._ask(ws, 1, setup + STREAM_START_JS, await_promise=False)
            buf = ""
            idle = 0.0
            while True:
                raw = json.loads(await self._ask(ws, 2, STREAM_DRAIN_JS) or "{}")
                if raw.get("missing"):
                    raise RuntimeError("the page lost the stream buffer")
                if raw.get("err"):
                    raise RuntimeError(f"in-page fetch failed: {raw['err']}")
                self.status = raw.get("status") or self.status
                text = raw.get("t") or ""
                if text:
                    idle = 0.0
                    if len(self.raw) < 4000:
                        self.raw += text
                    buf += text
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        for out in self._line(line):
                            yield out
                if raw.get("done"):
                    for out in self._line(buf):
                        yield out
                    break
                idle += self.poll
                if idle > IDLE_LIMIT:
                    raise RuntimeError(f"no data from the page for {IDLE_LIMIT:.0f}s")
                await asyncio.sleep(self.poll)
            if self.status != 200:
                raise RuntimeError(f"chat.z.ai returned {self.status}: "
                                   f"{self.raw[:300]}")
            yield "data: [DONE]"
        finally:
            await ws.close()


def flatten(messages: list[dict]) -> str:
    """Render the transcript into the one string the web endpoint reads.

    Measured, not assumed: `usage.prompt_tokens` stays flat across turns of the
    same chat (18, 19, then 35 which is the longer prompt itself), and a fact
    planted in the `messages` array never reaches the model -- it answers as if
    the conversation had just started. The endpoint reads `signature_prompt`
    and ignores `messages`, so the conversation has to be written into the
    prompt by hand.

    The upside is that nothing accumulates server-side, so a long session
    never double-counts its own history and the stored chat stays empty.
    """
    out: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(_block_text(b) for b in content)
        text = str(content or "")
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            text += f"\n[called {fn.get('name')} with {fn.get('arguments')}]"
        label = {"system": "System", "user": "User", "assistant": "Assistant",
                 "tool": "Tool result"}.get(str(m.get("role") or "user"), "User")
        out.append(f"{label}: {text}")
    out.append("Assistant:")   # the endpoint wants a prompt, not a transcript
    return "\n\n".join(out)


def _block_text(block) -> str:
    if not isinstance(block, dict):
        return str(block)
    if block.get("type") == "image_url":
        return "[image]"
    return str(block.get("text") or block.get("content") or "")


async def web_stream(session: WebSession, oai: dict) -> WebStream:
    """An OpenAI-shaped request turned into a live z.ai web completion."""
    prompt = flatten(oai.get("messages") or [])
    url, headers, body = await session.build(
        prompt, model=WEB_MODEL, stream=True, effort=WEB_EFFORT)
    body["captcha_verify_param"] = await session.captcha()
    return WebStream(url, headers, body)


async def collect(session: WebSession, oai: dict) -> dict:
    """Assemble a non-streaming OpenAI-shaped response from the frame stream."""
    stream = await web_stream(session, oai)
    content, reasoning, usage = "", "", {}
    async for line in stream.aiter_lines():
        if line == "data: [DONE]":
            break
        chunk = json.loads(line[5:])
        delta = chunk["choices"][0].get("delta") or {}
        content += delta.get("content") or ""
        reasoning += delta.get("reasoning_content") or ""
        if chunk.get("usage"):
            usage = chunk["usage"]
    message: dict = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": usage}


# --------------------------------------------------------------------------

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


def frames(text: str) -> list[dict]:
    """The completion endpoint answers in SSE frames even when stream=False."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        blob = line[5:].strip()
        if not blob or blob == "[DONE]":
            continue
        try:
            out.append(json.loads(blob))
        except Exception:
            pass
    return out


def dig(frame: dict) -> dict:
    """`chat:completion` frames nest the payload one or two levels down."""
    d = frame.get("data")
    if isinstance(d, dict) and isinstance(d.get("data"), dict):
        return d["data"]
    return d if isinstance(d, dict) else {}


def harvest(fr: list[dict]) -> tuple[dict, object, dict | None]:
    """Walk the frames and pull out text, tool_calls and any error."""
    text, tc, err = {}, None, None
    for f in fr:
        d = dig(f)
        if isinstance(d.get("error"), dict):
            err = d["error"]
        for ch in (d.get("choices") or []):
            delta = ch.get("delta") or ch.get("message") or {}
            if delta.get("content"):
                text["content"] = (text.get("content") or "") + delta["content"]
            if delta.get("reasoning_content"):
                text["reasoning"] = (text.get("reasoning") or "") + delta["reasoning_content"]
            if delta.get("tool_calls"):
                tc = delta["tool_calls"]
    return text, tc, err


async def main() -> None:
    s = WebSession()
    await s.arefresh()
    print(f"token {len(s.token)} chars, user_id={s.user_id}, "
          f"device_id={s.device_id or '(none)'}")

    if "--whoami" in sys.argv:
        return

    prompt = "Reply with exactly: PONG"
    if "--chat" in sys.argv:
        prompt = sys.argv[sys.argv.index("--chat") + 1]

    effort = WEB_EFFORT
    if "--effort" in sys.argv:
        effort = sys.argv[sys.argv.index("--effort") + 1]

    fresh = "--new-chat" in sys.argv
    chat_id = await s.chat(WEB_MODEL, fresh=fresh)
    print(f"chat {chat_id}{' (new)' if fresh else ''}")

    url, headers, body = await s.build(prompt, model=WEB_MODEL, stream=True,
                                       effort=effort)
    body["captcha_verify_param"] = await s.captcha()

    if "--inpage" in sys.argv:
        # Buffered: the whole reply arrives at once, which makes the frames
        # readable. The stream below is what the proxy actually uses.
        res = await s.post_in_page(url, headers, body)
        print(f"POST from inside the page -> {res.get('status')}")
        for f in frames(res.get("body") or ""):
            d = dig(f)
            if not isinstance(d, dict):
                continue
            bits = []
            if d.get("phase"):
                bits.append(f"phase={d['phase']}")
            if d.get("delta_content"):
                bits.append(f"text={str(d['delta_content'])[:150]!r}")
            if d.get("error"):
                bits.append("ERROR=" + json.dumps(d["error"], ensure_ascii=False)[:300])
            if d.get("usage"):
                bits.append("usage=" + json.dumps(d["usage"])[:200])
            if bits:
                print("   " + " | ".join(bits))
        return

    print("streaming through the page\n")
    answer = ""
    try:
        async for line in WebStream(url, headers, body).aiter_lines():
            if line == "data: [DONE]":
                break
            chunk = json.loads(line[5:])
            delta = chunk["choices"][0].get("delta") or {}
            if delta.get("reasoning_content"):
                print("  think " + str(delta["reasoning_content"])[:160])
            if delta.get("content"):
                answer += delta["content"]
                print("  " + str(delta["content"])[:160])
            if chunk.get("usage"):
                print("  usage " + json.dumps(chunk["usage"])[:200])
    except RuntimeError as exc:
        print(f"\nstream failed: {exc}")
        return
    print(f"\nanswer: {answer.strip()[:300]!r}")


if __name__ == "__main__":
    asyncio.run(main())
