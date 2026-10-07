"""
cdp.py -- drive the Chrome that pomelli_driver already owns, over CDP.

This is a recon harness. It exists so the z.ai web backend can be observed for
real instead of guessed at: what endpoint the chat app posts to, what carries
the auth, what the stream looks like coming back, and whether the request
supports tool definitions at all.

  python cdp.py tabs
  python cdp.py open https://chat.z.ai/
  python cdp.py nav  https://chat.z.ai/
  python cdp.py eval "document.title"
  python cdp.py eval --json "Object.keys(localStorage)"
  python cdp.py watch 40 --out recon.json
  python cdp.py watch 40 --filter z.ai,google --out recon.json

Loopback must never go through Clash, so the proxy environment is stripped
before anything connects. Chrome's own page loads still use the Windows system
proxy; this only affects our socket to 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any
from urllib.parse import quote

for _v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_v, None)
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost,::1"

import httpx          # noqa: E402
import websockets     # noqa: E402

PORT = int(os.environ.get("ZAI_CDP_PORT", os.environ.get("POMELLI_CDP_PORT", "9222")))
BASE = f"http://127.0.0.1:{PORT}"

SKIP_EXT = (".js", ".css", ".png", ".jpg", ".jpeg", ".svg", ".woff", ".woff2",
            ".ico", ".map", ".gif", ".webp", ".mp4")


def http() -> httpx.Client:
    return httpx.Client(timeout=10, trust_env=False)


def tabs() -> list[dict]:
    with http() as c:
        return c.get(f"{BASE}/json/list").json()


def pages() -> list[dict]:
    return [t for t in tabs() if t.get("type") == "page"]


def pick(match: str = "") -> dict:
    ps = pages()
    if not ps:
        raise SystemExit("no page targets. is Chrome running with --remote-debugging-port?")
    if match:
        for p in ps:
            if match.lower() in (p.get("url", "") + p.get("title", "")).lower():
                return p
        raise SystemExit(f"no page matching {match!r}. open tabs:\n" +
                         "\n".join(f"  {p.get('url','')[:110]}" for p in ps))
    return ps[0]


def create_tab(url: str) -> dict:
    """Chrome wants PUT here on current versions; GET is gone."""
    with http() as c:
        try:
            r = c.put(f"{BASE}/json/new?{quote(url, safe='')}")
        except httpx.HTTPError:
            r = c.get(f"{BASE}/json/new?{quote(url, safe='')}")
        if r.status_code >= 400:
            raise SystemExit(f"could not open a tab: {r.status_code} {r.text[:200]}")
        return r.json()


class Page:
    def __init__(self, ws) -> None:
        self.ws = ws
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self.events: asyncio.Queue = asyncio.Queue()
        self._reader_task: asyncio.Task | None = None

    async def __aenter__(self) -> "Page":
        self._reader_task = asyncio.create_task(self._reader())
        return self

    async def __aexit__(self, *exc) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        await self.ws.close()

    async def _reader(self) -> None:
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                if "id" in msg:
                    fut = self._pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                else:
                    await self.events.put(msg)
        except Exception:
            pass

    async def send(self, method: str, timeout: float = 40, **params) -> dict:
        self._id += 1
        mid = self._id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        return await asyncio.wait_for(fut, timeout)

    async def drain(self, seconds: float, handler) -> None:
        deadline = time.time() + seconds
        while True:
            left = deadline - time.time()
            if left <= 0:
                return
            try:
                ev = await asyncio.wait_for(self.events.get(), left)
            except asyncio.TimeoutError:
                return
            handler(ev)


async def with_page(fn, match: str = ""):
    target = pick(match)
    ws_url = target["webSocketDebuggerUrl"]
    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        async with Page(ws) as page:
            return await fn(page, target)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

async def cmd_eval(js: str, match: str, as_json: bool) -> None:
    async def run(page: Page, target: dict):
        await page.send("Runtime.enable")
        res = await page.send("Runtime.evaluate", expression=js,
                              returnByValue=True, awaitPromise=True)
        return res

    res = await with_page(run, match)
    result = (res.get("result") or {})
    if "exceptionDetails" in result:
        print("JS threw:", json.dumps(result["exceptionDetails"], ensure_ascii=False)[:800])
        return
    val = (result.get("result") or {}).get("value")
    if as_json or isinstance(val, (dict, list)):
        print(json.dumps(val, ensure_ascii=False, indent=2)[:20000])
    else:
        print(val)


async def cmd_nav(url: str, match: str) -> None:
    async def run(page: Page, target: dict):
        await page.send("Page.enable")
        await page.send("Page.navigate", url=url)
        await page.drain(6.0, lambda ev: None)
        r = await page.send("Runtime.evaluate", expression="location.href",
                            returnByValue=True)
        return (r.get("result") or {}).get("result", {}).get("value")

    href = await with_page(run, match)
    print("now at:", href)


async def cmd_watch(seconds: float, filters: list[str], out: str, match: str) -> None:
    records: list[dict] = []
    by_request: dict[str, dict] = {}
    bodies: list[str] = []

    def interesting(url: str) -> bool:
        low = url.lower()
        if low.endswith(SKIP_EXT):
            return False
        if filters:
            return any(f.lower() in low for f in filters)
        return True

    def handler(ev: dict) -> None:
        m = ev.get("method")
        p = ev.get("params") or {}
        if m == "Network.requestWillBeSent":
            req = p.get("request") or {}
            url = req.get("url", "")
            if not interesting(url):
                return
            rec = {
                "id": p.get("requestId"),
                "method": req.get("method"),
                "url": url,
                "headers": req.get("headers") or {},
                "postData": req.get("postData"),
                "ts": p.get("timestamp"),
            }
            by_request[rec["id"]] = rec
            records.append(rec)
        elif m == "Network.responseReceived":
            rid = p.get("requestId")
            rec = by_request.get(rid)
            if rec is None:
                return
            resp = p.get("response") or {}
            rec["status"] = resp.get("status")
            rec["mimeType"] = resp.get("mimeType")
            rec["resp_headers"] = resp.get("headers") or {}
            if resp.get("status") in (301, 302, 303, 307, 308):
                rec["redirect"] = resp.get("headers", {}).get("location")
        elif m == "Network.loadingFinished":
            rid = p.get("requestId")
            rec = by_request.get(rid)
            if rec is not None and rec.get("status") == 200:
                bodies.append(rid)

    async def run(page: Page, target: dict):
        await page.send("Network.enable")
        print(f"watching {seconds:.0f}s ... do the thing in the browser now")
        await page.drain(seconds, handler)

        # pull bodies for the ones that completed
        for rid in bodies[:120]:
            rec = by_request.get(rid)
            if rec is None:
                continue
            try:
                r = await page.send("Network.getResponseBody", requestId=rid, timeout=15)
            except Exception:
                continue
            body = ((r.get("result") or {}).get("body")) or ""
            rec["resp_body"] = body[:60000]
            rec["resp_body_len"] = len(body)
        return records

    recs = await with_page(run, match)

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(recs, fh, ensure_ascii=False, indent=2)

    print(f"\n{len(recs)} requests -> {out}\n")
    for rec in recs:
        blen = rec.get("resp_body_len")
        print(f"  {rec.get('method','?'):5} {rec.get('status','---')} "
              f"{(str(blen)+'B') if blen is not None else '':>8}  {rec['url'][:110]}")

    print("\n--- auth-carrying headers seen ---")
    seen: set[str] = set()
    for rec in recs:
        for k in (rec.get("headers") or {}):
            if k.lower() in ("cookie", "authorization", "x-api-key", "x-token",
                             "x-signature", "x-csrf-token", "x-device-id",
                             "x-request-id", "x-session-id") and k.lower() not in seen:
                seen.add(k.lower())
                print(f"  {k}")
    if not seen:
        print("  (none of the usual suspects)")

    print("\n--- bodies that mention tools ---")
    hit = False
    for rec in recs:
        body = rec.get("resp_body") or ""
        post = rec.get("postData") or ""
        if '"tools"' in post or '"tool_calls"' in body or '"tool"' in post:
            hit = True
            print(f"  {rec['url'][:110]}")
    if not hit:
        print("  nothing. no tool schema in any request body, no tool_calls in any response.")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]
    rest = args[1:]
    match = ""
    as_json = False
    if "--match" in rest:
        i = rest.index("--match")
        match = rest[i + 1]
        rest = rest[:i] + rest[i + 2:]
    if "--json" in rest:
        as_json = True
        rest.remove("--json")

    if cmd == "tabs":
        for i, t in enumerate(tabs()):
            print(f"[{i}] {t.get('type'):9} {t.get('title','')[:50]:52} {t.get('url','')[:95]}")
    elif cmd == "open":
        t = create_tab(rest[0] if rest else "about:blank")
        print("opened:", t.get("id"), t.get("url", ""))
    elif cmd == "nav":
        asyncio.run(cmd_nav(rest[0], match))
    elif cmd == "eval":
        asyncio.run(cmd_eval(" ".join(rest), match, as_json))
    elif cmd == "watch":
        seconds = float(rest[0]) if rest and rest[0].replace(".", "").isdigit() else 30.0
        filters: list[str] = []
        if "--filter" in rest:
            i = rest.index("--filter")
            filters = [f for f in rest[i + 1].split(",") if f]
        out = "recon.json"
        if "--out" in rest:
            out = rest[rest.index("--out") + 1]
        asyncio.run(cmd_watch(seconds, filters, out, match))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
