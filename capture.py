"""
capture.py -- pull the real shapes out of your logged-in z.ai browser session.

Two modes:

  1) HAR (no extra tooling, recommended)
       Chrome -> chat.z.ai -> F12 -> Network -> send one message
       -> right-click the list -> "Save all as HAR with content"
       python capture.py har chat.z.ai.har

     Prints every z.ai request with its auth headers, body and response, and
     writes web_profile.draft.json -- the file the web upstream needs.

  2) mitmproxy addon (live, keeps raw SSE framing)
       mitmdump -s capture.py --set confdir=~/.mitmproxy
     Writes capture_zai.jsonl as you use the site.

Nothing here is sent anywhere. The dumps contain your session token -- keep
them local and delete when done.
"""

from __future__ import annotations

import json
import os
import sys
from urllib.parse import urlparse

WANT_HOSTS = ("z.ai", "chat.z.ai", "api.z.ai", "open.bigmodel.cn", "bigmodel.cn")
SKIP_SUFFIX = (".js", ".css", ".png", ".jpg", ".svg", ".woff", ".woff2", ".ico", ".map")

OUT_DRAFT = "web_profile.draft.json"
OUT_JSONL = "capture_zai.jsonl"


def interesting(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    if not any(h in host for h in WANT_HOSTS):
        return False
    path = urlparse(url).path.lower()
    return not path.endswith(SKIP_SUFFIX)


def show(label: str, value: str, limit: int = 4000) -> None:
    if value is None:
        return
    text = str(value)
    if len(text) > limit:
        text = text[:limit] + f"\n... [{len(text) - limit} more chars]"
    print(f"    {label}: {text}")


# --------------------------------------------------------------------------
# HAR mode
# --------------------------------------------------------------------------

def from_har(path: str) -> None:
    with open(path, "r", encoding="utf-8") as fh:
        har = json.load(fh)

    entries = (har.get("log") or {}).get("entries") or []
    hits = []
    for e in entries:
        req = e.get("request") or {}
        url = req.get("url") or ""
        if not interesting(url):
            continue
        resp = e.get("response") or {}
        ctype = ""
        for h in resp.get("headers") or []:
            if h.get("name", "").lower() == "content-type":
                ctype = h.get("value", "")
        body = ((resp.get("content") or {}).get("text")) or ""
        if req.get("method") == "GET" and not body:
            continue
        hits.append((req, resp, ctype, body))

    print(f"\n{len(hits)} candidate requests out of {len(entries)} HAR entries\n")

    drafts = []
    for req, resp, ctype, body in hits:
        url = req["url"]
        print("=" * 78)
        print(f"{req.get('method')} {url}")
        print(f"  status  : {resp.get('status')}  {ctype}")
        print("  -- request headers --")
        hdrs = {h.get("name"): h.get("value") for h in (req.get("headers") or [])}
        for k, v in hdrs.items():
            if k.lower() in ("cookie", "authorization", "x-api-key", "x-token", "x-signature"):
                print(f"    {k}: {v[:120]}{'...' if len(v) > 120 else ''}")
        print("  -- all header names --")
        print("    " + ", ".join(hdrs))
        post = (req.get("postData") or {}).get("text")
        if post:
            print("  -- request body --")
            try:
                show("json", json.dumps(json.loads(post), ensure_ascii=False, indent=2))
            except Exception:
                show("raw", post)
        if body:
            print("  -- response body (first 3000) --")
            show("body", body, 3000)
            if "event-stream" in ctype or body.lstrip().startswith("data:"):
                print("  >> looks like SSE -- this is the completion stream")
        print()

        drafts.append({
            "url": url,
            "method": req.get("method"),
            "request_headers": {k: ("<SESSION_TOKEN>" if k.lower() in
                                    ("cookie", "authorization", "x-api-key", "x-token", "x-signature")
                                    else v) for k, v in hdrs.items()},
            "request_body_example": post,
            "response_content_type": ctype,
            "response_sample": body[:4000],
        })

    if drafts:
        with open(OUT_DRAFT, "w", encoding="utf-8") as fh:
            json.dump(drafts, fh, ensure_ascii=False, indent=2)
        print(f"wrote {OUT_DRAFT} ({len(drafts)} entries) -- hand this file back.")
    else:
        print("nothing matched. did the HAR include content? "
              'it must be "Save all as HAR with content", not plain HAR.')


# --------------------------------------------------------------------------
# mitmproxy addon mode
# --------------------------------------------------------------------------

class ZaiCapture:
    """Usage: mitmdump -s capture.py"""

    def __init__(self) -> None:
        self.fh = open(OUT_JSONL, "a", encoding="utf-8")
        print(f"[capture] appending to {os.path.abspath(OUT_JSONL)}")

    def _write(self, rec: dict) -> None:
        self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.fh.flush()

    def request(self, flow) -> None:
        if not interesting(flow.request.pretty_url):
            return
        flow._zai_req = {
            "ts": flow.request.timestamp_start,
            "method": flow.request.method,
            "url": flow.request.pretty_url,
            "headers": dict(flow.request.headers),
            "body": flow.request.get_text(strict=False),
        }

    def response(self, flow) -> None:
        rec = getattr(flow, "_zai_req", None)
        if not rec:
            return
        rec["status"] = flow.response.status_code
        rec["resp_headers"] = dict(flow.response.headers)
        rec["resp_body"] = flow.response.get_text(strict=False)
        self._write(rec)
        print(f"[capture] {rec['method']} {rec['url']} -> {rec['status']} "
              f"({len(rec['resp_body'] or '')} bytes)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "har":
        from_har(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
