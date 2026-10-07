"""
websig.py -- the chat.z.ai web-app request signature (X-Signature).

The web app signs every completion request. Reverse-engineered from the
production bundle, which is javascript-obfuscator output: the salt the
signature is keyed with lives inside an RC4-encrypted string table, so it has
to be pulled back out of the bundle every time the frontend version bumps.

Shape of the signature, from the deobfuscated `sne` function:

    p   = base64(utf8(prompt))                 # the `signature_prompt` field
    h   = t + "|" + p + "|" + timestamp        # timestamp as a decimal string
    m   = floor(timestamp / 300000)            # 5-minute bucket
    k   = HMAC(<secret>, str(m))
    sig = HmacSHA256(key = k, msg = h).hex()

`timestamp` is the same millisecond value that rides in `signature_timestamp`
on the query string, so a signature is bound to one exact timestamp.

Two entry points:

    python websig.py extract path/to/index-XXXX.js
        decode the string table and print every literal `sne` uses, so the
        salt and the algorithm names can be re-read after a frontend update

    python websig.py sign "prompt text" [timestamp_ms]
        print the signature for a prompt
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import re
import sys

# --------------------------------------------------------------------------
# the obfuscator's primitives
# --------------------------------------------------------------------------

# Note the alphabet: lowercase first. This is javascript-obfuscator's, not the
# RFC 4648 one, so a plain base64.b64decode will silently produce garbage.
ALPHA = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/="


def _b64_custom(p: str) -> bytes:
    """Faithful port of the obfuscator's inline base64 decoder.

    `=` maps to index 64 rather than terminating, exactly as in the original,
    because that is what the encoder was paired with.
    """
    out = bytearray()
    x = y = 0
    for ch in p:
        w = ALPHA.find(ch)
        if w == -1:
            continue
        old = x % 4
        y = y * 64 + w if old else w
        x += 1
        if old:
            out.append((y >> ((-2 * x) & 6)) & 255)
    return bytes(out)


def _rc4(data: str, key: str) -> str:
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + ord(key[i % len(key)])) % 256
        s[i], s[j] = s[j], s[i]
    out = []
    i = j = 0
    for ch in data:
        i = (i + 1) % 256
        j = (j + s[i]) % 256
        s[i], s[j] = s[j], s[i]
        out.append(chr(ord(ch) ^ s[(s[i] + s[j]) % 256]))
    return "".join(out)


def _decode(raw: str, key: str) -> str:
    """base64 -> utf8 -> rc4, the order the obfuscator encodes in reverse."""
    return _rc4(_b64_custom(raw).decode("utf-8"), key)


def _js_parse_int(text: str) -> float:
    """parseInt: leading integer prefix, NaN when there is none."""
    m = re.match(r"^[+-]?\d+", text.strip())
    return float(int(m.group(0))) if m else float("nan")


def _unescape_array(body: str) -> list[str]:
    """The table is JSON-compatible once \\xNN is rewritten to \\u00NN."""
    return json.loads("[" + re.sub(r"\\x([0-9a-fA-F]{2})", r"\\u00\1", body) + "]")


class Bundle:
    """The string table plus the single index the table is offset by."""

    def __init__(self, src: str) -> None:
        self.src = src
        m = re.search(
            r"\}\)\(\s*([A-Za-z_$][\w$]*)\s*,\s*(\d+)\s*\);\s*"      # })(arrFn, checksum)
            r"function\s+([A-Za-z_$][\w$]*)\s*\(t,e\)\{const\s+r="  # function dec(t,e){const r=
            r"([A-Za-z_$][\w$]*)\s*\(\);\s*return\s+\3=function\(n,a\)\{"
            r"n=n-(\d+);",                                          # n = n - OFFSET
            src,
        )
        if not m:
            raise SystemExit("no obfuscator IIFE found - is this the right bundle?")
        self.arr_fn, self.checksum, self.offset = m.group(1), int(m.group(2)), int(m.group(5))

        start = src.index(f"function {self.arr_fn}(){{const t=[")
        start = src.index("[", start)
        depth = 0
        for i in range(start, len(src)):
            if src[i] == "[":
                depth += 1
            elif src[i] == "]":
                depth -= 1
                if depth == 0:
                    break
        self.table = _unescape_array(src[start + 1:i])
        self._rotate()

    def _get(self, index: int, key: str) -> str:
        return _decode(self.table[index - self.offset], key)

    CHECKSUM_ENTRIES = (
        (369, "A4Q]"), (390, "csrG"), (333, "dbV)"), (349, "iauZ"),
        (362, "#OXP"), (383, "XdC7"), (384, "a4tz"), (350, "o!4)"),
        (402, "N&aF"), (375, "xQg3"), (367, "t3yY"),
    )

    def _checksum(self) -> float:
        """The guard expression verbatim, products and all."""
        v = {i: _js_parse_int(self._get(i, k)) for i, k in self.CHECKSUM_ENTRIES}
        return (v[369] / 1 * (v[390] / 2)
                + v[333] / 3
                + v[349] / 4
                + v[362] / 5 * (-v[383] / 6)
                + -v[384] / 7 * (-v[350] / 8)
                + -v[402] / 9
                + v[375] / 10 * (-v[367] / 11))

    def _rotate(self) -> None:
        """The IIFE rotates the table until a checksum of decoded entries hits.

        A wrong rotation usually makes some entry fail to decode, which the
        original swallows with try/catch, so a raise here is just a failed
        candidate. The best clean-decode count is kept only for the error path,
        where it says whether the table is close or the decoder is wrong.
        """
        best = (-1, 0)
        for rot in range(len(self.table)):
            clean = 0
            for idx, key in self.CHECKSUM_ENTRIES:
                try:
                    self._get(idx, key)
                    clean += 1
                except Exception:
                    pass
            if clean > best[0]:
                best = (clean, rot)
            try:
                if self._checksum() == self.checksum:
                    return
            except Exception:
                pass
            self.table = self.table[1:] + self.table[:1]
        raise SystemExit(
            f"no rotation matched checksum {self.checksum}; "
            f"best had {best[0]}/{len(self.CHECKSUM_ENTRIES)} entries decoding cleanly "
            f"at rotation {best[1]} - bundle format likely changed")

    # -- reading the signature function ------------------------------------

    def sne_strings(self, cfg_prop_map: dict[str, str]) -> list[tuple[str, int, str, str]]:
        """Decode every `a(N, "key")` literal inside `sne`, with context."""
        m = re.search(r"sne=\(t,e,r\)=>\{const n=\{([^}]*)\},a=Uc,(.*?)\};", self.src, re.S)
        if not m:
            raise SystemExit("`sne` not found in the bundle")
        cfg_txt, body = m.group(1), m.group(2)

        cfg: dict[str, str] = {}
        for k, v in re.findall(r"(_0x[0-9a-f]+):(\d+|\"[^\"]*\")", cfg_txt):
            cfg[k] = v
        for prop, val in cfg.items():
            body = body.replace("n." + prop, val)

        out = []
        for idx, key in re.findall(r'a\((\d+),\s*"([^"]*)"\)', body):
            out.append((f"a({idx},{key!r})", int(idx), key, self._get(int(idx), key)))
        return out


# --------------------------------------------------------------------------
# the signature itself
# --------------------------------------------------------------------------

# Recovered from the bundle by `extract`. This one literal is the whole reason
# the signature could not just be copied from the network tab.
SALT = "key-@@@@)))()((9))-xxxx&&&%%%%%"


def payload(request_id: str, timestamp_ms: int, user_id: str) -> str:
    """`sortedPayload` from `ane()`.

    The name is misleading: it is `Object.entries` of
    `{timestamp, requestId, user_id}`, sorted by key, and the pairs are
    flattened by `Array.toString` before the comma join. So each entry
    contributes "key,value" and the whole thing is one long string. The
    requestId inside it is the same one that rides on the query string.
    """
    return ",".join([
        f"requestId,{request_id}",
        f"timestamp,{timestamp_ms}",
        f"user_id,{user_id}",
    ])


def sign(prompt: str, timestamp_ms: int, request_id: str, user_id: str,
         salt: str = SALT) -> str:
    """X-Signature for one prompt at one timestamp.

    `prompt` is the `signature_prompt` field byte for byte -- the web app
    passes the trimmed message text. `request_id` and `user_id` must be the
    same values the request itself carries, or the server sees a mismatch.
    """
    p = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
    h = f"{payload(request_id, timestamp_ms, user_id)}|{p}|{timestamp_ms}"
    bucket = str(int(math.floor(timestamp_ms / (5 * 60 * 1000))))
    key = hmac.new(salt.encode("utf-8"), bucket.encode("utf-8"),
                   hashlib.sha256).hexdigest()
    return hmac.new(key.encode("utf-8"), h.encode("utf-8"),
                    hashlib.sha256).hexdigest()


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

# One real request, captured off the wire. The requestId and user_id are the
# ones that rode on that request's query string.
VECTOR = {
    "prompt": "Reply with exactly: PONG",
    "timestamp": 1790085925834,
    "request_id": "85eec991-0553-4106-a20b-cc2535160ff6",
    "user_id": "8a4ffcc1-f663-4479-92ef-d62ba33cfd16",
    "signature": "0db3dd6e753100a4f58afc708a1f1e5612d48103864dbd4631012732a4fd8270",
}


def cmd_extract(path: str) -> None:
    b = Bundle(open(path, encoding="utf-8", errors="replace").read())
    print(f"table: {len(b.table)} entries, offset {b.offset}, "
          f"checksum {b.checksum} OK\n")
    print("literals used by `sne`:")
    for label, idx, key, val in b.sne_strings({}):
        print(f"  {label:24} -> {val!r}")


def cmd_test() -> None:
    got = sign(VECTOR["prompt"], VECTOR["timestamp"],
               VECTOR["request_id"], VECTOR["user_id"])
    ok = got == VECTOR["signature"]
    print(f"prompt     {VECTOR['prompt']!r}")
    print(f"timestamp  {VECTOR['timestamp']}")
    print(f"requestId  {VECTOR['request_id']}")
    print(f"want       {VECTOR['signature']}")
    print(f"got        {got}")
    print("MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


def main() -> None:
    a = sys.argv[1:]
    if a and a[0] == "extract":
        cmd_extract(a[1] if len(a) > 1 else ".tmp_bundle.js")
    elif a and a[0] == "test":
        cmd_test()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
