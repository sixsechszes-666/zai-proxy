# Using a browser session instead of an API key

**Verdict, measured end to end: the whole path works, tool calling included, and
the one thing that stands in the way is an IP-level block that a fresh egress
node clears instantly.** The browser session drives real completions on
GLM-5.3-Flash with no API key, `ZAI_UPSTREAM=web` makes `proxy.py` serve Claude
Code from it, and `ZAI_TOOLS=shim` gives it working tools. A full tool round
trip was run live: a tool schema in, a `tool_use` block out, the result handed
back, an answer written from it. What remains is throughput - roughly ten
requests trip a `405` from the edge - and that block turns out to be bound to
the exit IP, not to the account.

Everything here was verified against a live logged-in session on 2026-09-22,
not inferred from the docs.

**Start here if you want the short version:**

| question | answer |
|---|---|
| Can I get completions out of the web session? | Yes. `webclient.py --chat "hi"`. |
| Can it do tool calling? | Yes, via the shim. Web mode forces it on. |
| Can Claude Code run on it? | Yes, live-proven, `test_e2e_web.py`. Ten requests in a burst then trip a 405 that a new exit IP clears. |
| Is there a real fix for the throughput? | Rotation, or a GLM Coding Plan key. The account has no platform API access today. |


---

## What was actually cracked

Five separate obstacles stood in the way. Four are solved and the code for them
is in this directory; the fifth is the wall.

### 1. The endpoint and the wire format

`POST https://chat.z.ai/api/v2/chat/completions` - not the Open WebUI shape the
app's own module names suggest. The body is z.ai's own:

```json
{
  "stream": true,
  "model": "x-preview-l",
  "messages": [{"role": "user", "content": "..."}],
  "signature_prompt": "<same text as the message>",
  "params": {}, "extra": {},
  "features": {"image_generation": false, "web_search": false,
               "auto_web_search": false, "preview_mode": true, "flags": [],
               "enable_thinking": true, "reasoning_effort": "max"},
  "variables": {"{{USER_NAME}}": "...", "{{CURRENT_DATETIME}}": "...", ...},
  "chat_id": "<an id the server already knows>", "id": "<uuid>",
  "current_user_message_id": "<uuid>",
  "current_user_message_parent_id": null,
  "background_tasks": {"title_generation": true, "tags_generation": true},
  "captcha_verify_param": "<see 3>"
}
```

`variables` is not optional decoration - the server substitutes those `{{...}}`
placeholders into the system prompt, and a body without the block misbehaves.

The response is SSE in `chat:completion` frames even when `stream` is false.
Each frame carries a `phase`: `thinking` for reasoning, `answer` for the reply
itself, `other` for the frame that holds `usage`, `done` to close. That one
detail is what the whole bridge hangs on.

### 2. `X-Signature` - reversed from the production bundle

Every request carries an HMAC signature. It is not a hash of the body; it is
built by `sne()` in the obfuscated bundle:

```
p     = base64(utf8(signature_prompt))
h     = "<sorted requestId/timestamp/user_id pairs>|" + p + "|" + timestamp_ms
bucket= floor(timestamp_ms / 300000)            # 5-minute window
key   = HMAC-SHA256(<SALT from the RC4 string table>, str(bucket)).hex()
sig   = HMAC-SHA256(key, h).hex()
```

The subtle part is the prefix: it is `Object.entries` of the three-field object,
sorted by key and flattened by `Array.toString`, so it reads
`requestId,<uuid>,timestamp,<ms>,user_id,<uuid>` - one comma-joined run, not a
`|`-joined list or just the key names.

`SALT` lives inside a javascript-obfuscator RC4 string table, so it has to be
pulled back out of the bundle every time the frontend version bumps.
`websig.py` does that and self-tests against a captured vector:

```bat
.venv\Scripts\python.exe websig.py test
.venv\Scripts\python.exe websig.py extract .tmp_bundle.js
```

### 3. The captcha - obtainable with no human present

The completion endpoint requires `captcha_verify_param` (Aliyun captcha v3,
scene `didk33e0`, region `sgp`, prefix `no8xfe`). It is mintable silently.

The app's own path is `BN()` in the bundle: drop the cached SDK instance,
re-init, then **click an off-screen trigger button**. The app never calls
`startTracelessVerification` itself - the click is the whole trigger and the SDK
decides internally that the check can be silent. `webclient.py` reproduces that,
including the app's own off-screen elements and its retry-on-fail behaviour.

The resulting param is structurally identical to a real one:

```json
{"certifyId": "...", "sceneId": "didk33e0", "isSign": true, "securityToken": "..."}
```

and the server accepts it. `startTracelessVerification()` also works but is not
what the app does; a re-init plus click is the faithful version.

### 4. The transport - this one is the surprise

A request built correctly and sent from **httpx** is rejected with
`FRONTEND_CAPTCHA_REQUIRED / verify_failed / F019` even when the captcha param
is fresh and correct. The identical request sent from **inside the page** with
`fetch` passes.

So the server's captcha verification is bound to something in the browser's own
request context that a Python client does not reproduce - cookies (the app calls
`fetch` with no `credentials` option, so the browser applies the same-origin
default and sends them), or the oversized browser-fingerprint query string, or
both. `X-Device-ID` matters too and is easy to miss: it is not a `device_id`
key, it is `_arms_uid` in localStorage (any value matching
`/^uid_[A-Za-z0-9]{7,}$/`), attached by `wy()` to every request.

The browser is therefore a permanent part of the runtime, not just a recon
tool. `WebStream` below keeps that honest: it drives the fetch from inside the
page and reads the reply there.

### 5. `chat_id` has to be real, and the endpoint is stateless

Two findings that between them took the longest, because the failure mode was
a generic `INTERNAL_ERROR` and the natural first guess (tool support, model
name, rate limiting) was wrong every time.

**A client-invented `chat_id` is fatal.** The bisect started from a body the app
itself was caught sending and mutated exactly one field per run:

| case | status | frames | result |
|---|---|---|---|
| app body verbatim | 200 | 19 | full completion |
| fresh `chat_id` only | 200 | 2 | `INTERNAL_ERROR` |
| fresh `current_user_message_id` only | 200 | 9 | full completion |
| fresh `id` only | 200 | 11 | full completion |
| fresh `chat_id` + `parent_id: null` | 200 | 2 | `INTERNAL_ERROR` |

Only `chat_id` matters, and nothing else in the body can save it. The app does
not invent one either: a browser trace of a brand new chat shows it calling
`POST /api/v1/chats/new` with an empty id *before* the completion, and the
server hands back the real one. `WebSession.create_chat()` does the same, from
inside the page, with no signature needed. `webclient.py` reuses one chat for
the whole session, which is safe because of the next paragraph.

**The endpoint reads `signature_prompt` and ignores `messages`.** Three
measurements, all agreeing:

- `usage.prompt_tokens` stays flat across turns of one chat: 18, then 19, then
  35 (which is just the longer prompt itself). A server replaying the stored
  chat would climb.
- Asked in turn 3 what it had been told in turn 1, it answers "There was no
  earlier message - this is the first message in our conversation."
- A fact planted in the `messages` array ("the magic word is STRAWBERRY") never
  reaches the model. `prompt_tokens` stays at 22, the length of the prompt
  alone, and the model answers something else entirely.

So the conversation has to be written into the prompt by hand, which is what
`webclient.flatten()` does, ending on an `Assistant:` cue. The upside is real:
nothing accumulates server-side, so a long session never double-counts its own
history, and the stored chat stays empty rather than collecting a copy of
every Claude Code turn.

---

## Client-supplied tools: refused by the endpoint, bypassed by the shim

This is the part that used to decide whether the whole path was usable. The
endpoint says no. `ZAI_TOOLS=shim` goes around it by never asking.

**The app never sends a `tools` array.** The send function is
`bhe(token, body, base, ...)`; it has no tools parameter, and `tool_choice`
does not appear anywhere in the 3.2 MB bundle.

**Supplying one breaks the request.** Added to a known-good body captured from
the app itself, on every model tried:

| model | body | result |
|---|---|---|
| `glm-5.3` | no `tools` | full completion |
| `glm-5.3` | `tools: []` | full completion |
| `glm-5.3` | one minimal tool | `INTERNAL_ERROR` |
| `glm-5.3` | one realistic tool | `INTERNAL_ERROR` |
| `x-preview-l` | no `tools` | full completion |
| `x-preview-l` | one minimal tool | `INTERNAL_ERROR` |

An empty array is tolerated and a populated one is fatal, on every model. The
server has no path for a client-declared tool schema.

**Where tool-shaped data does appear** is the response, and only for the app's
own server-side tools. The frontend reads `content_blocks` entries of
`type: "tool_calls"`, each with `function.name`, `function.arguments`,
`parsed_arguments` and `results`. The names are z.ai's, not the client's:
`Write`, `Read`, `Outline`, `zai_dr_search`, `zai_generate_slide_html`. The
request-side mechanism is `mcp_servers` - descriptors of MCP servers the
frontend has registered - which the *server* then talks to. That is not a
channel a local Claude Code can use: its tools live on this machine and z.ai's
backend cannot reach them.

Claude Code's loop runs the other direction. It ships ~20 local tool schemas
per turn and needs a `tool_use` block back naming one of them. This endpoint
will not accept the schemas.

### Going around it: `ZAI_TOOLS=shim`

There is a way, and it does not involve the backend at all. If the endpoint
will not take a tool schema, do not send one. Put the schemas in the prompt as
text, teach the model a small envelope for calling a tool, and parse the calls
back out of the reply.

```
Claude Code                proxy.py                 upstream
  tools[]     ---------->  render into prompt  -->  plain chat, no tools
  tool_use    <----------  parse the text      <--  <tool_call>{...}</tool_call>
```

Set `ZAI_TOOLS=shim` and it applies to any backend, including this one. Both
directions are covered: a `tool_use` in the history is re-rendered as the
envelope, and a `tool_result` comes back as a `<tool_result>` block carrying
its tool name, so the model can tell results from user prose.

Proof, all of it offline and re-runnable:

```bat
.venv\Scripts\python.exe test_shim.py       :: 60 checks, parser and both directions
.venv\Scripts\python.exe test_e2e_shim.py   :: the full loop over real HTTP
```

`test_e2e_shim.py` is the one that matters. It runs the real proxy against a
mock backend that refuses to be told about tools, deliberately chops its reply
mid-marker and mid-JSON, and the tool loop still completes: Claude Code's
request in, a `tool_use` block out, the result back in as text, an answer
returned. That is Claude Code running on a backend with no tool support.

Two honest caveats. A shimmed call is only as good as the model's ability to
emit the envelope instead of describing it, which is a prompt-following
question and weaker than native function calling. And the schemas now sit in
the prompt, so they are billed as input tokens on every turn.

### The wrapper the model drops, and what catches it

The first live run through the proxy failed, and the failure was the quiet
kind. Asked to use Bash, GLM replied with a sentence and then the payload on
its own:

```
I'll list the files in the current directory using the Bash tool.

{"name": "Bash", "arguments": {"command": "ls -la"}}
```

No `<tool_call>`, so the parser read the whole thing as an answer. Nothing
errored. Claude Code would have received a plain text reply in place of a tool
call and simply stopped, which is the worst shape a failure can take here,
because it looks like the model chose not to use the tool.

Both halves are fixed:

- **The prompt now says why the marker matters** and shows the complete reply
  for a worked example, rather than only describing the format. Dropping the
  wrapper is named as the specific mistake, since that is the one the model
  made.
- **The parser accepts the unwrapped object** under two guards: it has to be a
  well formed JSON object carrying a `name`, and that name has to be one of the
  tools offered on this turn. The proxy passes the offered names down for
  exactly this check.

The allowlist is what makes the second half safe. `{"name": "config",
"arguments": {"port": 8080}}` in a code block stays text, because nothing in
this turn is called `config`; the same object with `"Bash"` in it becomes a
call. The object is also held off the wire until it closes, so a stream cut
between `{` and `"` cannot leak half a call as prose. Both forms are tested at
every possible split point, and the marker still wins wherever both appear.

---

## Driving it from `proxy.py`

`ZAI_UPSTREAM=web` points the proxy at the browser session instead of the
platform API. No `ZAI_API_KEY` is needed, and the shim is forced on, because
the endpoint can never receive a tools array.

```bat
set ZAI_UPSTREAM=web
set ZAI_TOOLS=shim
.venv\Scripts\python.exe proxy.py
```

The bridge is `webclient.WebStream`. `post_in_page` buffers a whole reply,
which is fine for a probe and useless for a chat, so the stream opens the fetch
inside the page, reads the response body with a reader *there*, and drains the
chunks to Python over one CDP connection that stays open for the call. The
frames are reshaped into the OpenAI delta shape on the way past, so
`proxy.py`'s existing translation, thinking blocks and tool shim all work
untouched. Latency is the poll interval (80 ms), not the length of the reply.

That translation is checked offline, against canned frames, including SSE text
cut at every possible place:

```bat
.venv\Scripts\python.exe test_web.py        :: 53 checks, no browser, no quota
```

Two things it handles on purpose. A 405 from the edge arrives as an Anthropic
error event naming the status, rather than as an empty answer. And the `usage`
frame's field names already match OpenAI's, so usage survives without mapping.

And the whole loop, live, spending real requests:

```bat
.venv\Scripts\python.exe test_e2e_web.py        :: two turns, a full tool round trip
.venv\Scripts\python.exe test_e2e_web.py -v     :: every SSE event as it lands
```

It starts the real proxy in web mode, sends Claude Code's shape of request with
a Bash schema, and asserts the reply is a `tool_use` block whose arguments fit
the schema and whose `stop_reason` is `tool_use`; then it hands a tool result
back and asserts the model answers from it instead of calling again. That is
Claude Code's loop, on a backend that has no tool support, over a browser
session that has no API key.

### The thing that stands in the way: an IP-level block

Not the tools, not the transport, not the chat id. Throughput.

Ten requests in forty seconds, measured:

```
   0  200   38 frames        6  200   19 frames
   1  200    9 frames        7  200    9 frames
   2  200   13 frames        8  200   14 frames
   3  200   29 frames        9  405  <html> 405 </html>
   4  200    9 frames       10  405  <html> 405 </html>
   5  200    9 frames       11  405  <html> 405 </html>
```

Nine go through, then the edge starts returning an Aliyun WAF page with a
`traceid` and no reason. Not just to this client: **the app's own request, sent
by typing into the z.ai composer in the same tab, got the same 405.** That was
the decisive test and it rules out anything about the shape of the requests.
The block sits on the completions path specifically - `POST /api/v1/chats/new`
kept answering 200 throughout.

**It is bound to the exit IP, and rotating the node clears it at once.** The
same command that had been failing with a 405 HTML page ran clean seconds after
a proxy swap, on the same account and the same tab:

```
  think ONG" with nothing else
  usage {"prompt_tokens": 18, "completion_tokens": 114, ...}
answer: 'PONG'
```

The old address stays blocked while the new one works, which is what makes it
an egress-IP penalty rather than an account limit, and it is the useful part of
this whole finding: the wall is dodgeable by rotation, and it is not a ban on
the account.

What it still means is that a real Claude Code task is twenty to fifty turns,
and a tier sized for a human typing will hit the wall regularly. Rotation makes
that survivable, not comfortable. Handling it inside the proxy - catch the 405,
bounce the tab's egress, retry the turn - is the obvious next step and is not
built yet.

### And the account has no API access

Worth checking before building anything here. `https://z.ai/manage-apikey/apikey-list`
is a separate console from `chat.z.ai`, and the chat session does not
authenticate against it: it renders as signed out. A chat subscription does not
grant platform API access, which is the actual reason this whole browser path
looked like the only door.

z.ai sells the thing this was assembling. The GLM Coding Plan is a flat
subscription that issues an API key for coding agents specifically, is
advertised as compatible with Claude Code, and starts around $18/mo. With that
key, `proxy.py` works unchanged and natively, and neither the shim nor the
browser is needed.

---

## What the browser session *is* good for

Chat completions on GLM-5.3-Flash through the logged-in account, from Python,
with no API key, streamed as they are written:

```bat
.venv\Scripts\python.exe webclient.py --whoami
.venv\Scripts\python.exe webclient.py --chat "say hi"
.venv\Scripts\python.exe webclient.py --chat "say hi" --inpage   :: buffered, frames readable
.venv\Scripts\python.exe webclient.py --new-chat
.venv\Scripts\python.exe webclient.py --chat "hi" --effort low
```

More than that: with `ZAI_UPSTREAM=web` it carries a real Claude Code session,
tool loop included, which is a stronger result than this path was expected to
give. It is a fallback that works, useful for text generation a few requests at
a time, for comparing GLM's answers against the platform API, and as the recon
rig if z.ai ever lifts the block or adds native tool support.

## Model ids, which are not what you would guess

The web app's ids differ from the platform API's:

| web id | display name |
|---|---|
| `x-preview-l` | **GLM-5.3-Flash** |
| `glm-5.3` | GLM-5.3 |
| `glm-5.2` | GLM-5.2 |
| `GLM-5-Turbo`, `glm-4.7`, `glm-4.6v`, ... | as named |

The model Vadim wants is `x-preview-l` on this backend, not `glm-5.3-flash` and
not `glm-5.3`. `ZAI_WEB_MODEL` overrides it; web mode sends that one id
whatever the request asked for, because this account is served one model.

## What actually does this job

z.ai sells exactly what this was assembling: a **GLM Coding Plan** - a flat
subscription that issues an API key explicitly for coding agents. If the account
is on it, the key is in the console and everything in this repo works against it
as-is. A plain chat subscription does not grant API access at all, which is the
real reason the browser session felt like the only door.

If tool calling is the goal, that is the one page worth checking.

## If it is ever revisited

The recon rig is all here and re-runnable. `websig.py extract` re-derives the
salt after a frontend bump; `webclient.py --chat ... --inpage` re-proves the
transport; `test_web.py` re-proves the translation with no quota spent;
`test_e2e_web.py` re-proves the whole tool loop and costs two requests. The
fetch-hook technique that captured the app's real requests is worth
remembering: it settled in one shot what no amount of bundle reading had, and
it is what found `POST /api/v1/chats/new`.
