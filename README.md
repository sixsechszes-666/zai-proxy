# zai_proxy

Anthropic Messages API → z.ai (GLM) bridge. Point Claude Code at it.

```
Claude Code ──Anthropic SSE──> zai_proxy ──OpenAI JSON/SSE──> z.ai
```

Not a passthrough: it translates both directions, repairs malformed tool
arguments, keeps tool_use ↔ tool_result pairs consistent, and re-frames the
stream into the exact event sequence Claude Code's parser expects.

## Quickstart

```bat
:: 1. put your key in .env
copy .env.example .env
notepad .env

:: 2. run the proxy (creates venv, installs deps, starts on :8788)
start_proxy.bat

:: 3. prove it works without spending quota
.venv\Scripts\python.exe selftest.py --offline

:: 4. prove it works for real
.venv\Scripts\python.exe selftest.py

:: 5. start Claude Code against it
claude-zai.bat
```

`ZAI_BASE_URL` matters for billing. `https://api.z.ai/api/coding/paas/v4`
draws on a GLM Coding Plan subscription. `https://api.z.ai/api/paas/v4` is
pay-per-token. Using the wrong one means your plan quota sits unused.

## What works in Claude Code

**Fully client-side - identical to running against Anthropic.** These never
touch the model, so a proxy can't break them: `CLAUDE.md` and memory files,
TodoWrite, Read/Write/Edit/Glob/Grep/Bash, skills, slash commands, hooks,
`.claude/settings.json`, permission rules, subagents, MCP servers, `/compact`
mechanics, session resume.

**Depends on the model, not the proxy.** Tool calling. The translation is
correct - `selftest.py --offline` proves it against canned streams, including
parallel tool calls and truncated JSON. Whether the model actually *emits* a
clean `Bash` call instead of narrating one is a model property. GLM is decent
at it and noticeably weaker than Claude at long, deep tool chains.

## Tool calling on a backend that has none

Some endpoints take a chat completion and nothing else. The z.ai web session is
one: hand it a `tools` array and it returns `INTERNAL_ERROR` on every model.
Claude Code is almost entirely tool calls, so that would normally be the end of
it.

`ZAI_TOOLS=shim` goes around it. The schemas go into the prompt as text, the
model is taught a one-line envelope for calling a tool, and the reply is parsed
back into `tool_use` blocks on the way out. The wire stays plain chat.

```
<t​ool_call>{"name": "Bash", "arguments": {"command": "ls"}}</t​ool_call>
```

The loop is closed in both directions: a `tool_use` in the history is
re-rendered as the envelope, and a `tool_result` comes back as a
`<tool_result tool="Bash">` block, so the model can tell results from prose.

| `ZAI_TOOLS` | |
|---|---|
| `native` | default. Send a `tools` array upstream. |
| `shim` | render schemas into the prompt, parse calls out of the text |
| `off` | drop tools, plain chat |

```bat
set ZAI_TOOLS=shim
.venv\Scripts\python.exe test_shim.py        :: 60 checks, offline
.venv\Scripts\python.exe test_e2e_shim.py    :: full tool loop over real HTTP
```

`test_e2e_shim.py` is the one to believe. It runs the real proxy against a mock
that refuses tools and chops its replies mid-marker, and the tool loop still
completes. Two caveats: a shimmed call depends on the model following the
envelope format, which is weaker than native function calling, and the schemas
now count as input tokens on every turn.

The envelope is the part models drop, so the parser has a fallback: a bare JSON
object is accepted as a call when it carries a `name` that matches one of the
tools offered on that turn. The allowlist is what keeps it safe - an object
naming nothing real stays prose, and so does an object inside a code block.

## Default model: glm-5.3-flash

1M context, native multimodal, 131072 max output, function calling, prompt
caching. $0.15/1M in, $0.50/1M out, $0.03/1M cached.

One thing to know before you tune anything: **its reasoning is always on and
cannot be disabled.** `thinking: {type: "disabled"}` is not honoured. So
`ZAI_THINKING` defaults to `none` - the proxy sends no thinking parameter at
all rather than asking for something the model ignores. Size the thinking with
`ZAI_REASONING_EFFORT` (`low` / `high` / `max`) instead. Upstream defaults to
`max`, which is a large amount of reasoning on every single Claude Code turn;
`low` is usually the better trade for the small/fast model slot.

Because reasoning always streams, GLM's `reasoning_content` shows up as a
thinking block on every reply. That is expected, not a bug.

Set `ZAI_MODEL=glm-4.6` if you want the older model, where thinking is optional
and `ZAI_THINKING` behaves the way its name suggests.

**Degraded.**

| Feature | Reality |
|---|---|
| Thinking | GLM's `reasoning_content` is shown as a thinking block with a faked signature. It is stripped before going back upstream, so it does not persist across turns. |
| Prompt caching | `cache_control` is ignored. Every turn resends the full context. Slower and pricier than Anthropic. |
| Images | Works if the GLM tier you're on accepts vision. |
| `/model` output | Shows whatever `ANTHROPIC_MODEL` is set to. Cosmetic. |

**Does not work.** Anthropic-hosted server tools (`web_search`, `computer_use`,
`code_execution`), the 1M-context beta, usage/limit meters, and anything that
reads Anthropic's account API. Claude Code's own local web fetch still works.

## Debugging broken tool calls

```bat
:: in .env
ZAI_DUMP=1
```

Every request lands in `dumps/` twice - `anthropic` (what Claude Code sent) and
`openai` (what went upstream). Diff the two and the bug is usually obvious:
a tool schema GLM won't accept, a result that never got paired, args it
truncated.

## Using a browser session instead of an API key

Investigated end to end on 2026-09-22 against a live session, and the answer
changed three times while doing it.

What works: the transport. The signature and the Aliyun captcha were both
reverse-engineered out of the production bundle, and real completions come back
on GLM-5.3-Flash from Python with no API key.

What used to be a wall, and is not any more: client-supplied tools. The web
endpoint refuses a `tools` array on every model. `ZAI_TOOLS=shim` sidesteps it
entirely by rendering the schemas into the prompt instead, and the full tool
loop was then run live against it - a Bash schema in, a `tool_use` block out,
the result handed back, an answer written from it. That run also found the one
prompt-following mistake worth knowing about: the model sometimes emits the call
JSON without the `<tool_call>` wrapper, which used to end the loop in silence.
The parser now accepts that form when the name matches a tool actually offered.

`ZAI_UPSTREAM=web` points the proxy at the browser session. No key, shim forced
on, because the endpoint can never receive a tools array.

```bat
set ZAI_UPSTREAM=web
set ZAI_TOOLS=shim
.venv\Scripts\python.exe proxy.py
```

What actually limits it: throughput, and it turns out to be the exit IP rather
than the account. Nine requests in a burst go through, then the edge answers
`405` to everything on the completions path - including the app's own request
sent by hand from the z.ai UI. A fresh egress node clears it immediately, on the
same account and the same tab, which is the difference between a dead end and a
dodge. Claude Code needs 20-50 turns per task, so rotation makes it survivable
rather than comfortable. Automatic retry-on-405 is not built yet.

The account also has no platform API access, which is why the browser looked
like the only door in the first place.

```bat
.venv\Scripts\python.exe webclient.py --whoami
.venv\Scripts\python.exe webclient.py --chat "say hi"
.venv\Scripts\python.exe test_web.py             :: 53 checks, no browser, no quota
.venv\Scripts\python.exe test_e2e_web.py         :: the full tool loop, live, 2 requests
.venv\Scripts\python.exe websig.py test          :: signature self-test
```

`WEB_MODE.md` has the evidence, the model ids, and the one page worth checking.

## Files

| File | |
|---|---|
| `proxy.py` | the server |
| `toolshim.py` | tool calling over a backend that has none: prompt rendering and the streaming parser |
| `selftest.py` | `--offline` translation tests, or live end-to-end |
| `test_shim.py` | 60 checks on the shim, both directions, no network |
| `test_e2e_shim.py` | the full tool loop over real HTTP against a tools-refusing mock |
| `test_web.py` | the web path's translation, offline against canned frames |
| `test_e2e_web.py` | the full tool loop, live, through the browser session |
| `mock_shim.py` | that mock, if you want to poke at it by hand |
| `websig.py` | the web app's `X-Signature`, and the bundle extractor that re-derives its salt |
| `webclient.py` | talks to chat.z.ai as the browser session: JWT, signature, captcha, in-page transport, streaming bridge |
| `cdp.py` | CDP recon harness: eval, navigate, watch traffic |
| `probe_web.py` | who am I / what models, against the web backend |
| `capture.py` | HAR / mitmproxy capture of the z.ai web session |
| `WEB_MODE.md` | the browser-session investigation, all of it |
| `start_proxy.bat` | venv + deps + run |
| `claude-zai.bat` | Claude Code with the right env vars |
| `.env.example` | every knob |
