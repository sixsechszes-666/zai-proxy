@echo off
rem ---------------------------------------------------------------------
rem  Launch Claude Code against the local zai_proxy.
rem  Run start_proxy.bat in another window first.
rem ---------------------------------------------------------------------
setlocal

set "ZAI_PROXY=http://127.0.0.1:8788"

set "ANTHROPIC_BASE_URL=%ZAI_PROXY%"
set "ANTHROPIC_API_KEY=zai-proxy-local"
set "ANTHROPIC_AUTH_TOKEN=zai-proxy-local"

rem Claude Code asks for these names; the proxy maps them onto GLM.
set "ANTHROPIC_MODEL=glm-5.3-flash"
set "ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5.3-flash"
set "ANTHROPIC_DEFAULT_SONNET_MODEL=glm-5.3-flash"
set "ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.3-flash"
set "ANTHROPIC_SMALL_FAST_MODEL=glm-5.3-flash"

rem Long generations + GLM is chatty. Do not let the client give up early.
set "API_TIMEOUT_MS=900000"
set "CLAUDE_CODE_MAX_OUTPUT_TOKENS=32000"

rem Nothing should reach Anthropic's own servers.
set "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"
set "DISABLE_TELEMETRY=1"
set "DISABLE_ERROR_REPORTING=1"
set "DISABLE_AUTOUPDATER=1"
set "DISABLE_BUG_COMMAND=1"

echo Claude Code -> %ANTHROPIC_BASE_URL%  (model %ANTHROPIC_MODEL%)
claude %*
