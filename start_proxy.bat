@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [*] creating venv...
    python -m venv .venv || goto :fail
)

echo [*] installing deps...
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || goto :fail

if not exist ".env" (
    copy /y ".env.example" ".env" >nul
    echo [!] .env created from .env.example - put your ZAI_API_KEY in it, then re-run.
    notepad ".env"
    exit /b 0
)

echo [*] starting zai_proxy on http://127.0.0.1:8788
".venv\Scripts\python.exe" -c "import os,sys;[os.environ.__setitem__(k.strip(),v.strip()) for k,v in (l.split('=',1) for l in open('.env',encoding='utf-8') if '=' in l and not l.strip().startswith('#'))];sys.argv=['proxy.py'];exec(open('proxy.py',encoding='utf-8').read())"
goto :eof

:fail
echo [!] setup failed.
exit /b 1
