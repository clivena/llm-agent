@echo off
rem ---------------------------------------------------------------------------
rem agentkit demo console - one-command launcher
rem
rem WHY THIS FILE IS PURE ASCII:
rem   cmd.exe parses a .cmd file using the console code page (936/GBK on a
rem   Chinese Windows). A UTF-8 file with CJK comments gets mis-decoded, and
rem   because UTF-8 CJK chars are 3 bytes while GBK expects 2, the leftover
rem   byte eats the following CRLF - lines get merged and the script dies with
rem   bogus "'cho' is not recognized" errors. A leading "chcp 65001" does NOT
rem   help: the parser has already committed to a code page by then.
rem   So: keep this file ASCII-only. Chinese docs live in README.md instead.
rem
rem PREREQUISITES
rem   Python 3.10+, with this repo's dependencies installed:
rem     python -m pip install -e ".[dev]"
rem
rem OVERRIDABLE ENVIRONMENT VARIABLES
rem   AGENTKIT_PYTHON      interpreter to use (skipped automatically if it
rem                        cannot import the deps; the script keeps searching)
rem   AGENTKIT_BACKEND     stub | direct | gateway
rem   AGENTKIT_API_KEY     required when backend=direct
rem   AGENTKIT_BASE_URL    upstream URL when backend=direct
rem   AGENTKIT_CHAT_MODEL  required when backend=gateway (see bottom)
rem
rem USAGE
rem   scripts\start-console.cmd                            offline stub on 127.0.0.1:8765
rem   scripts\start-console.cmd --backend direct           real model
rem   scripts\start-console.cmd --backend gateway --port 9000
rem
rem This script does only three things: find a usable interpreter, cd into the
rem repo root, and validate preconditions. Every other argument is forwarded to
rem agentkit as-is.
rem
rem WHY IT MUST cd INTO THE REPO ROOT
rem   index_path defaults to the RELATIVE path out/index.jsonl. Started from
rem   anywhere else, the index cannot be resolved and the server exits early
rem   claiming the index is empty.
rem
rem WHY backend=gateway NEEDS AN EXPLICIT chat_model
rem   agentkit hardcodes gpt-4o-mini as the default chat model for the gateway
rem   backend, but the gateway only has adapters for the models it was
rem   configured with. Without an override the very first question fails with
rem   HTTP 502 (no adapter for model gpt-4o-mini) - and that failure happens
rem   after startup, so nothing in the boot log hints at it.
rem ---------------------------------------------------------------------------
setlocal EnableDelayedExpansion

rem --- repo root (this script lives in <repo>\scripts\) ---
set "REPO_DIR=%~dp0.."
for %%I in ("%REPO_DIR%") do set "REPO_DIR=%%~fI"

rem --- only --backend / --port are inspected (for preconditions); the rest is
rem     forwarded verbatim ---
set "BACKEND="
set "PORT="
set "FORWARD="

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--backend" (
  set "BACKEND=%~2"
  set "FORWARD=!FORWARD! --backend %~2"
  shift
  shift
  goto parse
)
if /i "%~1"=="--port" (
  set "PORT=%~2"
  set "FORWARD=!FORWARD! --port %~2"
  shift
  shift
  goto parse
)
set "FORWARD=!FORWARD! %~1"
shift
goto parse

:parsed
if not defined BACKEND set "BACKEND=%AGENTKIT_BACKEND%"
if not defined BACKEND set "BACKEND=stub"
if not defined PORT set "PORT=8765"

rem --- find an interpreter: env var -> repo venv -> py launcher -> PATH ---
rem Every candidate must actually import the deps to count. Reason: on this
rem machine the first python on PATH is a bare managed interpreter with none of
rem the dependencies, so PATH order alone would pick the wrong one.
set "PY="
set "PY_SRC="

if defined AGENTKIT_PYTHON (
  call :probe "%AGENTKIT_PYTHON%" "AGENTKIT_PYTHON"
  if defined PY goto py_ready
)
if exist "%REPO_DIR%\.venv\Scripts\python.exe" (
  call :probe "%REPO_DIR%\.venv\Scripts\python.exe" ".venv"
  if defined PY goto py_ready
)
for /f "delims=" %%P in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do call :probe "%%P" "py -3"
if defined PY goto py_ready
for /f "delims=" %%P in ('where python 2^>nul') do call :probe "%%P" "PATH"
if defined PY goto py_ready
for /f "delims=" %%P in ('where python3 2^>nul') do call :probe "%%P" "PATH"
if defined PY goto py_ready

echo [start-console] ERROR: no usable Python found.
echo [start-console]   A candidate counts as usable only if it can import
echo [start-console]   typer / rich / pydantic.
echo [start-console]   Tried: AGENTKIT_PYTHON, .venv, "py -3", python, python3
echo [start-console]   Fix one of these:
echo [start-console]     1) install deps into any Python 3.10+
echo [start-console]          python -m pip install -e ".[dev]"
echo [start-console]     2) point this script at the right interpreter
echo [start-console]          set AGENTKIT_PYTHON=C:\path\to\python.exe
endlocal
exit /b 2

:py_ready
echo [start-console] interpreter = %PY%
echo [start-console]   found via   = %PY_SRC%
echo [start-console] repo        = %REPO_DIR%
echo [start-console] backend     = %BACKEND%

cd /d "%REPO_DIR%"

rem --- block the two preconditions that would otherwise fail invisibly ---
if /i "%BACKEND%"=="direct" (
  if not defined AGENTKIT_API_KEY (
    echo.
    echo [start-console] ERROR: backend=direct needs an upstream key, but
    echo [start-console]        AGENTKIT_API_KEY is not set.
    echo [start-console]        set AGENTKIT_API_KEY=sk-xxxx
    echo [start-console]        set AGENTKIT_BASE_URL=https://api.deepseek.com/v1
    endlocal
    exit /b 2
  )
)
if /i "%BACKEND%"=="gateway" (
  if not defined AGENTKIT_CHAT_MODEL (
    echo.
    echo [start-console] ERROR: backend=gateway needs an explicit AGENTKIT_CHAT_MODEL.
    echo [start-console]        Its built-in default is gpt-4o-mini, which the gateway
    echo [start-console]        has no adapter for, so the first question would fail
    echo [start-console]        with HTTP 502.
    echo [start-console]        set AGENTKIT_CHAT_MODEL=deepseek-flash
    endlocal
    exit /b 2
  )
)

rem --- build the index first if it is missing (offline, zero cost, seconds) ---
if not exist "out\index.jsonl" (
  echo [start-console] index not found, running ingest first ...
  "%PY%" -m agentkit.cli ingest
  if errorlevel 1 (
    echo [start-console] ERROR: ingest failed, aborting.
    endlocal
    exit /b 1
  )
)
if exist "out\index.meta.json" (
  echo [start-console] index       = out\index.jsonl
  type "out\index.meta.json"
  echo.
)
echo [start-console] console     = http://127.0.0.1:%PORT%
echo [start-console] starting ... press Ctrl+C to stop
echo.

"%PY%" -m agentkit.cli serve%FORWARD%
set "RC=%errorlevel%"
endlocal & exit /b %RC%

rem ---------------------------------------------------------------------------
rem :probe  is a candidate interpreter able to import this repo's deps?
rem   %1 = absolute path to the interpreter (may contain spaces)
rem   %2 = source label, log only
rem Skipped when PY is already set, so for-loops can call it unconditionally.
rem ---------------------------------------------------------------------------
:probe
if defined PY exit /b 0
"%~1" -c "import typer, rich, pydantic" >nul 2>&1
if errorlevel 1 exit /b 0
set "PY=%~1"
set "PY_SRC=%~2"
exit /b 0
