@echo off
title Quark AI Companion
cd /d "%~dp0"

echo ========================================================
echo               QUARK AI COMPANION
echo ========================================================
echo [*] Starting server on http://localhost:8000 ...
echo.

py -3.11 -m uvicorn server:app --host 127.0.0.1 --port 8000

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server stopped with error.
    pause
)
