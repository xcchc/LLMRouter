@echo off
chcp 65001 >nul
cd /d "%~dp0"
title LLM Router Console
echo ============================================
echo   LLM Router - Local Gateway Console
echo ============================================
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [First run] Creating Python environment and installing dependencies...
    echo This may take a few minutes, please wait...
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip >nul
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    echo Dependencies installed.
)
echo.
echo Starting LLM Router... (closing the window hides it to the tray; exit from the tray icon)
.venv\Scripts\python.exe -X utf8 -u app.py
echo.
echo ============================================
echo Program exited with code %errorlevel%.
echo If it exited too fast, please check crash.log
echo Press any key to close this window.
pause >nul
