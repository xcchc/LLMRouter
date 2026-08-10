@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Build LLM Router

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
)

.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-build.txt
if errorlevel 1 goto :failed

.venv\Scripts\python.exe -m unittest discover -s tests
if errorlevel 1 goto :failed

.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean LLMRouter.spec
if errorlevel 1 goto :failed

copy /b /y "dist\LLMRouter.exe" "LLMRouter.new.exe" >nul
if errorlevel 1 goto :failed

echo.
echo Build completed: dist\LLMRouter.exe
echo Update package staged: LLMRouter.new.exe
echo Open the dashboard Settings -> Update, then click Update and restart.
exit /b 0

:failed
echo.
echo Build failed with code %errorlevel%.
exit /b %errorlevel%
