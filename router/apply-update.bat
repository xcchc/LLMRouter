@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title Apply LLM Router Update

if not exist "LLMRouter.new.exe" (
    echo Update file not found: LLMRouter.new.exe
    echo Build or stage the new executable first.
    pause
    exit /b 1
)

echo ============================================
echo   LLM Router Manual Update
echo ============================================
echo This window will wait for LLM Router to exit.
echo It will replace the executable and preserve local stats.
echo It will NOT start LLM Router automatically.
echo.
echo Now exit LLM Router from the system tray.

set /a attempts=0

:wait_for_exit
tasklist /FI "IMAGENAME eq LLMRouter.exe" /NH | find /I "LLMRouter.exe" >nul
if not errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_for_exit
)

timeout /t 2 /nobreak >nul

:replace_old
tasklist /FI "IMAGENAME eq LLMRouter.exe" /NH | find /I "LLMRouter.exe" >nul
if not errorlevel 1 goto wait_for_exit

del /q "LLMRouter.previous.exe" 2>nul
move /y "LLMRouter.exe" "LLMRouter.previous.exe" >nul 2>nul
if errorlevel 1 (
    set /a attempts+=1
    if !attempts! GEQ 60 goto failed_locked
    if !attempts! EQU 1 echo Waiting for Windows to release the old executable...
    timeout /t 1 /nobreak >nul
    goto replace_old
)

copy /b /y "LLMRouter.new.exe" "LLMRouter.exe" >nul 2>nul
if errorlevel 1 goto restore_old

fc /b "LLMRouter.new.exe" "LLMRouter.exe" >nul 2>nul
if errorlevel 1 goto restore_old

del /q "LLMRouter.new.exe" 2>nul
del /q "LLMRouter.previous.exe" 2>nul
del /q "crash.log" 2>nul

echo.
echo Update completed successfully.
echo Please close this window, then manually start LLMRouter.exe.
pause
exit /b 0

:restore_old
del /q "LLMRouter.exe" 2>nul
move /y "LLMRouter.previous.exe" "LLMRouter.exe" >nul 2>nul
echo.
echo Update failed, and the old executable was restored.
echo LLMRouter.new.exe was kept for retry.
pause
exit /b 1

:failed_locked
echo.
echo Update failed because Windows kept the old executable locked for 60 seconds.
echo No files were changed. LLMRouter.new.exe was kept for retry.
pause
exit /b 1
