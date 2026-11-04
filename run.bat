@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "NO_PROXY=localhost,127.0.0.1,0.0.0.0,::1"
set "no_proxy=localhost,127.0.0.1,0.0.0.0,::1"
set "ANIMA_RUNNER=1"
if not exist "venv\Scripts\python.exe" (
    echo [!] venv not found. Run install.bat first.
    pause
    exit /b 1
)
:start
"venv\Scripts\python.exe" app.py
if %errorlevel%==3 (
    rem language switch: restart without opening a new browser tab
    set "ANIMA_NO_BROWSER=1"
    goto start
)
pause
