@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [!] Python not found. Install Python 3.10-3.12 from python.org ^(check "Add to PATH"^).
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo [*] Creating venv...
    %PY% -m venv venv
    if errorlevel 1 goto :err
)

set "VPY=venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :err

echo [*] Installing torch (CPU build)...
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo [!] CPU index unavailable, trying default PyPI torch...
    "%VPY%" -m pip install torch
    if errorlevel 1 goto :err
)

echo [*] Installing gradio and safetensors...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :err

echo.
echo [OK] Done. Start with run.bat
pause
exit /b 0

:err
echo.
echo [!] Installation failed. See messages above.
pause
exit /b 1
