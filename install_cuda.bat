@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [!] venv not found. Run install.bat first.
    pause
    exit /b 1
)
echo [*] Replacing CPU torch with CUDA build (about 2.5 GB download)...
"venv\Scripts\python.exe" -m pip uninstall -y torch
"venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [!] Failed. Your NVIDIA driver may be too old; update it and retry.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo.
echo [OK] Done. Restart run.bat
pause
