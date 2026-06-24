@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"

set "PORT=8501"
set "URL=http://localhost:%PORT%"

where py > nul 2> nul
if not errorlevel 1 (
    set "PY=py -3"
) else (
    where python > nul 2> nul
    if not errorlevel 1 (
        set "PY=python"
    ) else (
        echo Python was not found in PATH.
        echo Please install Python 3.10+ or add Python to PATH, then run this file again.
        pause
        exit /b 1
    )
)

echo Using Python:
%PY% --version
echo.

%PY% -m streamlit version > nul 2> nul
if errorlevel 1 (
    echo Streamlit is not installed in the selected Python environment.
    echo Run this command first:
    echo pip install -r requirements.txt
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not errorlevel 1 (
    echo A local service is already running on %URL%.
    echo Opening the existing page...
    start "" "%URL%"
    pause
    exit /b 0
)

echo Starting Streamlit on %URL%
echo Keep this window open while using the website.
start "" powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 3; Start-Process '%URL%'"

%PY% -m streamlit run app.py --server.port %PORT% --server.headless false

echo.
echo Streamlit has stopped. If there was an error, read the message above.
pause
