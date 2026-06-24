@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"

python --version
if errorlevel 1 (
    echo 未检测到 Python，请先安装 Python 3.10 或 3.11。
    pause
    exit /b 1
)

python -m streamlit run app.py
pause
