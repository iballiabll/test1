@echo off
chcp 65001 > nul
setlocal
set PYTHONUTF8=1

cd /d "%~dp0"

echo =============================================
echo    路眼 - 智能交通检测系统
echo =============================================
echo.

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    where python > nul 2> nul
    if errorlevel 1 (
        echo [错误] 未找到 Python，且项目 .venv 不存在。
        echo 请先运行：一键部署环境.bat
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

echo [1/2] 快速检查环境 ...
%PYTHON_EXE% scripts\verify_environment.py
if errorlevel 1 (
    echo.
    echo [提示] 环境不完整，自动执行一键部署环境...
    set SKIP_FINAL_PAUSE=1
    call "%~dp0一键部署环境.bat"
    if errorlevel 1 (
        echo [错误] 自动部署失败，请根据上方提示处理。
        pause
        exit /b 1
    )
    set SKIP_FINAL_PAUSE=
    set "PYTHON_EXE=.venv\Scripts\python.exe"
)

echo.
echo [2/2] 启动系统 ...
%PYTHON_EXE% "智能交通检测系统_比赛版.py"

echo.
echo 程序已退出。
pause
