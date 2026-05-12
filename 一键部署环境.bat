@echo off
chcp 65001 > nul
setlocal
set PYTHONUTF8=1

cd /d "%~dp0"

echo =============================================
echo    路眼 - 一键部署环境
echo =============================================
echo.

where python > nul 2> nul
if errorlevel 1 (
    echo [错误] 未找到 Python。请先安装 Python 3.10 或更高版本，并勾选 Add Python to PATH。
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] 创建项目虚拟环境 .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败。
        pause
        exit /b 1
    )
) else (
    echo [1/4] 已检测到 .venv，跳过创建。
)

echo.
echo [2/4] 升级 pip / setuptools / wheel ...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [错误] pip 基础工具升级失败，请检查网络或 Python 环境。
    pause
    exit /b 1
)

echo.
echo [3/4] 安装项目依赖 ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败。GPU 版 PyTorch 可按显卡 CUDA 版本单独安装后重试。
    pause
    exit /b 1
)

echo.
echo [4/4] 验证环境与模型 ...
".venv\Scripts\python.exe" scripts\verify_environment.py --load-model
if errorlevel 1 (
    echo [错误] 环境验证失败，请根据上方提示修复。
    pause
    exit /b 1
)

echo.
echo =============================================
echo 环境部署完成。后续启动请运行：一键启动系统.bat
echo =============================================
if /I "%SKIP_FINAL_PAUSE%"=="1" exit /b 0
pause
