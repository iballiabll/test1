@echo off
chcp 65001 > nul
setlocal
set PYTHONUTF8=1

cd /d "%~dp0"

echo =============================================
echo    智能交通检测系统 - 比赛版
echo =============================================
echo.
echo 已拆分为两个脚本：
echo   1. 一键部署环境.bat
echo   2. 一键启动系统.bat
echo.

call "%~dp0一键启动系统.bat"
