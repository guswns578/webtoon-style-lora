@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo 먼저 install.bat를 실행하세요.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -u "launcher.py"
if errorlevel 1 pause
