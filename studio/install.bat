@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto packages
py -3.11 -m venv ".venv"
if errorlevel 1 (
 echo Python 3.11을 먼저 설치하세요. 설치할 때 Python Launcher도 포함해야 합니다.
 pause
 exit /b 1
)
:packages
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 (
 echo 설치에 실패했습니다. 인터넷 연결과 위 오류 메시지를 확인하세요.
 pause
 exit /b 1
)
echo 설치가 끝났습니다. run.bat를 실행하세요.
pause
