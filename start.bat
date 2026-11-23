@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem ===== 配置 =====
rem Ollama 可执行文件位置（便携版解压在 C 盘）
set OLLAMA_BIN=C:\Users\ASUS\ollama-portable\bin\ollama.exe

rem 1) 启动推理服务（如果已经在运行，这一句不会有影响）
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
if errorlevel 1 (
  echo 启动推理服务...
  start "ollama" /min "%OLLAMA_BIN%" serve
  timeout /t 3 >nul
) else (
  echo 推理服务已在运行
)

rem 2) 打开界面
start "" http://127.0.0.1:8420

rem 3) 启动助手服务（这个窗口关掉 = 助手停止）
echo.
echo   私人助手  ->  http://127.0.0.1:8420
echo   关闭本窗口即停止（后台的模型服务会继续留着）
echo.
python server.py

pause
