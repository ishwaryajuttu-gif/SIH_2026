@echo off
REM ============================================================
REM  Start BAS-HAR:  backend starts -> waits until ready -> browser opens.
REM    run.bat          webcam from config.yaml (default 0)
REM    run.bat 1        use webcam 1
REM    run.bat file     use the sample video (fallback only)
REM  Press Ctrl+C in this window to stop.
REM ============================================================
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (echo Run setup.bat first. & pause & exit /b 1)
call .venv\Scripts\activate.bat
if "%~1"=="" (
  python backend\scripts\launch.py
) else if /i "%~1"=="file" (
  python backend\scripts\launch.py --video data/videos/hand_demo.mp4
) else (
  python backend\scripts\launch.py --camera %1
)
pause
