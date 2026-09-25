@echo off
REM Pre-demo check. "verify.bat live" also runs 15 s of real webcam frames through YOLO + MediaPipe.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (echo Run setup.bat first. & pause & exit /b 1)
call .venv\Scripts\activate.bat
if /i "%~1"=="live" (
  python backend\scripts\verify_setup.py --live
) else (
  python backend\scripts\verify_setup.py
)
pause
