@echo off
REM ============================================================
REM  BAS-HAR one-time setup (Windows). Needs Python 3.10 - 3.12 and internet ONCE.
REM  After this, the demo runs fully offline.
REM ============================================================
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && (py -3.11 -V >nul 2>nul && set PY=py -3.11)
echo Using: %PY%
%PY% -V || (echo Python not found - install Python 3.11 from python.org & pause & exit /b 1)

if not exist .venv (
  echo Creating virtual environment...
  %PY% -m venv .venv || goto :err
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r backend\requirements.txt || goto :err

echo.
echo Downloading model files into backend\models (one time)...
python backend\scripts\download_models.py || goto :err

echo.
echo Verifying the installation...
python backend\scripts\verify_setup.py
echo.
echo  Setup finished. Fix any FAIL lines above, then start the demo with run.bat
pause
exit /b 0
:err
echo.
echo  Setup FAILED - see the error above.
pause
exit /b 1
