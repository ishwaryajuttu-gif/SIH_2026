@echo off
REM Unit + integration tests, then a real-server smoke test on the sample video.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
pip install pytest -q
cd backend
python -m pytest -q
python scripts\smoke_test.py
pause
