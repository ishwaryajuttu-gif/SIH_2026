@echo off
REM Optional (developers only): React dev server with hot reload on http://localhost:5173 (needs Node.js 20+).
REM run.bat must be running too. The demo itself does NOT need Node.js.
cd /d "%~dp0frontend"
if not exist node_modules call npm install
call npm run dev
pause
