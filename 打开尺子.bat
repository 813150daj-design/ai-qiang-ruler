@echo off
cd /d "%~dp0"
where pyw >nul 2>nul
if not errorlevel 1 (
    start "" pyw -3 ruler_gui.py
    if not errorlevel 1 exit /b 0
)
where pythonw >nul 2>nul
if not errorlevel 1 (
    start "" pythonw ruler_gui.py
    if not errorlevel 1 exit /b 0
)
python ruler_gui.py
if errorlevel 1 pause
