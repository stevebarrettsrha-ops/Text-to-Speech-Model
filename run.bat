@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Script Builder

rem Flat control flow on purpose. Nested parenthesised blocks with delayed
rem expansion are where batch fails silently, so every branch is a label.

call :findpy
if defined PY goto haspy

echo.
echo   Python 3.10 or newer was not found.
echo.

where winget >nul 2>nul
if errorlevel 1 goto manualpy

echo   WinGet is here, so this can be done for you: the Python install manager
echo   from the Microsoft Store, then Python 3.13 through it.
echo.
choice /c YN /n /m "  Install Python now? [Y/N] "
if errorlevel 2 goto declined

echo.
echo   Installing the Python install manager...
winget install 9NQ7512CXL7T --accept-package-agreements --accept-source-agreements
echo.
echo   Installing Python 3.13...
py install 3.13

rem A window reads PATH once, when it opens, so a py that has just been put
rem there is invisible here. Re-test anyway - it does work when the manager was
rem already present and only the runtime was missing.
call :findpy
if defined PY goto haspy
goto restartneeded

:manualpy
echo   Install it from https://www.python.org/downloads/ and tick
echo   "Add python.exe to PATH" during setup, then run this file again.
echo.
pause
exit /b 1

:declined
echo.
echo   Nothing was installed. Get Python from
echo   https://www.python.org/downloads/ and run this file again.
echo.
pause
exit /b 1

:restartneeded
echo.
echo   Python is installed, but this window cannot see it yet.
echo   Close it and run run.bat again.
echo.
pause
exit /b 1

:haspy
echo   Using: %PY%

rem Script Builder's own packages go into a virtual environment beside this
rem file, so the Python that was found is left as it was found.
if exist ".venv\Scripts\python.exe" goto hasvenv
echo   Setting up Script Builder's packages (first run only)...
%PY% -m venv .venv
if errorlevel 1 goto novenv

:hasvenv
".venv\Scripts\python.exe" -c "import sys" >nul 2>nul
if errorlevel 1 goto novenv
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --quiet -r requirements.txt
if errorlevel 1 goto pipfail
".venv\Scripts\python.exe" server.py
pause
exit /b 0

rem No usable environment could be built. Windows Pythons are not marked
rem externally managed, so installing into the one we found still works.
:novenv
echo   Could not build a separate environment; using %PY% as it is.
%PY% -m pip install --disable-pip-version-check --quiet -r requirements.txt
if errorlevel 1 goto pipfail
%PY% server.py
pause
exit /b 0

:pipfail
echo   Could not install the Python packages. Check your internet connection.
pause
exit /b 1

rem --------------------------------------------------------------------- rem
rem Sets PY to the first interpreter that is really 3.10+. Tested by running
rem each one: Microsoft Store stubs answer `where` and then fail on execution.
:findpy
set "PY="
for %%C in ("py -3.13" "py -3.12" "py -3.11" "py -3.10" "py -3" "python") do (
  if not defined PY (
    %%~C -c "import sys;raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if !errorlevel! equ 0 set "PY=%%~C"
  )
)
exit /b 0
