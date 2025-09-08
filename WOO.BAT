@echo off
REM ========================================
REM 3D Audio Processing Launcher (Windows)
REM ========================================

REM --- Check for admin rights ---
net session >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Please run this script as Administrator!
    pause
    exit /b 1
)

REM --- Prompt for WAV file URL or local path ---
set /p MUSIC_INPUT="Enter full path to WAV file or URL: "

REM --- Function to detect Python ---
set PYTHON_EXE=

for %%P in (
    "%ProgramFiles%\Python312\python.exe"
    "%LocalAppData%\Programs\Python\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%LocalAppData%\Programs\Python\Python311\python.exe"
) do (
    if exist "%%P" set PYTHON_EXE=%%P
)

REM --- Install Python if not found ---
IF "%PYTHON_EXE%"=="" (
    echo Python not found! Installing Python 3.12...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe' -OutFile 'python_installer.exe'"
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    del python_installer.exe

    REM Try detecting Python again
    for %%P in (
        "%ProgramFiles%\Python312\python.exe"
        "%LocalAppData%\Programs\Python\Python312\python.exe"
    ) do (
        if exist "%%P" set PYTHON_EXE=%%P
    )

    IF "%PYTHON_EXE%"=="" (
        echo Python installation failed. Please install manually and re-run this script.
        pause
        exit /b 1
    )
    echo Python installed successfully.
)

REM --- Ensure pip is upgraded ---
"%PYTHON_EXE%" -m ensurepip
"%PYTHON_EXE%" -m pip install --upgrade pip

REM --- Install required Python dependencies ---
"%PYTHON_EXE%" -m pip install numpy scipy pydub

REM --- Download the WAV file if it's a URL ---
echo %MUSIC_INPUT% | findstr /b "http" >nul
IF %ERRORLEVEL% EQU 0 (
    set MUSIC_FILE=music_input.wav
    echo Downloading WAV file...
    powershell -Command "Invoke-WebRequest -Uri '%MUSIC_INPUT%' -OutFile '%MUSIC_FILE%'"
    IF %ERRORLEVEL% NEQ 0 (
        echo Failed to download WAV file.
        pause
        exit /b 1
    )
) ELSE (
    set MUSIC_FILE=%MUSIC_INPUT%
    IF NOT EXIST "%MUSIC_FILE%" (
        echo File not found!
        pause
        exit /b 1
    )
)

REM --- Check if Python script exists, if not download ---
set PY_SCRIPT=PYTHONSCRIPT.py
IF NOT EXIST "%PY_SCRIPT%" (
    echo Python script not found, downloading...
    powershell -Command "Invoke-WebRequest -Uri 'https://drive.usercontent.google.com/download?id=1NYLVw1kMjRypD2QG6FPLtxArFiLQdnsX&export=download' -OutFile '%PY_SCRIPT%'"
    IF %ERRORLEVEL% NEQ 0 (
        echo Failed to download Python script.
        pause
        exit /b 1
    )
)

REM --- Run Python script ---
echo Running 3D audio processing...
"%PYTHON_EXE%" "%PY_SCRIPT%" --music_url "%MUSIC_FILE%"

echo Done.
pause
