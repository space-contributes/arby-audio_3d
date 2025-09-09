@echo off
REM ========================================
REM 3D Audio Processing Launcher (Windows)
REM ========================================
setlocal enabledelayedexpansion

REM --- Prompt user for output folder ---
set /p OUTPUT_DIR="Enter full path for output folder (will be created if it doesn't exist): "
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)
echo Files will be saved to: %OUTPUT_DIR%
echo.

REM --- Check for admin rights ---
net session >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Please run this script as Administrator!
    pause
    exit /b 1
)

REM --- Prompt for WAV file URL or local path ---
set /p MUSIC_INPUT="Enter full path to WAV file or URL: "

REM --- Prompt for optional video file ---
set /p VIDEO_INPUT="Enter full path to video file or URL (optional, press Enter to skip): "

REM --- Detect Python ---
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
        echo Python installation almost done! Re-run this script, or install manually if it still fails.
        pause
        exit /b 1
    )
    echo Python installed successfully.
)

REM --- Ensure pip is upgraded ---
"%PYTHON_EXE%" -m ensurepip
"%PYTHON_EXE%" -m pip install --upgrade pip

REM --- Install required Python dependencies ---
"%PYTHON_EXE%" -m pip install --upgrade numpy scipy pydub

REM --- Download or verify Python script ---
set PY_SCRIPT=%OUTPUT_DIR%\pa1112.py
IF NOT EXIST "%PY_SCRIPT%" (
    echo Python script not found, downloading...
    powershell -Command "Invoke-WebRequest -Uri 'https://shorturl.at/i45Tk' -OutFile '%PY_SCRIPT%'"
    IF %ERRORLEVEL% NEQ 0 (
        echo Failed to download Python script.
        pause
        exit /b 1
    )
)

REM --- Handle music input (URL or local) ---
set MUSIC_FILE=%OUTPUT_DIR%\music_input.wav
echo %MUSIC_INPUT% | findstr /b /i "http" >nul
if %errorlevel%==0 (
    echo Downloading WAV file...
    powershell -Command "Invoke-WebRequest -Uri '%MUSIC_INPUT%' -OutFile '%MUSIC_FILE%'"
    if not exist "%MUSIC_FILE%" (
        echo ❌ Failed to download WAV file.
        pause
        exit /b 1
    )
) else (
    if exist "%MUSIC_INPUT%" (
        copy "%MUSIC_INPUT%" "%MUSIC_FILE%"
    ) else (
        echo ❌ WAV file not found: %MUSIC_INPUT%
        pause
        exit /b 1
    )
)

REM --- Handle video input (optional) ---
set VIDEO_FILE=
if defined VIDEO_INPUT (
    echo %VIDEO_INPUT% | findstr /b /i "http" >nul
    if %errorlevel%==0 (
        set VIDEO_FILE=%OUTPUT_DIR%\input_video.mp4
        echo Downloading video file...
        powershell -Command "Invoke-WebRequest -Uri '%VIDEO_INPUT%' -OutFile '%VIDEO_FILE%'"
        if not exist "%VIDEO_FILE%" (
            echo ❌ Failed to download video file.
            pause
            exit /b 1
        )
    ) else (
        if exist "%VIDEO_INPUT%" (
            copy "%VIDEO_INPUT%" "%OUTPUT_DIR%\"
            set VIDEO_FILE=%OUTPUT_DIR%\%~nxVIDEO_INPUT%
        ) else (
            echo ⚠️ Video file not found: %VIDEO_INPUT%. Skipping video...
            set VIDEO_FILE=
        )
    )
)

echo.
echo ✅ Music file: %MUSIC_FILE%
if not "%VIDEO_FILE%"=="" (
    echo ✅ Video file: %VIDEO_FILE%
) else (
    echo ⚠️ No video file provided; only generating audio.
)

REM --- Run Python script ---
echo Running 3D audio processing...
"%PYTHON_EXE%" "%PY_SCRIPT%" --music_url "%MUSIC_FILE%" %VIDEO_FILE%

echo Done.
pause
