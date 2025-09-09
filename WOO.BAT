@echo off
REM ========================================
REM 3D Audio Processing Launcher (Windows)
REM Single-folder workflow with music URL
REM Warn if no video, ask to proceed
REM ========================================

setlocal enabledelayedexpansion

:: --- Ask for folder path ---
set /p FOLDER_PATH="Enter full path for folder for downloads and outputs: "
if "%FOLDER_PATH%"=="" set FOLDER_PATH=%cd%\output
if not exist "%FOLDER_PATH%" mkdir "%FOLDER_PATH%"
echo Using folder: %FOLDER_PATH%
echo.

:: --- Ask for music URL ---
set /p MUSIC_URL="Enter URL to WAV music file: "
if "%MUSIC_URL%"=="" (
    echo ❌ Music URL is required!
    pause
    exit /b 1
)
set MUSIC_FILE=%FOLDER_PATH%\music_input.wav

:: --- Detect optional video file in folder ---
set VIDEO_FILE=
for %%V in ("%FOLDER_PATH%\*.mp4" "%FOLDER_PATH%\*.mov" "%FOLDER_PATH%\*.mkv") do (
    set VIDEO_FILE=%%V
    goto :FOUND_VIDEO
)
:FOUND_VIDEO
if "%VIDEO_FILE%"=="" (
    echo ⚠️ No video file found in folder.
    set /p PROCEED="Do you want to proceed without a video file? (y/n): "
    if /i NOT "%PROCEED%"=="y" (
        echo Exiting as requested.
        pause
        exit /b 1
    )
) else (
    echo ✅ Video file found: %VIDEO_FILE%
)

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
set PY_SCRIPT=%FOLDER_PATH%\pa1112.py
IF NOT EXIST "%PY_SCRIPT%" (
    echo Downloading Python script...
    powershell -Command "Invoke-WebRequest -Uri 'https://tinyurl.com/ARBYAUDIO2' -OutFile '%PY_SCRIPT%'"
    IF %ERRORLEVEL% NEQ 0 (
        echo ❌ Failed to download Python script.
        pause
        exit /b 1
    )
)

:: --- Download music file ---
echo Downloading music file from URL...
powershell -Command "Invoke-WebRequest -Uri '%MUSIC_URL%' -OutFile '%MUSIC_FILE%'"
IF NOT EXIST "%MUSIC_FILE%" (
    echo ❌ Failed to download music file.
    pause
    exit /b 1
)

:: --- Run Python script ---
echo Running 3D audio processing...
if "%VIDEO_FILE%"=="" (
    "%PYTHON_EXE%" "%PY_SCRIPT%" --music_file "%MUSIC_FILE%"
) else (
    "%PYTHON_EXE%" "%PY_SCRIPT%" --music_file "%MUSIC_FILE%" --video_file "%VIDEO_FILE%"
)

echo ✅ Done. All outputs are stored in: %FOLDER_PATH%
pause
