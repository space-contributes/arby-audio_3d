@echo off
REM ========================================
REM 3D Audio Processing Launcher (Windows)
REM Fully rewritten with updated Python script link
REM ========================================
setlocal enabledelayedexpansion

:: --- Prompt for output folder ---
set /p OUTPUT_DIR="Enter full path for output folder (will be created if it doesn't exist): "
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)
echo Files will be saved to: %OUTPUT_DIR%
echo.

:: --- Check for admin rights ---
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Please run this script as Administrator!
    pause
    exit /b 1
)

:: --- Prompt for WAV file URL or local path ---
set /p MUSIC_INPUT="Enter full path to WAV file or URL: "
if "%MUSIC_INPUT%"=="" (
    echo ❌ Music file is required!
    pause
    exit /b 1
)
set MUSIC_FILE=%OUTPUT_DIR%\music_input.wav

:: --- Prompt for optional video file ---
set /p VIDEO_INPUT="Enter full path to video file or URL (optional, press Enter to skip): "

:: --- Detect Python 3 ---
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
if "%PYTHON_EXE%"=="" (
    echo Python not found! Installing Python 3.12...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe' -OutFile 'python_installer.exe'"
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    del python_installer.exe

    :: Try detecting Python again
    for %%P in (
        "%ProgramFiles%\Python312\python.exe"
        "%LocalAppData%\Programs\Python\Python312\python.exe"
    ) do (
        if exist "%%P" set PYTHON_EXE=%%P
    )

    if "%PYTHON_EXE%"=="" (
        echo Python installation almost done! Re-run this script or install manually.
        pause
        exit /b 1
    )
    echo Python installed successfully.
)

:: --- Ensure pip is upgraded ---
"%PYTHON_EXE%" -m ensurepip
"%PYTHON_EXE%" -m pip install --upgrade pip

:: --- Install Python dependencies ---
"%PYTHON_EXE%" -m pip install --upgrade numpy scipy pydub

:: --- Download Python script ---
set PY_SCRIPT=%OUTPUT_DIR%\pa1112.py
if not exist "%PY_SCRIPT%" (
    echo Python script not found, downloading...
    powershell -Command "Invoke-WebRequest -Uri 'https://tinyurl.com/ARBYAUDIO2' -OutFile '%PY_SCRIPT%'"
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Failed to download Python script.
        pause
        exit /b 1
    )
)

:: --- Handle music input ---
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
        copy "%MUSIC_INPUT%" "%MUSIC_FILE%" >nul
    ) else (
        echo ❌ WAV file not found: %MUSIC_INPUT%
        pause
        exit /b 1
    )
)

:: --- Handle video input (optional) ---
set VIDEO_FILE=SKIPPED
if not "%VIDEO_INPUT%"=="" (
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
            copy "%VIDEO_INPUT%" "%OUTPUT_DIR%" >nul
            set VIDEO_FILE=%OUTPUT_DIR%\%~nxVIDEO_INPUT%
        ) else (
            echo ⚠️ Video file not found: %VIDEO_INPUT%. Skipping video...
            set VIDEO_FILE=SKIPPED
        )
    )
)

:: --- Show status ---
echo.
echo ✅ Music file: %MUSIC_FILE%
if "%VIDEO_FILE%"=="SKIPPED" (
    echo ⚠️ No video file provided; only generating audio.
) else (
    echo ✅ Video file: %VIDEO_FILE%
)

:: --- Run Python script ---
echo Running 3D audio processing...
if "%VIDEO_FILE%"=="SKIPPED" (
    "%PYTHON_EXE%" "%PY_SCRIPT%" --music_url "%MUSIC_FILE%"
) else (
    "%PYTHON_EXE%" "%PY_SCRIPT%" --music_url "%MUSIC_FILE%" --video_file "%VIDEO_FILE%"
)

echo ✅ Done.
pause
