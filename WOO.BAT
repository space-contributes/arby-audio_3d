@echo off
REM ========================================
REM 3D Audio Processing Launcher (Windows)
REM True Hybrid Batch + Partial PowerShell
REM ========================================

setlocal enabledelayedexpansion

:: --- Prompt for output folder ---
set /p OUTPUT_DIR="Enter full path for output folder (will be created if it doesn't exist): "
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
echo Files will be saved to: %OUTPUT_DIR%
echo.

:: --- Prompt for WAV file ---
set /p MUSIC_INPUT="Enter full path to WAV file or URL: "
if "%MUSIC_INPUT%"=="" (
    echo ❌ Music file is required!
    pause
    exit /b 1
)
set MUSIC_FILE=%OUTPUT_DIR%\music_input.wav

:: --- Prompt for optional video ---
set /p VIDEO_INPUT="Enter full path to video file or URL (optional, press Enter to skip): "

:: --- Check/download Python using PowerShell ---
powershell -NoProfile -Command ^
$py=@("$env:ProgramFiles\Python312\python.exe","$env:LocalAppData\Programs\Python\Python312\python.exe") | Where-Object {Test-Path $_} | Select-Object -First 1; ^
if (-not $py) { ^
    Write-Host "Python not found. Installing 3.12..."; ^
    Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe' -OutFile 'python_installer.exe'; ^
    Start-Process -Wait python_installer.exe -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1'; ^
    Remove-Item python_installer.exe; ^
    $py=@("$env:ProgramFiles\Python312\python.exe","$env:LocalAppData\Programs\Python\Python312\python.exe") | Where-Object {Test-Path $_} | Select-Object -First 1; ^
    if (-not $py) { Write-Host "Python installation failed"; pause; exit 1 } ^
}; ^
Write-Host $py; ^
$env:PYTHON_EXE=$py; exit 0

:: --- Install Python dependencies ---
"%PYTHON_EXE%" -m ensurepip
"%PYTHON_EXE%" -m pip install --upgrade pip numpy scipy pydub

:: --- Download Python script ---
set PY_SCRIPT=%OUTPUT_DIR%\pa1112.py
powershell -NoProfile -Command ^
if (-not (Test-Path '%PY_SCRIPT%')) { ^
    Write-Host "Downloading Python script..."; ^
    Invoke-WebRequest -Uri 'https://tinyurl.com/ARBYAUDIO2' -OutFile '%PY_SCRIPT%' -ErrorAction Stop; ^
}

:: --- Download music if URL ---
echo %MUSIC_INPUT% | findstr /b /i "http" >nul
if %errorlevel%==0 (
    powershell -NoProfile -Command ^
    Invoke-WebRequest -Uri '%MUSIC_INPUT%' -OutFile '%MUSIC_FILE%' -ErrorAction Stop ^
) else (
    if exist "%MUSIC_INPUT%" (
        copy "%MUSIC_INPUT%" "%MUSIC_FILE%" >nul
    ) else (
        echo ❌ WAV file not found: %MUSIC_INPUT%
        pause
        exit /b 1
    )
)

:: --- Handle optional video ---
set VIDEO_FILE=SKIPPED
if not "%VIDEO_INPUT%"=="" (
    echo %VIDEO_INPUT% | findstr /b /i "http" >nul
    if %errorlevel%==0 (
        set VIDEO_FILE=%OUTPUT_DIR%\input_video.mp4
        powershell -NoProfile -Command ^
        try { Invoke-WebRequest -Uri '%VIDEO_INPUT%' -OutFile '%VIDEO_FILE%' -ErrorAction Stop } ^
        catch { Write-Host "❌ Failed to download video"; pause; exit 1 }
    ) else (
        if exist "%VIDEO_INPUT%" (
            copy "%VIDEO_INPUT%" "%OUTPUT_DIR%" >nul
            set VIDEO_FILE=%OUTPUT_DIR%\%~nxVIDEO_INPUT%
        ) else (
            echo ⚠️ Video file not found. Skipping...
        )
    )
)

:: --- Status ---
echo ✅ Music file: %MUSIC_FILE%
if "%VIDEO_FILE%"=="SKIPPED" (
    echo ⚠️ No video file provided; only generating audio.
) else (
    echo ✅ Video file: %VIDEO_FILE%
)

:: --- Run Python ---
if "%VIDEO_FILE%"=="SKIPPED" (
    "%PYTHON_EXE%" "%PY_SCRIPT%" --music_url "%MUSIC_FILE%"
) else (
    "%PYTHON_EXE%" "%PY_SCRIPT%" --music_url "%MUSIC_FILE%" --video_file "%VIDEO_FILE%"
)

echo ✅ Done.
pause
