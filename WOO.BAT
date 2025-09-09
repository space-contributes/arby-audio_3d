@echo off
REM ========================================
REM 3D Audio Processing Launcher (Windows)
REM Single-folder workflow with music URL
REM ========================================

setlocal enabledelayedexpansion

:: --- Ask for folder path ---
set /p FOLDER_PATH="Enter full path to folder for downloads and outputs: "
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

:: --- Detect optional video file ---
set VIDEO_FILE=
for %%V in ("%FOLDER_PATH%\*.mp4" "%FOLDER_PATH%\*.mov" "%FOLDER_PATH%\*.mkv") do (
    set VIDEO_FILE=%%V
    goto :FOUND_VIDEO
)
:FOUND_VIDEO
if "%VIDEO_FILE%"=="" (
    echo ⚠️ No video file found; will only generate audio.
) else (
    echo ✅ Video file found: %VIDEO_FILE%
)

:: --- PowerShell: Python detection, install, dependencies, downloads ---
powershell -NoProfile -Command ^
$ErrorActionPreference='Stop'; ^
$py=@("$env:ProgramFiles\Python312\python.exe","$env:LocalAppData\Programs\Python\Python312\python.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1; ^
if (-not $py) { ^
    Write-Host "Python 3.12 not found. Installing..."; ^
    Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe' -OutFile 'python_installer.exe'; ^
    Start-Process -Wait python_installer.exe -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1'; ^
    Remove-Item python_installer.exe; ^
    $py=@("$env:ProgramFiles\Python312\python.exe","$env:LocalAppData\Programs\Python\Python312\python.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1; ^
    if (-not $py) { Write-Host "Python installation failed"; pause; exit 1 } ^
}; ^
$env:PYTHON_EXE=$py; ^
Write-Host "Python found at $py"; ^
& $py -m ensurepip; ^
& $py -m pip install --upgrade pip numpy scipy pydub; ^

:: --- Download Python script ---
$PY_SCRIPT=Join-Path '%FOLDER_PATH%' 'pa1112.py'; ^
if (-not (Test-Path $PY_SCRIPT)) { ^
    Write-Host "Downloading Python script..."; ^
    Invoke-WebRequest -Uri 'https://tinyurl.com/ARBYAUDIO2' -OutFile $PY_SCRIPT; ^
}

:: --- Download music from URL ---
Write-Host "Downloading music file from URL..."; ^
try { Invoke-WebRequest -Uri '%MUSIC_URL%' -OutFile '%MUSIC_FILE%' -ErrorAction Stop } ^
catch { Write-Host '❌ Failed to download music file'; pause; exit 1 } ^

exit 0

:: --- Run Python script ---
if "%VIDEO_FILE%"=="" (
    "%PYTHON_EXE%" "%FOLDER_PATH%\pa1112.py" --music_url "%MUSIC_FILE%"
) else (
    "%PYTHON_EXE%" "%FOLDER_PATH%\pa1112.py" --music_url "%MUSIC_FILE%" --video_file "%VIDEO_FILE%"
)

echo ✅ Done. Output stored in folder: %FOLDER_PATH%
pause
