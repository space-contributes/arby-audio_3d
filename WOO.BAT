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
echo Checking for Python installation...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"$py=@('$env:ProgramFiles\Python312\python.exe','$env:LocalAppData\Programs\Python\Python312\python.exe') | Where-Object {Test-Path $_} | Select-Object -First 1; ^
if (-not $py) { ^
    Write-Host 'Python not found. Installing 3.12...'; ^
    Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe' -OutFile 'python_installer.exe'; ^
    Start-Process -Wait python_installer.exe -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1'; ^
    Remove-Item python_installer.exe; ^
    $py=@('$env:ProgramFiles\Python312\python.exe','$env:LocalAppData\Programs\Python\Python312\python.exe') | Where-Object {Test-Path $_} | Select-Object -First 1; ^
    if (-not $py) { Write-Host 'Python installation failed'; Read-Host 'Press Enter to continue'; exit 1 } ^
}; ^
Write-Host $py; ^
[System.Environment]::SetEnvironmentVariable('PYTHON_EXE_TEMP', $py, [System.EnvironmentVariableTarget]::User)"

:: Get Python path from environment variable
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "[System.Environment]::GetEnvironmentVariable('PYTHON_EXE_TEMP', [System.EnvironmentVariableTarget]::User)"') do set PYTHON_EXE=%%i

if "%PYTHON_EXE%"=="" (
    echo ❌ Failed to locate Python executable
    pause
    exit /b 1
)

echo Using Python: %PYTHON_EXE%

:: --- Install Python dependencies ---
echo Installing Python dependencies...
"%PYTHON_EXE%" -m ensurepip
"%PYTHON_EXE%" -m pip install --upgrade pip numpy scipy pydub

:: --- Download Python script ---
set PY_SCRIPT=%OUTPUT_DIR%\pa1112.py
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"if (-not (Test-Path '%PY_SCRIPT%')) { ^
    Write-Host 'Downloading Python script...'; ^
    try { ^
        Invoke-WebRequest -Uri 'https://tinyurl.com/ARBYAUDIO2' -OutFile '%PY_SCRIPT%' -ErrorAction Stop; ^
        Write-Host 'Script downloaded successfully'; ^
    } catch { ^
        Write-Host 'Failed to download script: ' $_; ^
        Read-Host 'Press Enter to continue'; ^
        exit 1; ^
    } ^
} else { ^
    Write-Host 'Script already exists'; ^
}"

:: --- Download music if URL ---
echo Processing music input...
echo %MUSIC_INPUT% | findstr /b /i "http" >nul
if %errorlevel%==0 (
    echo Downloading music from URL...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "try { ^
        Invoke-WebRequest -Uri '%MUSIC_INPUT%' -OutFile '%MUSIC_FILE%' -ErrorAction Stop; ^
        Write-Host 'Music downloaded successfully'; ^
    } catch { ^
        Write-Host 'Failed to download music: ' $_; ^
        Read-Host 'Press Enter to continue'; ^
        exit 1; ^
    }"
) else (
    if exist "%MUSIC_INPUT%" (
        echo Copying local music file...
        copy "%MUSIC_INPUT%" "%MUSIC_FILE%" >nul
        if %errorlevel% neq 0 (
            echo ❌ Failed to copy music file
            pause
            exit /b 1
        )
    ) else (
        echo ❌ WAV file not found: %MUSIC_INPUT%
        pause
        exit /b 1
    )
)

:: --- Handle optional video ---
set VIDEO_FILE=SKIPPED
set VIDEO_ARG=

if NOT "%VIDEO_INPUT%"=="" (
    echo Processing video input...
    echo %VIDEO_INPUT% | findstr /b /i "http" >nul
    if !errorlevel!==0 (
        set VIDEO_FILE=%OUTPUT_DIR%\input_video.mp4
        echo Downloading video from URL...
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "try { ^
            Invoke-WebRequest -Uri '%VIDEO_INPUT%' -OutFile '!VIDEO_FILE!' -ErrorAction Stop; ^
            Write-Host 'Video downloaded successfully'; ^
        } catch { ^
            Write-Host 'Failed to download video: ' $_; ^
            Write-Host 'Continuing without video...'; ^
            exit 0; ^
        }"
        
        if !errorlevel! neq 0 (
            set VIDEO_FILE=SKIPPED
            echo ⚠️ Video download failed. Continuing with audio only...
        ) else (
            set VIDEO_ARG=--video_file "!VIDEO_FILE!"
        )
    ) else (
        if exist "%VIDEO_INPUT%" (
            echo Copying local video file...
            for %%F in ("%VIDEO_INPUT%") do set VIDEO_FILENAME=%%~nxF
            set VIDEO_FILE=%OUTPUT_DIR%\!VIDEO_FILENAME!
            copy "%VIDEO_INPUT%" "!VIDEO_FILE!" >nul
            if !errorlevel! neq 0 (
                echo ⚠️ Failed to copy video file. Continuing with audio only...
                set VIDEO_FILE=SKIPPED
            ) else (
                set VIDEO_ARG=--video_file "!VIDEO_FILE!"
            )
        ) else (
            echo ⚠️ Video file not found: %VIDEO_INPUT%
            echo ⚠️ Continuing with audio only...
            set VIDEO_FILE=SKIPPED
        )
    )
) else (
    echo No video input provided. Processing audio only...
)

:: --- Status ---
echo.
echo ========================================
echo PROCESSING STATUS:
echo ========================================
echo ✅ Music file: %MUSIC_FILE%
if "%VIDEO_FILE%"=="SKIPPED" (
    echo ⚠️ No video file - generating audio only
) else (
    echo ✅ Video file: !VIDEO_FILE!
)
echo ========================================
echo.

:: --- Run Python ---
echo Starting audio processing...
if "%VIDEO_FILE%"=="SKIPPED" (
    "%PYTHON_EXE%" "%PY_SCRIPT%" --music_url "%MUSIC_FILE%"
) else (
    "%PYTHON_EXE%" "%PY_SCRIPT%" --music_url "%MUSIC_FILE%" !VIDEO_ARG!
)

if %errorlevel% neq 0 (
    echo ❌ Python script execution failed
    pause
    exit /b 1
)

echo.
echo ✅ Processing completed successfully!
echo Check the output folder: %OUTPUT_DIR%
pause
