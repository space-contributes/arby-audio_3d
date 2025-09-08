@echo off
REM ========================================
REM 3D Audio Processing Launcher (Windows)
REM ========================================

REM --- Prompt for WAV file URL or local path ---
set /p MUSIC_INPUT="Enter full path to WAV file or URL: "

REM --- Check if Python is installed ---
where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo Python not found! Installing Python 3.12...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe' -OutFile 'python_installer.exe'"
    python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    IF %ERRORLEVEL% NEQ 0 (
        echo Python installation failed. Please install manually.
        pause
        exit /b 1
    )
    del python_installer.exe
    echo Python installed successfully.
)

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

REM --- Check/install Python dependencies ---
python -m pip show numpy >nul 2>nul || python -m pip install numpy
python -m pip show scipy >nul 2>nul || python -m pip install scipy
python -m pip show pydub >nul 2>nul || python -m pip install pydub

REM --- Run Python script ---
echo Running 3D audio processing...
python "%PY_SCRIPT%" --music_url "%MUSIC_FILE%"

echo Done.
pause
