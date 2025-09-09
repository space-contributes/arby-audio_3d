@echo off
REM ========================================
REM 3D Audio Processing Launcher (Wrapper)
REM ========================================

:: Check if PowerShell is available
powershell -Command "Write-Host 'PowerShell Available'" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ PowerShell is not available on this system
    echo This tool requires PowerShell to run
    pause
    exit /b 1
)

:: Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
set PS_SCRIPT=%SCRIPT_DIR%AudioProcessor.ps1

:: Create the PowerShell script if it doesn't exist
if not exist "%PS_SCRIPT%" (
    echo Creating PowerShell script...
    call :CREATE_PS1
)

:: Run the PowerShell script
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%"

:: Check if PowerShell script executed successfully
if %errorlevel% neq 0 (
    echo.
    echo ❌ PowerShell script execution failed
    pause
    exit /b %errorlevel%
)

echo.
echo ✅ Execution completed
pause
goto :EOF

:CREATE_PS1
(
echo # ========================================
echo # 3D Audio Processing PowerShell Script
echo # ========================================
echo.
echo param^(
echo     [string]$OutputDir,
echo     [string]$MusicInput,
echo     [string]$VideoInput
echo ^)
echo.
echo # Function to write colored output
echo function Write-Status {
echo     param^([string]$Message, [string]$Type = "Info"^)
echo     switch ^($Type^) {
echo         "Success" { Write-Host "✅ $Message" -ForegroundColor Green }
echo         "Warning" { Write-Host "⚠️ $Message" -ForegroundColor Yellow }
echo         "Error" { Write-Host "❌ $Message" -ForegroundColor Red }
echo         "Info" { Write-Host "ℹ️ $Message" -ForegroundColor Cyan }
echo         default { Write-Host $Message }
echo     }
echo }
echo.
echo # Function to prompt for input with validation
echo function Get-UserInput {
echo     param^([string]$Prompt, [bool]$Required = $true^)
echo     do {
echo         $input = Read-Host $Prompt
echo         if ^(-not $Required -or -not [string]::IsNullOrWhiteSpace^($input^)^) {
echo             return $input
echo         }
echo         Write-Status "This field is required!" "Error"
echo     } while ^($true^)
echo }
echo.
echo # Function to check if path is URL
echo function Test-IsUrl {
echo     param^([string]$Path^)
echo     return $Path -match '^https?://'
echo }
echo.
echo # Function to download file
echo function Get-FileFromUrl {
echo     param^([string]$Url, [string]$OutputPath^)
echo     try {
echo         Write-Status "Downloading from $Url..." "Info"
echo         Invoke-WebRequest -Uri $Url -OutFile $OutputPath -ErrorAction Stop
echo         Write-Status "Download completed: $OutputPath" "Success"
echo         return $true
echo     }
echo     catch {
echo         Write-Status "Download failed: $^($_.Exception.Message^)" "Error"
echo         return $false
echo     }
echo }
echo.
echo # Function to copy local file
echo function Copy-LocalFile {
echo     param^([string]$Source, [string]$Destination^)
echo     try {
echo         if ^(Test-Path $Source^) {
echo             Copy-Item $Source $Destination -Force
echo             Write-Status "File copied: $Destination" "Success"
echo             return $true
echo         }
echo         else {
echo             Write-Status "Source file not found: $Source" "Error"
echo             return $false
echo         }
echo     }
echo     catch {
echo         Write-Status "Copy failed: $^($_.Exception.Message^)" "Error"
echo         return $false
echo     }
echo }
echo.
echo # Function to find Python installation
echo function Find-Python {
echo     $pythonPaths = @^(
echo         "$env:ProgramFiles\Python312\python.exe",
echo         "$env:LocalAppData\Programs\Python\Python312\python.exe",
echo         "$env:ProgramFiles\Python311\python.exe",
echo         "$env:LocalAppData\Programs\Python\Python311\python.exe",
echo         "$env:ProgramFiles\Python310\python.exe",
echo         "$env:LocalAppData\Programs\Python\Python310\python.exe"
echo     ^)
echo     
echo     foreach ^($path in $pythonPaths^) {
echo         if ^(Test-Path $path^) {
echo             Write-Status "Found Python: $path" "Success"
echo             return $path
echo         }
echo     }
echo     
echo     # Try to find python in PATH
echo     try {
echo         $pythonCmd = Get-Command python -ErrorAction Stop
echo         Write-Status "Found Python in PATH: $^($pythonCmd.Source^)" "Success"
echo         return $pythonCmd.Source
echo     }
echo     catch {
echo         # Python not found anywhere
echo         return $null
echo     }
echo }
echo.
echo # Function to install Python
echo function Install-Python {
echo     Write-Status "Python not found. Installing Python 3.12..." "Warning"
echo     try {
echo         $installerPath = "python_installer.exe"
echo         Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe' -OutFile $installerPath
echo         
echo         Write-Status "Running Python installer..." "Info"
echo         Start-Process -Wait $installerPath -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1'
echo         Remove-Item $installerPath -Force
echo         
echo         # Refresh environment variables
echo         $env:Path = [System.Environment]::GetEnvironmentVariable^("Path", "Machine"^) + ";" + [System.Environment]::GetEnvironmentVariable^("Path", "User"^)
echo         
echo         # Try to find Python again
echo         return Find-Python
echo     }
echo     catch {
echo         Write-Status "Python installation failed: $^($_.Exception.Message^)" "Error"
echo         return $null
echo     }
echo }
echo.
echo # Main execution starts here
echo Clear-Host
echo Write-Host "========================================" -ForegroundColor Magenta
echo Write-Host "3D Audio Processing Tool" -ForegroundColor Magenta
echo Write-Host "========================================" -ForegroundColor Magenta
echo Write-Host ""
echo.
echo # Get output directory
echo if ^(-not $OutputDir^) {
echo     $OutputDir = Get-UserInput "Enter full path for output folder ^(will be created if it doesn't exist^)"
echo }
echo.
echo if ^(-not ^(Test-Path $OutputDir^)^) {
echo     try {
echo         New-Item -Path $OutputDir -ItemType Directory -Force ^| Out-Null
echo         Write-Status "Created output directory: $OutputDir" "Success"
echo     }
echo     catch {
echo         Write-Status "Failed to create output directory: $^($_.Exception.Message^)" "Error"
echo         Read-Host "Press Enter to exit"
echo         exit 1
echo     }
echo }
echo else {
echo     Write-Status "Using existing directory: $OutputDir" "Info"
echo }
echo.
echo # Get music input
echo if ^(-not $MusicInput^) {
echo     $MusicInput = Get-UserInput "Enter full path to WAV file or URL"
echo }
echo.
echo $MusicFile = Join-Path $OutputDir "music_input.wav"
echo.
echo # Get video input ^(optional^)
echo if ^(-not $PSBoundParameters.ContainsKey^('VideoInput'^)^) {
echo     $VideoInput = Read-Host "Enter full path to video file or URL ^(optional, press Enter to skip^)"
echo }
echo.
echo # Find or install Python
echo Write-Status "Checking Python installation..." "Info"
echo $PythonExe = Find-Python
echo if ^(-not $PythonExe^) {
echo     $PythonExe = Install-Python
echo     if ^(-not $PythonExe^) {
echo         Write-Status "Failed to install or find Python. Exiting." "Error"
echo         Read-Host "Press Enter to exit"
echo         exit 1
echo     }
echo }
echo.
echo # Install Python dependencies
echo Write-Status "Installing Python dependencies..." "Info"
echo try {
echo     ^& $PythonExe -m ensurepip --upgrade 2^>$null
echo     ^& $PythonExe -m pip install --upgrade pip numpy scipy pydub
echo     if ^($LASTEXITCODE -ne 0^) {
echo         Write-Status "Warning: Some dependencies might not have installed correctly" "Warning"
echo     }
echo     else {
echo         Write-Status "Dependencies installed successfully" "Success"
echo     }
echo }
echo catch {
echo     Write-Status "Failed to install dependencies: $^($_.Exception.Message^)" "Error"
echo }
echo.
echo # Download Python script
echo $PyScript = Join-Path $OutputDir "pa1112.py"
echo if ^(-not ^(Test-Path $PyScript^)^) {
echo     Write-Status "Downloading Python processing script..." "Info"
echo     if ^(-not ^(Get-FileFromUrl "https://tinyurl.com/ARBYAUDIO2" $PyScript^)^) {
echo         Write-Status "Failed to download Python script. Exiting." "Error"
echo         Read-Host "Press Enter to exit"
echo         exit 1
echo     }
echo }
echo else {
echo     Write-Status "Python script already exists" "Info"
echo }
echo.
echo # Process music input
echo Write-Status "Processing music input..." "Info"
echo $musicSuccess = $false
echo if ^(Test-IsUrl $MusicInput^) {
echo     $musicSuccess = Get-FileFromUrl $MusicInput $MusicFile
echo }
echo else {
echo     $musicSuccess = Copy-LocalFile $MusicInput $MusicFile
echo }
echo.
echo if ^(-not $musicSuccess^) {
echo     Write-Status "Failed to process music input. Exiting." "Error"
echo     Read-Host "Press Enter to exit"
echo     exit 1
echo }
echo.
echo # Process video input ^(optional^)
echo $VideoFile = $null
echo $VideoArg = @^(^)
echo.
echo if ^(-not [string]::IsNullOrWhiteSpace^($VideoInput^)^) {
echo     Write-Status "Processing video input..." "Info"
echo     
echo     if ^(Test-IsUrl $VideoInput^) {
echo         $VideoFile = Join-Path $OutputDir "input_video.mp4"
echo         if ^(Get-FileFromUrl $VideoInput $VideoFile^) {
echo             $VideoArg = @^("--video_file", $VideoFile^)
echo         }
echo         else {
echo             Write-Status "Video download failed. Continuing with audio only..." "Warning"
echo         }
echo     }
echo     else {
echo         if ^(Test-Path $VideoInput^) {
echo             $VideoFileName = Split-Path $VideoInput -Leaf
echo             $VideoFile = Join-Path $OutputDir $VideoFileName
echo             if ^(Copy-LocalFile $VideoInput $VideoFile^) {
echo                 $VideoArg = @^("--video_file", $VideoFile^)
echo             }
echo             else {
echo                 Write-Status "Video copy failed. Continuing with audio only..." "Warning"
echo             }
echo         }
echo         else {
echo             Write-Status "Video file not found: $VideoInput. Continuing with audio only..." "Warning"
echo         }
echo     }
echo }
echo else {
echo     Write-Status "No video input provided. Processing audio only..." "Info"
echo }
echo.
echo # Display processing status
echo Write-Host ""
echo Write-Host "========================================" -ForegroundColor Magenta
echo Write-Host "PROCESSING STATUS:" -ForegroundColor Magenta
echo Write-Host "========================================" -ForegroundColor Magenta
echo Write-Status "Music file: $MusicFile" "Success"
echo if ^($VideoFile^) {
echo     Write-Status "Video file: $VideoFile" "Success"
echo }
echo else {
echo     Write-Status "No video file - generating audio only" "Warning"
echo }
echo Write-Host "========================================" -ForegroundColor Magenta
echo Write-Host ""
echo.
echo # Run Python processing script
echo Write-Status "Starting audio processing..." "Info"
echo try {
echo     $arguments = @^("--music_url", $MusicFile^) + $VideoArg
echo     ^& $PythonExe $PyScript @arguments
echo     
echo     if ^($LASTEXITCODE -eq 0^) {
echo         Write-Host ""
echo         Write-Status "Processing completed successfully!" "Success"
echo         Write-Status "Check the output folder: $OutputDir" "Info"
echo     }
echo     else {
echo         Write-Status "Python script execution failed with exit code: $LASTEXITCODE" "Error"
echo     }
echo }
echo catch {
echo     Write-Status "Failed to run Python script: $^($_.Exception.Message^)" "Error"
echo }
echo.
echo Write-Host ""
echo Read-Host "Press Enter to exit"
) > "%PS_SCRIPT%"
echo ✅ PowerShell script created: %PS_SCRIPT%
goto :EOF
