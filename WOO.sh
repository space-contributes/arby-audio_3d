#!/bin/bash
# ========================================
# 3D Audio Processing Launcher (Linux/macOS)
# ========================================

# --- Prompt for WAV file URL or local path ---
read -p "Enter full path to WAV file or URL: " MUSIC_INPUT

# --- Check if Python is installed ---
if ! command -v python3 &> /dev/null
then
    echo "Python3 not found! Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt update && sudo apt install -y python3 python3-pip
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install python
    else
        echo "Unsupported OS. Install Python manually."
        exit 1
    fi
fi

# --- Download WAV file if it's a URL ---
if [[ "$MUSIC_INPUT" == http* ]]; then
    MUSIC_FILE="music_input.wav"
    echo "Downloading WAV file..."
    curl -L "$MUSIC_INPUT" -o "$MUSIC_FILE" || { echo "Failed to download WAV file"; exit 1; }
else
    MUSIC_FILE="$MUSIC_INPUT"
    if [[ ! -f "$MUSIC_FILE" ]]; then
        echo "File not found!"
        exit 1
    fi
fi

# --- Check if Python script exists, if not download ---
PY_SCRIPT="PYTHONSCRIPT.py"
if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "Python script not found, downloading..."
    curl -L "https://drive.usercontent.google.com/download?id=1NYLVw1kMjRypD2QG6FPLtxArFiLQdnsX&export=download" -o "$PY_SCRIPT" \
    || { echo "Failed to download Python script"; exit 1; }
fi

# --- Install Python dependencies ---
python3 -m pip install --upgrade pip
python3 -m pip install numpy scipy pydub --user

# --- Run Python script ---
echo "Running 3D audio processing..."
python3 "$PY_SCRIPT" --music_url "$MUSIC_FILE"

echo "Done."
