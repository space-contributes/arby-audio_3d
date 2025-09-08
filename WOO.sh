#!/usr/bin/env bash
# ========================================
# 3D Audio Processing Launcher (macOS/Linux)
# ========================================

# --- Ask user for OS ---
echo "Select your OS: MacOS is not Linux, at least in this case"
echo "1) macOS"
echo "2) Linux"
read -p "Enter 1 or 2: " OS_CHOICE

# --- Check if Python3 is installed ---
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found! Installing..."
    if [[ "$OS_CHOICE" == "1" ]]; then
        # macOS
        if ! command -v brew &> /dev/null; then
            echo "Homebrew not found! Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install python
    elif [[ "$OS_CHOICE" == "2" ]]; then
        # Linux (Debian/Ubuntu)
        sudo apt update && sudo apt install -y python3 python3-pip
    else
        echo "Invalid choice. Exiting."
        exit 1
    fi
fi

# --- Prompt for WAV file URL or local path ---
read -p "Enter full path to WAV file or URL: " MUSIC_INPUT

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
echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip --user
python3 -m pip install numpy scipy pydub --user

# --- Run Python script ---
echo "Running 3D audio processing..."
python3 "$PY_SCRIPT" --music_url "$MUSIC_FILE"

echo "Done."
