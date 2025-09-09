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
#!/bin/bash

# --- Music input ---
read -p "Enter full path to WAV file or URL: " MUSIC_INPUT
if [[ "$MUSIC_INPUT" == http* ]]; then
    MUSIC_FILE="music_input.wav"
    echo "Downloading WAV file..."
    curl -L "$MUSIC_INPUT" -o "$MUSIC_FILE" || { echo "Failed to download WAV file"; exit 1; }
else
    MUSIC_FILE="$MUSIC_INPUT"
    if [[ ! -f "$MUSIC_FILE" ]]; then
        echo "❌ WAV file not found: $MUSIC_FILE"
        exit 1
    fi
fi

# --- Video input ---
read -p "Enter full path to video file or URL (optional, press enter to skip): " VIDEO_INPUT
if [[ -n "$VIDEO_INPUT" ]]; then
    if [[ "$VIDEO_INPUT" == http* ]]; then
        VIDEO_FILE="input_video.mp4"
        echo "Downloading video file..."
        curl -L "$VIDEO_INPUT" -o "$VIDEO_FILE" || { echo "Failed to download video file"; exit 1; }
    else
        VIDEO_FILE="$VIDEO_INPUT"
        if [[ ! -f "$VIDEO_FILE" ]]; then
            echo "❌ Video file not found: $VIDEO_FILE"
            exit 1
        fi
    fi
else
    VIDEO_FILE=""
fi

echo "✅ Music file: $MUSIC_FILE"
if [[ -n "$VIDEO_FILE" ]]; then
    echo "✅ Video file: $VIDEO_FILE"
else
    echo "⚠️ No video file provided; will only generate audio."
fi

# --- Check if Python script exists, if not download ---
PY_SCRIPT="arbyaudioisthebest111.py"
if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "Python script not found, downloading..."
    curl -L "https://shorturl.at/i45Tk" -o "arbyaudioisthebest111.py" \
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
