#!/usr/bin/env bash
# ========================================
# 3D Audio Processing Launcher (macOS/Linux)
# ========================================

# --- Ask user for OS ---
echo "Select your OS:"
echo "1) macOS"
echo "2) Linux"
read -p "Enter 1 or 2: " OS_CHOICE

# --- Prompt for output folder ---
read -p "Enter full path for output folder (will be created if it doesn't exist): " OUTPUT_DIR
mkdir -p "$OUTPUT_DIR"
echo "Files will be saved to: $OUTPUT_DIR"
echo

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
        sudo apt update && sudo apt install -y python3 python3-pip curl
    else
        echo "Invalid choice. Exiting."
        exit 1
    fi
fi

# --- Prompt for WAV file URL or local path ---
read -p "Enter full path to WAV file or URL: " MUSIC_INPUT
MUSIC_FILE="$OUTPUT_DIR/music_input.wav"

if [[ "$MUSIC_INPUT" == http* ]]; then
    echo "Downloading WAV file..."
    curl -L "$MUSIC_INPUT" -o "$MUSIC_FILE" || { echo "❌ Failed to download WAV file"; read -p "Press Enter to exit..."; exit 1; }
else
    if [[ -f "$MUSIC_INPUT" ]]; then
        cp "$MUSIC_INPUT" "$MUSIC_FILE"
    else
        echo "❌ WAV file not found: $MUSIC_INPUT"
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# --- Prompt for optional video file ---
read -p "Enter full path to video file or URL (optional, press Enter to skip): " VIDEO_INPUT
VIDEO_FILE=""

if [[ -n "$VIDEO_INPUT" ]]; then
    if [[ "$VIDEO_INPUT" == http* ]]; then
        VIDEO_FILE="$OUTPUT_DIR/input_video.mp4"
        echo "Downloading video file..."
        curl -L "$VIDEO_INPUT" -o "$VIDEO_FILE" || { echo "❌ Failed to download video file"; read -p "Press Enter to exit..."; exit 1; }
    else
        if [[ -f "$VIDEO_INPUT" ]]; then
            cp "$VIDEO_INPUT" "$OUTPUT_DIR/"
            VIDEO_FILE="$OUTPUT_DIR/$(basename "$VIDEO_INPUT")"
        else
            echo "⚠️ Video file not found: $VIDEO_INPUT. Skipping video..."
            VIDEO_FILE="SKIPPED"
        fi
    fi
else
    VIDEO_FILE="SKIPPED"
fi

echo "✅ Music file: $MUSIC_FILE"
if [[ "$VIDEO_FILE" == "SKIPPED" ]]; then
    echo "⚠️ No video file provided; only generating audio."
else
    echo "✅ Video file: $VIDEO_FILE"
fi

# --- Check if Python script exists ---
PY_SCRIPT="$OUTPUT_DIR/arbyaudioisthebest111.py"
if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "Python script not found, downloading..."
    curl -L "https://shorturl.at/i45Tk" -o "$PY_SCRIPT" || { echo "❌ Failed to download Python script"; read -p "Press Enter to exit..."; exit 1; }
fi

# --- Install Python dependencies ---
echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip --user
python3 -m pip install numpy scipy pydub --user

# --- Run Python script ---
echo "Running 3D audio processing..."
if [[ "$VIDEO_FILE" == "SKIPPED" ]]; then
    python3 "$PY_SCRIPT" --music_file "$MUSIC_FILE"
else
    python3 "$PY_SCRIPT" --music_file "$MUSIC_FILE" --video_file "$VIDEO_FILE"
fi

echo "✅ Done."
