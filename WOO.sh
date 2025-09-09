#!/usr/bin/env bash
# ========================================
# 3D Audio Processing Launcher (macOS/Linux)
# Single-folder workflow with music URL
# ========================================

set -e

# --- Ask for output folder ---
read -p "Enter full path for folder containing downloads and outputs: " OUTPUT_DIR
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$PWD/output"
fi
mkdir -p "$OUTPUT_DIR"
echo "Files will be saved to: $OUTPUT_DIR"
echo

# --- Ask for music WAV URL ---
read -p "Enter URL to WAV music file: " MUSIC_URL
if [[ -z "$MUSIC_URL" ]]; then
    echo "❌ Music URL is required!"
    read -p "Press Enter to exit..."
    exit 1
fi
MUSIC_FILE="$OUTPUT_DIR/music_input.wav"

# --- Detect optional video file in folder ---
VIDEO_FILE=""
for f in "$OUTPUT_DIR"/*.mp4 "$OUTPUT_DIR"/*.mov "$OUTPUT_DIR"/*.mkv; do
    if [[ -f "$f" ]]; then
        VIDEO_FILE="$f"
        break
    fi
done

if [[ -z "$VIDEO_FILE" ]]; then
    echo "⚠️ No video file found in folder."
    read -p "Do you want to proceed without a video file? (y/n): " PROCEED
    if [[ "$PROCEED" != "y" && "$PROCEED" != "Y" ]]; then
        echo "Exiting as requested."
        exit 1
    fi
else
    echo "✅ Video file found: $VIDEO_FILE"
fi

# --- Check Python3 ---
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found!"
    read -p "Please install Python3 and rerun this script. Press Enter to exit..."
    exit 1
fi

# --- Download Python script ---
PY_SCRIPT="$OUTPUT_DIR/pa1112.py"
if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "Downloading Python script..."
    curl -L "https://tinyurl.com/ARBYAUDIO2" -o "$PY_SCRIPT" || { echo "❌ Failed to download Python script"; read -p "Press Enter to exit..."; exit 1; }
fi

# --- Download music file ---
echo "Downloading music file from URL..."
curl -L "$MUSIC_URL" -o "$MUSIC_FILE" || { echo "❌ Failed to download music file"; read -p "Press Enter to exit..."; exit 1; }

# --- Install Python dependencies ---
echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip --user
python3 -m pip install numpy scipy pydub --user

# --- Run Python script ---
echo "Running 3D audio processing..."
if [[ -z "$VIDEO_FILE" ]]; then
    python3 "$PY_SCRIPT" --music_file "$MUSIC_FILE"
else
    python3 "$PY_SCRIPT" --music_file "$MUSIC_FILE" --video_file "$VIDEO_FILE"
fi

echo "✅ Done. All outputs are stored in: $OUTPUT_DIR"
