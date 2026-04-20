#!/usr/bin/env bash
# setup_pi.sh — One-shot installer for wiz-voice on Raspberry Pi 5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

echo "=== Installing system packages ==="
sudo apt-get update
sudo apt-get install -y \
    libcamera-dev \
    libcap-dev \
    python3-picamera2 \
    alsa-utils \
    portaudio19-dev \
    python3-venv \
    wget

# ── Models directory ──────────────────────────────────────────
mkdir -p "$MODELS_DIR"

# ── Whisper model ─────────────────────────────────────────────
WHISPER_MODEL="$MODELS_DIR/ggml-base.en.bin"
if [ ! -f "$WHISPER_MODEL" ]; then
    echo "=== Downloading Whisper base.en model ==="
    wget -q --show-progress -O "$WHISPER_MODEL" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
else
    echo "Whisper model already present, skipping."
fi

# ── Piper TTS ─────────────────────────────────────────────────
PIPER_MODEL="$MODELS_DIR/en_US-lessac-medium.onnx"
PIPER_CONFIG="$MODELS_DIR/en_US-lessac-medium.onnx.json"
if [ ! -f "$PIPER_MODEL" ]; then
    echo "=== Downloading Piper voice model ==="
    wget -q --show-progress -O "$PIPER_MODEL" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    wget -q --show-progress -O "$PIPER_CONFIG" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
else
    echo "Piper model already present, skipping."
fi

# ── Python venv ───────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "=== Creating Python venv ==="
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

# ── Sanity checks ─────────────────────────────────────────────
echo ""
echo "=== Audio devices ==="
arecord -l 2>/dev/null || echo "(no capture devices found)"
echo ""
aplay -l 2>/dev/null || echo "(no playback devices found)"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Copy .env.example to .env and fill in your API keys."
