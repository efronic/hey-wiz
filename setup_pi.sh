#!/usr/bin/env bash
# setup_pi.sh — One-shot installer for wiz-voice on Raspberry Pi 5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
ASSETS_DIR="$SCRIPT_DIR/assets"
FILLERS_DIR="$ASSETS_DIR/fillers"

echo "=== Installing system packages ==="
sudo apt-get update
sudo apt-get install -y \
    libcamera-dev \
    libcap-dev \
    python3-picamera2 \
    alsa-utils \
    portaudio19-dev \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    libfreetype6 \
    python3-venv \
    wget \
    build-essential \
    cmake \
    git

# ── Models directory ──────────────────────────────────────────
mkdir -p "$MODELS_DIR"
mkdir -p "$MODELS_DIR/wake_word"
mkdir -p "$ASSETS_DIR/face"

# ── Whisper.cpp ───────────────────────────────────────────────
WHISPER_DIR="$SCRIPT_DIR/whisper.cpp"
WHISPER_BIN="/usr/local/bin/whisper-cpp"
WHISPER_MODEL_Q8="$MODELS_DIR/ggml-small.en-q8_0.bin"

if [ ! -f "$WHISPER_BIN" ]; then
    echo "=== Building whisper.cpp ==="
    if [ ! -d "$WHISPER_DIR" ]; then
        git clone https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
    fi
    cd "$WHISPER_DIR"
    cmake -B build
    cmake --build build --config Release
    sudo cp build/bin/whisper-cli "$WHISPER_BIN"
    sudo cp build/bin/whisper-quantize /usr/local/bin/whisper-quantize
    cd "$SCRIPT_DIR"
else
    echo "whisper-cpp binary already present, skipping build."
fi

if [ ! -f "$WHISPER_MODEL_Q8" ]; then
    echo "=== Downloading & quantising Whisper small.en model ==="
    WHISPER_MODEL_RAW="$MODELS_DIR/ggml-small.en.bin"
    if [ ! -f "$WHISPER_MODEL_RAW" ]; then
        wget -q --show-progress -O "$WHISPER_MODEL_RAW" \
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin"
    fi
    "$WHISPER_DIR/build/bin/whisper-quantize" \
        "$WHISPER_MODEL_RAW" \
        "$WHISPER_MODEL_Q8" \
        q8_0
    echo "Quantised model saved to $WHISPER_MODEL_Q8"
else
    echo "Quantised Whisper model already present, skipping."
fi

# ── Piper TTS voice model ────────────────────────────────────
PIPER_MODEL="$MODELS_DIR/en_US-lessac-high.onnx"
PIPER_CONFIG="$MODELS_DIR/en_US-lessac-high.onnx.json"
if [ ! -f "$PIPER_MODEL" ]; then
    echo "=== Downloading Piper voice model ==="
    wget -q --show-progress -O "$PIPER_MODEL" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx"
    wget -q --show-progress -O "$PIPER_CONFIG" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx.json"
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

# ── Pre-generate filler WAVs ─────────────────────────────────
if [ ! -d "$FILLERS_DIR" ] || [ -z "$(ls -A "$FILLERS_DIR" 2>/dev/null)" ]; then
    echo "=== Generating filler WAVs with Piper ==="
    mkdir -p "$FILLERS_DIR"
    PHRASES=("On it!" "Thinking..." "Give me a sec." "Let me check." "Working on it.")
    for i in "${!PHRASES[@]}"; do
        echo "${PHRASES[$i]}" | python3 -c "
import sys, wave
from piper import PiperVoice
voice = PiperVoice.load('$PIPER_MODEL')
with wave.open('$FILLERS_DIR/filler_${i}.wav', 'wb') as wf:
    voice.synthesize_wav(sys.stdin.read().strip(), wf)
"
        echo "  Generated filler_${i}.wav: ${PHRASES[$i]}"
    done
else
    echo "Filler WAVs already present, skipping."
fi

# ── Sanity checks ─────────────────────────────────────────────
echo ""
echo "=== Audio devices ==="
arecord -l 2>/dev/null || echo "(no capture devices found)"
echo ""
aplay -l 2>/dev/null || echo "(no playback devices found)"

echo ""
echo "=== Verifying whisper-cpp ==="
whisper-cpp --help 2>&1 | head -1 || echo "(whisper-cpp not found in PATH)"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Copy .env.example to .env and fill in your API keys."
echo "Set WAKE_WORD_MODEL_PATH in .env to your custom ONNX wake model."
