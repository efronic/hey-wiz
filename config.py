"""Centralized configuration for wiz-voice.

All tunables are loaded from environment variables with sane defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"

# ---------------------------------------------------------------------------
# Wake Word
# ---------------------------------------------------------------------------
WAKE_WORD_MODEL = os.getenv("WAKE_WORD_MODEL", "hey_jarvis")
WAKE_WORD_THRESHOLD = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "48000"))          # USB mic native
WHISPER_SAMPLE_RATE = 16000                                    # Whisper expects 16 kHz
DOWNSAMPLE_FACTOR = SAMPLE_RATE // WHISPER_SAMPLE_RATE         # 3
CHUNK_SAMPLES = 1280                                           # ~26 ms at 48 kHz

SILENCE_THRESHOLD = int(os.getenv("SILENCE_THRESHOLD", "500")) # RMS value
SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", "1.5")) # seconds
MAX_RECORD_SECONDS = int(os.getenv("MAX_RECORD_SECONDS", "30"))

# ---------------------------------------------------------------------------
# Whisper STT
# ---------------------------------------------------------------------------
WHISPER_MODEL_PATH = os.getenv(
    "WHISPER_MODEL_PATH",
    str(MODELS_DIR / "ggml-base.en.bin"),
)

# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "anthropic")    # "anthropic" | "openai"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL_ANTHROPIC = os.getenv("VISION_MODEL_ANTHROPIC", "claude-sonnet-4-20250514")
VISION_MODEL_OPENAI = os.getenv("VISION_MODEL_OPENAI", "gpt-4o")

VISION_PROMPT = (
    "Analyze this image. If it shows a movie or TV show, return ONLY the "
    "exact title of that movie/show. If it's something else, return a brief "
    "factual description."
)

VISION_INTENT_KEYWORDS = [
    "look at",
    "what is this",
    "identify",
    "show me",
    "what do you see",
    "can you see",
]

# ---------------------------------------------------------------------------
# OpenClaw
# ---------------------------------------------------------------------------
OPENCLAW_URL = os.getenv(
    "OPENCLAW_URL",
    "http://127.0.0.1:18789/v1/chat/completions",
)
OPENCLAW_TOKEN = os.getenv("OPENCLAW_TOKEN", "")
OPENCLAW_MODEL = os.getenv("OPENCLAW_MODEL", "openai-codex/gpt-5.4")

# ---------------------------------------------------------------------------
# Piper TTS
# ---------------------------------------------------------------------------
PIPER_BINARY = os.getenv("PIPER_BINARY", "piper")
PIPER_MODEL_PATH = os.getenv(
    "PIPER_MODEL_PATH",
    str(MODELS_DIR / "en_US-lessac-medium.onnx"),
)
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "500"))

# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
CAPTURE_PATH = os.getenv("CAPTURE_PATH", "/tmp/wiz_capture.jpg")
