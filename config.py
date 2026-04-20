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
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "48000"))  # USB mic native
WHISPER_SAMPLE_RATE = 16000  # Whisper expects 16 kHz
DOWNSAMPLE_FACTOR = SAMPLE_RATE // WHISPER_SAMPLE_RATE  # 3
CHUNK_SAMPLES = 15360  # 320ms buffer to prevent input overflow
MIC_DEVICE = 0
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "48000"))  # USB mic native

SILENCE_THRESHOLD = int(os.getenv("SILENCE_THRESHOLD", "500"))  # RMS value
SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", "1.5"))  # seconds
MIN_RECORD_SECONDS = float(os.getenv("MIN_RECORD_SECONDS", "2.0"))  # grace period
MAX_RECORD_SECONDS = int(os.getenv("MAX_RECORD_SECONDS", "30"))

# ---------------------------------------------------------------------------
# Whisper STT
# ---------------------------------------------------------------------------
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base.en")
WHISPER_MODEL_PATH = os.getenv(
    "WHISPER_MODEL_PATH",
    str(MODELS_DIR / "ggml-base.en.bin"),
)

# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "anthropic")  # "anthropic" | "openai"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL_ANTHROPIC = os.getenv("VISION_MODEL_ANTHROPIC", "claude-sonnet-4-20250514")
VISION_MODEL_OPENAI = os.getenv("VISION_MODEL_OPENAI", "gpt-4o")

# ---------------------------------------------------------------------------
# Command Tags — tag-based routing for predefined skills
#
# Each tag maps to a routine.  The user prefixes their voice command with
# the tag keyword (e.g. "IMDB") and the system runs the matching pipeline.
# No tag → pass-through to OpenClaw as a general assistant.
#
# Fields:
#   triggers      – list of spoken keywords that activate the tag
#   needs_vision  – whether the routine requires a camera capture
#   vision_prompt – prompt sent to the Vision API (only if needs_vision)
#   openclaw_template – prompt template for OpenClaw; {result} is replaced
#                       with the Vision API output (or empty string)
# ---------------------------------------------------------------------------
COMMAND_TAGS: dict[str, dict] = {
    "imdb": {
        "triggers": ["imdb"],
        "needs_vision": True,
        "vision_prompt": (
            "Analyze this image. If it shows a movie or TV show, return "
            "ONLY the exact title of that movie/show. If it's something "
            "else, return a brief factual description."
        ),
        "openclaw_template": (
            "Using the agent-browser skill, go to imdb.com, search for "
            "{result}, read the accessibility tree, and return only the "
            "IMDB rating and Metascore as a natural language string."
        ),
    },
    # Add more tags here, e.g.:
    # "identify": {
    #     "triggers": ["identify", "what is this"],
    #     "needs_vision": True,
    #     "vision_prompt": "Describe what you see in this image in one sentence.",
    #     "openclaw_template": "{result}",
    # },
}

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
