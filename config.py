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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# Wake Word
# ---------------------------------------------------------------------------
WAKE_WORD_MODEL_PATH = os.getenv(
    "WAKE_WORD_MODEL_PATH",
    str(MODELS_DIR / "wake_word" / "hey_wiz.onnx"),
)
WAKE_WORD_THRESHOLD = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))
WAKE_SAMPLE_RATE = int(os.getenv("WAKE_SAMPLE_RATE", "16000"))
WAKE_WORD_ALLOW_BUNDLED_FALLBACK = _env_bool(
    "WAKE_WORD_ALLOW_BUNDLED_FALLBACK", True
)
WAKE_WORD_BUNDLED_MODEL_NAME = os.getenv("WAKE_WORD_BUNDLED_MODEL_NAME", "hey_jarvis")

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "48000"))  # USB mic native
WHISPER_SAMPLE_RATE = 16000  # Whisper expects 16 kHz
DOWNSAMPLE_FACTOR = SAMPLE_RATE // WHISPER_SAMPLE_RATE  # 3
CHUNK_SAMPLES = 3840  # 80ms at 48 kHz → 1280 at 16 kHz (matches pibot)
MIC_DEVICE = None  # Resolved at runtime; override with int index if needed
MIC_ALSA_DEVICE = None  # Resolved at runtime when arecord fallback is needed

# Device names for lookup (survives USB re-enumeration across reboots)
MIC_NAME = os.getenv("MIC_NAME", "USB PnP Sound Device")
SPEAKER_NAME = os.getenv("SPEAKER_NAME", "UACDemoV1.0")
SPEAKER_VOLUME = int(os.getenv("SPEAKER_VOLUME", "75"))  # ALSA PCM volume %

SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.01"))  # normalised RMS
SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", "1.5"))  # seconds
MIN_RECORD_SECONDS = float(os.getenv("MIN_RECORD_SECONDS", "2.0"))  # grace period
MAX_RECORD_SECONDS = int(os.getenv("MAX_RECORD_SECONDS", "30"))

# Adaptive gain normalisation for weak USB mics
GAIN_TARGET_PEAK = float(os.getenv("GAIN_TARGET_PEAK", "0.9"))

# ---------------------------------------------------------------------------
# Whisper STT  (whisper.cpp subprocess)
# ---------------------------------------------------------------------------
WHISPER_BINARY_PATH = os.getenv("WHISPER_BINARY_PATH", "/usr/local/bin/whisper-cpp")
WHISPER_THREADS = int(os.getenv("WHISPER_THREADS", "4"))
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
WHISPER_MODEL_PATH = os.getenv(
    "WHISPER_MODEL_PATH",
    str(MODELS_DIR / "ggml-small.en-q8_0.bin"),
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
OPENCLAW_GW_URL = os.getenv("OPENCLAW_GW_URL", "ws://127.0.0.1:18789")
OPENCLAW_GW_ORIGIN = os.getenv("OPENCLAW_GW_ORIGIN", "http://192.168.50.18:18789")
OPENCLAW_TOKEN = os.getenv("OPENCLAW_TOKEN", "")
OPENCLAW_SESSION_ID = os.getenv("OPENCLAW_SESSION_ID", "wiz-voice")
OPENCLAW_TIMEOUT = int(os.getenv("OPENCLAW_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# Piper TTS  (in-process via piper-tts Python package)
# ---------------------------------------------------------------------------
PIPER_MODEL_PATH = os.getenv(
    "PIPER_MODEL_PATH",
    str(MODELS_DIR / "en_US-lessac-high.onnx"),
)
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "500"))

# ---------------------------------------------------------------------------
# UI / Touchscreen display
# ---------------------------------------------------------------------------
ENABLE_UI = _env_bool("ENABLE_UI", True)
DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", "800"))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", "480"))
UI_FPS = int(os.getenv("UI_FPS", "30"))
UI_BACKEND = os.getenv("UI_BACKEND", "wayland")
UI_FBCON_FALLBACK = _env_bool("UI_FBCON_FALLBACK", True)
UI_ASSETS_PATH = Path(
    os.getenv("UI_ASSETS_PATH", str(PROJECT_DIR / "assets" / "face"))
)

# ---------------------------------------------------------------------------
# Filler phrases (pre-generated WAVs played while processing)
# ---------------------------------------------------------------------------
FILLER_DIR = PROJECT_DIR / "assets" / "fillers"
FILLER_PHRASES = [
    "On it!",
    "Thinking...",
    "Give me a sec.",
    "Let me check.",
    "Working on it.",
]

# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
CAPTURE_PATH = os.getenv("CAPTURE_PATH", "/tmp/wiz_capture.jpg")
