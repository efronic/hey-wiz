"""Voice output: text-to-speech via piper-tts (in-process) and ALSA playback.

Mutes the microphone while speaking to prevent self-triggering the wake word.
"""

import logging
import os
import random
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

import audio_pipeline
import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_voice = None  # PiperVoice instance
_speaker_alsa: str = "plughw:0,0"
_filler_wavs: dict[str, str] = {}


# ---------------------------------------------------------------------------
# ALSA speaker lookup by name
# ---------------------------------------------------------------------------


def _find_alsa_card_by_name(name_substring: str) -> str:
    """Find ALSA card number by name, returns 'plughw:N,0' string."""
    if not name_substring:
        return "plughw:0,0"
    try:
        result = subprocess.run(
            ["aplay", "-l"], capture_output=True, text=True, check=True
        )
        for line in result.stdout.splitlines():
            if line.startswith("card ") and name_substring in line:
                card_num = line.split(":")[0].replace("card ", "").strip()
                return f"plughw:{card_num},0"
    except Exception:
        pass
    log.warning("ALSA card matching '%s' not found — using plughw:0,0", name_substring)
    return "plughw:0,0"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init() -> None:
    """Pre-load Piper voice model and resolve speaker device (call once)."""
    global _voice, _speaker_alsa

    from piper import PiperVoice

    model_path = _resolve_piper_model(config.PIPER_MODEL_PATH)
    if model_path != config.PIPER_MODEL_PATH:
        log.warning(
            "Piper model not found at %s; using %s",
            config.PIPER_MODEL_PATH,
            model_path,
        )

    log.info("Loading Piper voice model: %s", model_path)
    _voice = PiperVoice.load(model_path)

    _speaker_alsa = _find_alsa_card_by_name(config.SPEAKER_NAME)
    log.info("Speaker ALSA device: %s (%s)", _speaker_alsa, config.SPEAKER_NAME)

    _set_volume(config.SPEAKER_VOLUME)

    # Load pre-generated filler WAVs
    _load_fillers()


def _set_volume(volume_pct: int) -> None:
    """Set ALSA PCM volume on the resolved speaker card."""
    # Extract card number from 'plughw:N,0'
    try:
        card_num = _speaker_alsa.split(":")[1].split(",")[0]
        subprocess.run(
            ["amixer", "-c", card_num, "sset", "PCM", f"{volume_pct}%"],
            capture_output=True,
            check=True,
        )
        log.info("Speaker volume set to %d%% (card %s)", volume_pct, card_num)
    except Exception as exc:
        log.warning("Could not set speaker volume: %s", exc)


def _resolve_piper_model(configured_path: str) -> str:
    """Resolve a usable Piper model path from configured and common local files."""
    configured = Path(configured_path) if configured_path else None
    if configured and configured.exists():
        return str(configured)

    project_root = Path(__file__).resolve().parent
    candidates = [
        project_root / "models" / "en_US-lessac-high.onnx",
        project_root / "models" / "en_US-lessac-medium.onnx",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Piper voice model not found at {} and no fallback model was found in models/."
        .format(configured_path)
    )


def _load_fillers() -> None:
    """Discover pre-generated filler WAVs in FILLER_DIR."""
    filler_dir = config.FILLER_DIR
    if not filler_dir.is_dir():
        log.info("No filler directory at %s — fillers disabled", filler_dir)
        return
    for i, phrase in enumerate(config.FILLER_PHRASES):
        wav_path = filler_dir / f"filler_{i}.wav"
        if wav_path.exists():
            _filler_wavs[phrase] = str(wav_path)
    if _filler_wavs:
        log.info("Loaded %d filler phrases", len(_filler_wavs))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(text: str) -> str:
    """Strip markdown artefacts and cap length."""
    text = re.sub(r"[*_`#\[\]]", "", text)
    text = text.strip()
    if len(text) > config.MAX_TTS_CHARS:
        text = text[: config.MAX_TTS_CHARS] + "…"
    return text


def _play_wav(filepath: str) -> None:
    """Play a WAV file through the resolved ALSA speaker device."""
    try:
        subprocess.run(
            ["aplay", "-D", _speaker_alsa, filepath],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError:
        # aplay not available — fall back to sounddevice
        import sounddevice as sd

        with wave.open(filepath, "rb") as wf:
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            sd.play(data, wf.getframerate())
            sd.wait()
    except subprocess.TimeoutExpired:
        log.error("WAV playback timed out: %s", filepath)
    except subprocess.CalledProcessError as exc:
        log.error("WAV playback failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def speak(text: str) -> None:
    """Synthesise *text* with Piper (in-process) and play via ALSA.

    Mutes the mic before playback and unmutes after to prevent
    the assistant's own voice from triggering the wake word.
    """
    text = _sanitize(text)
    if not text:
        log.warning("Nothing to speak.")
        return

    if _voice is None:
        raise RuntimeError("Call voice_output.init() before speak()")

    log.info("Speaking: %s", text[:80])

    # Synthesise to a temp WAV
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        with wave.open(tmp_path, "wb") as wf:
            _voice.synthesize_wav(text, wf)

        audio_pipeline.mute()
        try:
            _play_wav(tmp_path)
        finally:
            audio_pipeline.unmute()
    finally:
        os.unlink(tmp_path)


def speak_filler() -> None:
    """Play a random pre-generated filler phrase (e.g. 'Thinking…').

    Fast because the WAV is pre-generated — no TTS latency.
    """
    if not _filler_wavs:
        return
    phrase = random.choice(list(_filler_wavs.keys()))
    wav_path = _filler_wavs[phrase]
    log.info("Filler: %s", phrase)
    audio_pipeline.mute()
    try:
        _play_wav(wav_path)
    finally:
        audio_pipeline.unmute()
