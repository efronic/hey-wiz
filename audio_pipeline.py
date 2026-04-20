"""Audio pipeline: wake word detection, voice recording, and Whisper STT."""

import logging
import threading
import time

import numpy as np
import sounddevice as sd
from openwakeword.model import Model as WakeWordModel

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (initialised by init())
# ---------------------------------------------------------------------------
_oww_model: WakeWordModel | None = None
_wake_event = threading.Event()
_listening = False


def init() -> None:
    """Pre-load the openwakeword model (call once at startup)."""
    global _oww_model
    log.info("Loading wake-word model: %s", config.WAKE_WORD_MODEL)
    _oww_model = WakeWordModel(
        wakeword_models=[config.WAKE_WORD_MODEL],
        inference_framework="onnx",
    )


# ---------------------------------------------------------------------------
# Wake word listener (runs on a daemon thread)
# ---------------------------------------------------------------------------

def _audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    """sounddevice callback — feed chunks to openwakeword."""
    if status:
        log.warning("Audio callback status: %s", status)
    if not _listening or _oww_model is None:
        return

    # openwakeword expects int16 mono
    chunk = indata[:, 0].copy()
    prediction = _oww_model.predict(chunk)

    for wake_word, score in prediction.items():
        if score > config.WAKE_WORD_THRESHOLD:
            log.info("Wake word detected: %s (score=%.3f)", wake_word, score)
            _oww_model.reset()
            _wake_event.set()
            return


def start_listening() -> None:
    """Open the mic stream and begin wake-word detection on a background thread."""
    global _listening
    if _oww_model is None:
        raise RuntimeError("Call init() before start_listening()")

    _listening = True
    _wake_event.clear()

    stream = sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=config.CHUNK_SAMPLES,
        callback=_audio_callback,
    )
    stream.start()
    log.info("Wake-word listener active (model=%s)", config.WAKE_WORD_MODEL)


def wait_for_wake() -> None:
    """Block until the wake word is detected."""
    _wake_event.wait()
    _wake_event.clear()


# ---------------------------------------------------------------------------
# Voice command recording
# ---------------------------------------------------------------------------

def record_command() -> np.ndarray:
    """Record audio until silence is detected. Returns int16 numpy array."""
    log.info("Recording command…")
    frames: list[np.ndarray] = []
    silence_start: float | None = None
    max_end = time.monotonic() + config.MAX_RECORD_SECONDS

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=config.CHUNK_SAMPLES,
    ) as stream:
        while time.monotonic() < max_end:
            chunk, status = stream.read(config.CHUNK_SAMPLES)
            if status:
                log.warning("Record status: %s", status)

            frames.append(chunk[:, 0].copy())

            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
            if rms < config.SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.monotonic()
                elif time.monotonic() - silence_start >= config.SILENCE_DURATION:
                    log.info("Silence detected, stopping recording.")
                    break
            else:
                silence_start = None

    audio = np.concatenate(frames)
    log.info("Recorded %d samples (%.1f s)", len(audio), len(audio) / config.SAMPLE_RATE)
    return audio


# ---------------------------------------------------------------------------
# Whisper STT
# ---------------------------------------------------------------------------

def transcribe(audio: np.ndarray) -> str:
    """Downsample 48 kHz int16 → 16 kHz float32 and run Whisper STT."""
    # Downsample by factor 3 (48000 → 16000)
    downsampled = audio[:: config.DOWNSAMPLE_FACTOR]

    # Normalise to float32 [-1.0, 1.0]
    audio_f32 = downsampled.astype(np.float32) / 32768.0

    log.info("Transcribing %d samples with Whisper…", len(audio_f32))

    try:
        from whispercpp import Whisper  # type: ignore[import-untyped]

        w = Whisper.from_pretrained(config.WHISPER_MODEL_PATH)
        result = w.transcribe(audio_f32)
        text = result.strip()
    except Exception:
        log.warning("whispercpp binding failed, falling back to subprocess")
        text = _transcribe_subprocess(audio_f32)

    log.info("Transcription: %s", text)
    return text


def _transcribe_subprocess(audio_f32: np.ndarray) -> str:
    """Fallback: write temp WAV and invoke the whisper-cpp binary."""
    import subprocess
    import tempfile
    import wave

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(config.WHISPER_SAMPLE_RATE)
            wf.writeframes((audio_f32 * 32767).astype(np.int16).tobytes())

        result = subprocess.run(
            [
                "whisper-cpp",
                "-m", config.WHISPER_MODEL_PATH,
                "-f", tmp.name,
                "--no-timestamps",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return result.stdout.strip()
    finally:
        import os
        os.unlink(tmp.name)
