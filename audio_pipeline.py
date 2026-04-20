"""Audio pipeline: wake word detection, voice recording, and Whisper STT."""

import logging
import threading
import time

import numpy as np
import openwakeword
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
_mic_stream = None


def init() -> None:
    """Pre-load the openwakeword model (call once at startup)."""
    global _oww_model
    log.info("Loading wake-word model: %s", config.WAKE_WORD_MODEL)

    # Get the absolute file paths to all bundled pre-trained ONNX models
    model_paths = openwakeword.get_pretrained_model_paths()

    # Find the specific path that contains "hey_jarvis"
    target_path = next((p for p in model_paths if config.WAKE_WORD_MODEL in p), None)

    if not target_path:
        raise ValueError(
            f"Could not find pre-trained model for {config.WAKE_WORD_MODEL}"
        )

    # Initialize the model using the exact absolute file path
    _oww_model = WakeWordModel(wakeword_model_paths=[target_path])


# ---------------------------------------------------------------------------
# Wake word listener (runs on a daemon thread)
# ---------------------------------------------------------------------------


def _audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    """sounddevice callback — feed chunks to openwakeword."""
    if status:
        log.warning("Audio callback status: %s", status)
    if not _listening or _oww_model is None:
        return

    # openwakeword expects int16 mono at 16kHz!
    chunk = indata[:: config.DOWNSAMPLE_FACTOR, 0].copy()
    prediction = _oww_model.predict(chunk)

    for wake_word, score in prediction.items():
        if score > config.WAKE_WORD_THRESHOLD:
            log.info("Wake word detected: %s (score=%.3f)", wake_word, score)
            _oww_model.reset()
            _wake_event.set()
            return


def start_listening() -> None:
    """Open the mic stream and begin wake-word detection."""
    global _listening, _mic_stream
    if _oww_model is None:
        raise RuntimeError("Call init() before start_listening()")

    # 1. Reset the logic state
    _listening = True
    _wake_event.clear()  # <--- YOUR EXCELLENT CATCH: Reset the "switch"

    # 2. If a stream is already hanging around, kill it properly first
    if _mic_stream is not None:
        try:
            _mic_stream.stop()
            _mic_stream.close()
        except:
            pass
        _mic_stream = None

    # 3. Open the fresh hardware cÏonnection
    _mic_stream = sd.InputStream(
        device=config.MIC_DEVICE,
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=config.CHUNK_SAMPLES,
        latency=0.5,
        callback=_audio_callback,
    )
    _mic_stream.start()
    log.info("Wake-word listener active (model=%s)", config.WAKE_WORD_MODEL)


def wait_for_wake() -> None:
    """Block until the wake word is detected."""
    import sys
    # Loop in 100ms increments so we can catch the shutdown signal from main.py
    while not getattr(sys.modules['__main__'], '_shutdown', False):
        if _wake_event.wait(timeout=0.1):
            _wake_event.clear()
            return


# ---------------------------------------------------------------------------
# Voice command recording
# ---------------------------------------------------------------------------


def record_command() -> np.ndarray:
    """Record audio until silence is detected. Returns int16 numpy array."""
    global _mic_stream
    
    # 1. Stop the background wake-word listener so the mic is free
    if _mic_stream is not None:
        _mic_stream.stop()
        _mic_stream.close()
        _mic_stream = None

    log.info("Recording command…")
    frames: list[np.ndarray] = []
    silence_start: float | None = None
    max_end = time.monotonic() + config.MAX_RECORD_SECONDS

    # 2. Add device=config.MIC_DEVICE so it knows which mic to use
    with sd.InputStream(
        device=config.MIC_DEVICE,
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
    log.info(
        "Recorded %d samples (%.1f s)", len(audio), len(audio) / config.SAMPLE_RATE
    )

    # 3. Restart the wake-word listener so Wiz can hear "Hey Jarvis" again
    start_listening()
    
    return audio


# ---------------------------------------------------------------------------
# Whisper STT
# ---------------------------------------------------------------------------


_whisper_model = None


def init_whisper() -> None:
    """Pre-load the faster-whisper model (call once at startup)."""
    global _whisper_model
    from faster_whisper import WhisperModel

    log.info("Loading Whisper model: %s", config.WHISPER_MODEL_NAME)
    _whisper_model = WhisperModel(
        config.WHISPER_MODEL_NAME,
        device="cpu",
        compute_type="int8",
    )


def transcribe(audio: np.ndarray) -> str:
    """Downsample 48 kHz int16 → 16 kHz float32 and run Whisper STT."""
    global _whisper_model
    if _whisper_model is None:
        init_whisper()

    # Downsample by factor 3 (48000 → 16000)
    downsampled = audio[:: config.DOWNSAMPLE_FACTOR]

    # Normalise to float32 [-1.0, 1.0]
    audio_f32 = downsampled.astype(np.float32) / 32768.0

    log.info("Transcribing %d samples with Whisper…", len(audio_f32))

    segments, _info = _whisper_model.transcribe(audio_f32, beam_size=1)
    text = " ".join(seg.text.strip() for seg in segments).strip()

    log.info("Transcription: %s", text)
    return text
