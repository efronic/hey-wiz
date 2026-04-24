"""Audio pipeline: wake word detection, voice recording, and Whisper STT.

Wake word  — openwakeword (queue-based, threaded listener)
STT        — whisper.cpp  (compiled C++ binary via subprocess)
Audio      — sounddevice  (adaptive gain, mic muting, device-by-name lookup)
"""

import logging
import os
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import openwakeword
import sounddevice as sd
from openwakeword.model import Model as WakeWordModel
from scipy.signal import decimate

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device helpers (survive USB re-enumeration across reboots)
# ---------------------------------------------------------------------------


def _find_device_by_name(name_substring: str, kind: str) -> int | None:
    """Find a sounddevice device index by name substring.

    Returns the index, or None if *name_substring* is empty / not found
    (falls back to system default).
    """
    if not name_substring:
        return None
    devices = sd.query_devices()
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(devices):
        if name_substring.lower() in d["name"].lower() and d[channel_key] > 0:
            return i
    log.warning(
        "Audio device matching '%s' (%s) not found — using system default",
        name_substring,
        kind,
    )
    return None


# ---------------------------------------------------------------------------
# Module-level state (initialised by init())
# ---------------------------------------------------------------------------
_oww_model: WakeWordModel | None = None
_wake_event = threading.Event()
_resume_event = threading.Event()
_running = False
_paused = False
_listen_thread: threading.Thread | None = None
_audio_queue: Queue = Queue()

# Adaptive gain state
_gain = 4.0

# Mic muting (prevents self-triggering during TTS playback)
_muted = False
_mute_lock = threading.Lock()


def mute() -> None:
    """Mute microphone input (call before TTS playback)."""
    global _muted
    with _mute_lock:
        _muted = True


def unmute() -> None:
    """Unmute microphone input (call after TTS playback)."""
    global _muted
    with _mute_lock:
        _muted = False


# ---------------------------------------------------------------------------
# Adaptive gain normalisation for weak USB mics
# ---------------------------------------------------------------------------


def _normalize(audio: np.ndarray) -> np.ndarray:
    """Apply adaptive, smoothed gain normalisation."""
    global _gain
    peak = np.max(np.abs(audio))
    if peak < 50:
        return audio.astype(np.int16)
    target = config.GAIN_TARGET_PEAK * 32767
    desired_gain = min(target / peak, 15.0)
    # Smooth: 30 % new + 70 % old — avoids sudden jumps
    _gain = 0.3 * desired_gain + 0.7 * _gain
    _gain = min(_gain, 15.0)
    gained = np.clip(audio.astype(np.float64) * _gain, -32768, 32767)
    return gained.astype(np.int16)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init() -> None:
    """Pre-load the openwakeword model and resolve mic device (call once)."""
    global _oww_model
    log.info("Loading wake-word model: %s", config.WAKE_WORD_MODEL)

    model_paths = openwakeword.get_pretrained_model_paths()
    target_path = next((p for p in model_paths if config.WAKE_WORD_MODEL in p), None)
    if not target_path:
        raise ValueError(
            f"Could not find pre-trained model for {config.WAKE_WORD_MODEL}"
        )
    _oww_model = WakeWordModel(wakeword_model_paths=[target_path])

    # Resolve mic device index by name so it survives USB re-enumeration
    resolved = _find_device_by_name(config.MIC_NAME, "input")
    if resolved is not None:
        config.MIC_DEVICE = resolved
        log.info("Mic device resolved: index %d (%s)", resolved, config.MIC_NAME)


# ---------------------------------------------------------------------------
# Queue-based wake word listener (runs on a daemon thread)
# ---------------------------------------------------------------------------


def _listen_loop() -> None:
    """Main listening loop — reopens stream after each pause/resume cycle."""
    global _paused

    while _running:
        _resume_event.wait()
        if not _running:
            break

        # Drain stale audio from the queue
        while not _audio_queue.empty():
            try:
                _audio_queue.get_nowait()
            except Empty:
                break

        def _raw_callback(indata, frames, time_info, status):
            if _muted or _paused:
                return
            _audio_queue.put(bytes(indata))

        try:
            stream = sd.RawInputStream(
                device=config.MIC_DEVICE,
                samplerate=config.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=config.CHUNK_SAMPLES,
                latency="high",
                callback=_raw_callback,
            )
            stream.start()
        except Exception as exc:
            log.error("Wake word stream error: %s", exc)
            if _running:
                time.sleep(1.0)
            continue

        detected = False
        while _running and not _paused:
            try:
                raw = _audio_queue.get(timeout=0.1)
            except Empty:
                continue

            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
            normalised = _normalize(audio)
            # Anti-aliased downsampling (low-pass filter before decimation)
            decimated = decimate(
                normalised, config.DOWNSAMPLE_FACTOR, ftype="fir"
            ).astype(np.int16)

            predictions = _oww_model.predict(decimated)
            for model_name, score in predictions.items():
                if score >= config.WAKE_WORD_THRESHOLD:
                    log.info("Wake word detected: %s (score=%.3f)", model_name, score)
                    detected = True
                    break
            if detected:
                break

        # Close stream BEFORE signalling — frees USB mic for recording
        stream.stop()
        stream.close()

        if detected:
            _paused = True
            _resume_event.clear()
            _oww_model.reset()
            _wake_event.set()


def start_listening() -> None:
    """Start the wake-word listener thread (safe to call multiple times)."""
    global _running, _paused, _listen_thread
    if _oww_model is None:
        raise RuntimeError("Call init() before start_listening()")

    _wake_event.clear()
    _paused = False
    _resume_event.set()

    if _listen_thread is None or not _listen_thread.is_alive():
        _running = True
        _listen_thread = threading.Thread(target=_listen_loop, daemon=True)
        _listen_thread.start()

    log.info("Wake-word listener active (model=%s)", config.WAKE_WORD_MODEL)


def resume_listening() -> None:
    """Resume wake-word detection after recording (re-opens mic stream)."""
    global _paused
    # Drain stale audio
    while not _audio_queue.empty():
        try:
            _audio_queue.get_nowait()
        except Empty:
            break
    _paused = False
    _resume_event.set()
    log.info("Wake-word listener resumed")


def wait_for_wake() -> None:
    """Block until the wake word is detected."""
    import sys

    while not getattr(sys.modules["__main__"], "_shutdown", False):
        if _wake_event.wait(timeout=0.1):
            _wake_event.clear()
            return


# ---------------------------------------------------------------------------
# Voice command recording (with gain normalisation)
# ---------------------------------------------------------------------------


def record_command() -> np.ndarray:
    """Record audio until silence is detected.  Returns 16 kHz int16 numpy array."""
    log.info("Recording command…")
    audio_buffer: list[np.ndarray] = []
    silence_samples = 0
    blocksize = 4096
    silence_samples_needed = int(
        config.SILENCE_DURATION * config.SAMPLE_RATE / blocksize
    )
    max_chunks = int(config.MAX_RECORD_SECONDS * config.SAMPLE_RATE / blocksize)
    grace_chunks = int(config.MIN_RECORD_SECONDS * config.SAMPLE_RATE / blocksize)
    total_chunks = 0

    buf: list[np.ndarray] = []

    def callback(indata, frames, time_info, status):
        if status:
            pass  # ignore non-fatal USB overflow
        if _muted:
            return
        buf.append(indata.copy())

    stream = sd.InputStream(
        device=config.MIC_DEVICE,
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=blocksize,
        latency="high",
        callback=callback,
    )
    stream.start()

    try:
        while total_chunks < max_chunks:
            sd.sleep(100)
            total_chunks += 1

            if not buf:
                continue

            # Grab the latest chunk for silence detection
            recent = buf[-1]
            audio_buffer.extend(buf)
            buf.clear()

            # Skip silence detection during grace period
            if total_chunks < grace_chunks:
                continue

            rms = np.sqrt(np.mean(recent.astype(np.float32) ** 2)) / 32768.0
            if rms < config.SILENCE_THRESHOLD:
                silence_samples += 1
                if silence_samples >= silence_samples_needed:
                    log.info("Silence detected, stopping recording.")
                    break
            else:
                silence_samples = 0
    finally:
        stream.stop()
        stream.close()

    if not audio_buffer:
        log.warning("No audio captured.")
        resume_listening()
        return np.array([], dtype=np.int16)

    raw_audio = np.concatenate(audio_buffer, axis=0).flatten()
    normalised = _normalize(raw_audio)
    # Anti-aliased downsampling 48 kHz → 16 kHz (low-pass filter + decimation)
    decimated = decimate(normalised, config.DOWNSAMPLE_FACTOR, ftype="fir").astype(
        np.int16
    )

    log.info(
        "Recorded %d samples (%.1f s @ 16 kHz)",
        len(decimated),
        len(decimated) / config.WHISPER_SAMPLE_RATE,
    )

    # Wake-word listener is resumed from main.py after processing completes
    return decimated


# ---------------------------------------------------------------------------
# Whisper STT  (whisper.cpp subprocess)
# ---------------------------------------------------------------------------


def transcribe(audio: np.ndarray) -> str:
    """Save 16 kHz int16 audio to a temp WAV, run whisper-cpp, return text."""
    if len(audio) == 0:
        return ""

    whisper_bin = config.WHISPER_BINARY_PATH
    if not Path(whisper_bin).exists():
        raise FileNotFoundError(f"whisper-cpp not found at {whisper_bin}")

    # Write temp WAV (16 kHz, mono, 16-bit)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(config.WHISPER_SAMPLE_RATE)
            wf.writeframes(audio.tobytes())

        log.info("Transcribing %d samples with whisper-cpp…", len(audio))

        result = subprocess.run(
            [
                whisper_bin,
                "-m",
                config.WHISPER_MODEL_PATH,
                "-f",
                tmp_path,
                "-l",
                "en",
                "-t",
                str(config.WHISPER_THREADS),
                "-bs",
                str(config.WHISPER_BEAM_SIZE),
                "--no-timestamps",
                "-np",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            log.error("whisper-cpp failed: %s", result.stderr)
            return ""

        text = result.stdout.strip()
        text = text.replace("[BLANK_AUDIO]", "").strip()
        log.info("Transcription: %s", text)
        return text

    finally:
        os.unlink(tmp_path)
