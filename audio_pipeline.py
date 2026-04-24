"""Audio pipeline using pibot-style wake-word and STT modules."""

import logging
import subprocess
import threading

import numpy as np
import sounddevice as sd
from audio.stt_engine import WhisperSTT
from senses.wake_word_detector import WakeWordDetector

import config

log = logging.getLogger(__name__)

_wake_detector: WakeWordDetector | None = None
_stt_engine: WhisperSTT | None = None
_wake_event = threading.Event()

_muted = False
_mute_lock = threading.Lock()
_record_gain = 4.0


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


def is_muted() -> bool:
    """Return current microphone mute state."""
    with _mute_lock:
        return _muted


def _normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Apply gain normalization for weak USB microphones."""
    global _record_gain
    peak = np.max(np.abs(audio.astype(np.float64)))
    if peak < 50:
        return audio
    gain = (target_peak * 32767) / peak
    gain = min(gain, 15.0)
    _record_gain = 0.3 * gain + 0.7 * _record_gain
    _record_gain = min(_record_gain, 15.0)
    gained = np.clip(audio.astype(np.float64) * _record_gain, -32768, 32767)
    return gained.astype(np.int16)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def _on_wake_word() -> None:
    _wake_event.set()


def init() -> None:
    """Initialize wake detector and STT engine."""
    global _wake_detector, _stt_engine

    _wake_detector = WakeWordDetector(
        model_path=config.WAKE_WORD_MODEL_PATH,
        threshold=config.WAKE_WORD_THRESHOLD,
        sample_rate=config.WAKE_SAMPLE_RATE,
        mic_sample_rate=config.SAMPLE_RATE,
        mic_name=config.MIC_NAME,
        gain_target_peak=config.GAIN_TARGET_PEAK,
        allow_bundled_fallback=config.WAKE_WORD_ALLOW_BUNDLED_FALLBACK,
        bundled_model_name=config.WAKE_WORD_BUNDLED_MODEL_NAME,
        is_muted=is_muted,
    )
    config.MIC_DEVICE = _wake_detector.mic_device
    config.MIC_ALSA_DEVICE = _wake_detector.alsa_capture_device

    _stt_engine = WhisperSTT(
        whisper_path=config.WHISPER_BINARY_PATH,
        model_path=config.WHISPER_MODEL_PATH,
        language=config.WHISPER_LANGUAGE,
        threads=config.WHISPER_THREADS,
        beam_size=config.WHISPER_BEAM_SIZE,
    )

    log.info("Wake + STT components initialized")


def start_listening() -> None:
    """Start wake-word detection."""
    if _wake_detector is None:
        raise RuntimeError("Call init() before start_listening()")
    _wake_event.clear()
    _wake_detector.start(callback=_on_wake_word)
    log.info("Wake-word listener active")


def resume_listening() -> None:
    """Resume wake-word detection after command handling."""
    if _wake_detector is not None:
        _wake_detector.resume()
    log.info("Wake-word listener resumed")


def shutdown() -> None:
    """Stop wake detector thread for clean process shutdown."""
    if _wake_detector is not None:
        _wake_detector.stop()


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
    """Record audio until silence, then return 16 kHz int16 audio."""
    if is_muted():
        return np.array([], dtype=np.int16)

    log.info("Recording command…")
    audio_buffer: list[np.ndarray] = []
    recording = True
    silence_samples = 0
    blocksize = 4096
    silence_samples_needed = int(
        config.SILENCE_DURATION * config.SAMPLE_RATE / blocksize
    )
    max_chunks = int(config.MAX_RECORD_SECONDS * config.SAMPLE_RATE / blocksize)
    total_chunks = 0

    if config.MIC_DEVICE is None:
        return _record_command_with_arecord(
            blocksize=blocksize,
            silence_samples_needed=silence_samples_needed,
            max_chunks=max_chunks,
        )

    def callback(indata, frames, time_info, status):
        if status:
            log.debug("recording status: %s", status)
        if is_muted() or not recording:
            return
        audio_buffer.append(indata.copy())

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

            if len(audio_buffer) == 0:
                continue

            recent = audio_buffer[-1]

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
        return np.array([], dtype=np.int16)

    raw_audio = np.concatenate(audio_buffer, axis=0).flatten()
    normalized = _normalize(raw_audio)
    decimated = normalized[:: config.DOWNSAMPLE_FACTOR].astype(np.int16)

    log.info(
        "Recorded %d samples (%.1f s @ 16 kHz)",
        len(decimated),
        len(decimated) / config.WHISPER_SAMPLE_RATE,
    )

    return decimated


def _record_command_with_arecord(
    blocksize: int,
    silence_samples_needed: int,
    max_chunks: int,
) -> np.ndarray:
    """Record command audio using ALSA arecord when sounddevice input is unavailable."""
    if not config.MIC_ALSA_DEVICE:
        log.error(
            "No microphone backend available (sounddevice + ALSA fallback missing)"
        )
        return np.array([], dtype=np.int16)

    cmd = [
        "arecord",
        "-q",
        "-D",
        str(config.MIC_ALSA_DEVICE),
        "-f",
        "S16_LE",
        "-c",
        "1",
        "-r",
        str(config.SAMPLE_RATE),
        "-t",
        "raw",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:
        log.error("arecord start failed: %s", exc)
        return np.array([], dtype=np.int16)

    audio_buffer: list[np.ndarray] = []
    silence_samples = 0
    total_chunks = 0
    bytes_per_chunk = blocksize * 2

    try:
        while total_chunks < max_chunks:
            raw = proc.stdout.read(bytes_per_chunk) if proc.stdout else b""
            if not raw:
                continue

            total_chunks += 1
            chunk = np.frombuffer(raw, dtype=np.int16)
            if chunk.size == 0:
                continue
            audio_buffer.append(chunk.reshape(-1, 1))

            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2)) / 32768.0
            if rms < config.SILENCE_THRESHOLD:
                silence_samples += 1
                if silence_samples >= silence_samples_needed:
                    log.info("Silence detected, stopping recording.")
                    break
            else:
                silence_samples = 0
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            proc.kill()

    if not audio_buffer:
        log.warning("No audio captured.")
        return np.array([], dtype=np.int16)

    raw_audio = np.concatenate(audio_buffer, axis=0).flatten()
    normalized = _normalize(raw_audio)
    decimated = normalized[:: config.DOWNSAMPLE_FACTOR].astype(np.int16)

    log.info(
        "Recorded %d samples (%.1f s @ 16 kHz)",
        len(decimated),
        len(decimated) / config.WHISPER_SAMPLE_RATE,
    )

    return decimated


# ---------------------------------------------------------------------------
# Whisper STT  (whisper.cpp subprocess)
# ---------------------------------------------------------------------------


def transcribe(audio: np.ndarray) -> str:
    """Transcribe 16 kHz int16 audio via Whisper.cpp wrapper."""
    if len(audio) == 0:
        return ""
    if _stt_engine is None:
        raise RuntimeError("Call init() before transcribe()")

    log.info("Transcribing %d samples with whisper-cpp…", len(audio))
    try:
        text = _stt_engine.transcribe_audio_array(
            audio, sample_rate=config.WHISPER_SAMPLE_RATE
        )
        log.info("Transcription: %s", text)
        return text
    except Exception as exc:
        log.error("whisper-cpp failed: %s", exc)
        return ""
