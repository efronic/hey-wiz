"""Wake-word detection using openWakeWord, adapted from pibot_local_agent."""

from __future__ import annotations

import logging
from pathlib import Path
from queue import Empty, Queue
import re
import subprocess
from threading import Event, Thread
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

try:
    from openwakeword.model import Model
    import openwakeword

    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False

log = logging.getLogger(__name__)


def _find_mic_device(name_substring: str) -> Optional[int]:
    """Find microphone device index by name substring with sensible fallbacks."""
    devices = sd.query_devices()
    input_devices = [
        (i, d) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0
    ]

    if name_substring:
        needle = name_substring.lower()
        for i, device in input_devices:
            if needle in device["name"].lower():
                return i

    default_index = sd.default.device[0] if sd.default.device else None
    if isinstance(default_index, int) and default_index >= 0:
        if default_index < len(devices) and devices[default_index].get("max_input_channels", 0) > 0:
            log.warning(
                "Mic '%s' not found; using default input device %s (%s)",
                name_substring,
                default_index,
                devices[default_index]["name"],
            )
            return int(default_index)

    if input_devices:
        fallback_index, fallback_device = input_devices[0]
        log.warning(
            "Mic '%s' not found; using first input device %s (%s)",
            name_substring,
            fallback_index,
            fallback_device["name"],
        )
        return fallback_index

    available = [(i, d["name"]) for i, d in enumerate(devices)]
    log.warning(
        "No input-capable sounddevice mic found for '%s'. Available devices: %s",
        name_substring,
        available,
    )
    return None


def _find_alsa_capture_device(name_substring: str) -> str:
    """Find ALSA capture device string (plughw:CARD,DEV), preferring name match."""
    try:
        result = subprocess.run(
            ["arecord", "-l"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError as exc:
        raise RuntimeError("arecord not available for ALSA capture fallback") from exc

    pattern = re.compile(
        r"card\s+(\d+):\s*[^\[]*\[([^\]]+)\],\s*device\s+(\d+):\s*[^\[]*\[([^\]]+)\]"
    )
    matches: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        card_num, card_name, device_num, device_name = match.groups()
        alsa_device = f"plughw:{card_num},{device_num}"
        label = f"{card_name} {device_name}".strip()
        matches.append((alsa_device, label))

    if not matches:
        raise RuntimeError("No ALSA capture devices found from arecord -l")

    needle = (name_substring or "").lower()
    if needle:
        for alsa_device, label in matches:
            if needle in label.lower():
                return alsa_device

    return matches[0][0]


def _find_bundled_model(name: str) -> str:
    """Find a bundled openWakeWord model by name."""
    pkg_dir = Path(openwakeword.__file__).parent / "resources" / "models"
    for model_file in pkg_dir.glob(f"{name}*.onnx"):
        return str(model_file)
    raise FileNotFoundError(f"Bundled model '{name}' not found in {pkg_dir}")


class WakeWordDetector:
    """Detect wake words via openWakeWord in a background thread."""

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        mic_sample_rate: int = 48000,
        mic_name: str = "USB PnP Sound Device",
        gain_target_peak: float = 0.9,
        allow_bundled_fallback: bool = True,
        bundled_model_name: str = "hey_jarvis",
        is_muted: Optional[Callable[[], bool]] = None,
    ):
        if not OPENWAKEWORD_AVAILABLE:
            raise RuntimeError("openwakeword not installed. Run: pip install openwakeword")

        self.threshold = threshold
        self.sample_rate = sample_rate
        self.mic_sample_rate = mic_sample_rate
        self.mic_name = mic_name
        self.gain_target_peak = gain_target_peak
        self._is_muted = is_muted

        self.downsample_factor = max(1, self.mic_sample_rate // self.sample_rate)
        self.mic_chunk_size = int(0.08 * self.mic_sample_rate)

        self.mic_device = _find_mic_device(self.mic_name)
        self.alsa_capture_device: Optional[str] = None
        self.capture_backend = "sounddevice"
        if self.mic_device is None:
            self.alsa_capture_device = _find_alsa_capture_device(self.mic_name)
            self.capture_backend = "arecord"

        use_custom = bool(model_path and Path(model_path).exists())
        if use_custom:
            resolved_model_path = model_path
        elif allow_bundled_fallback:
            resolved_model_path = _find_bundled_model(bundled_model_name)
            log.warning(
                "Wake model not found at %s; falling back to bundled model %s",
                model_path,
                resolved_model_path,
            )
        else:
            raise FileNotFoundError(
                "Wake-word model not found: {}. "
                "Set WAKE_WORD_MODEL_PATH to a valid .onnx file or enable "
                "WAKE_WORD_ALLOW_BUNDLED_FALLBACK.".format(model_path)
            )

        self.model = Model(wakeword_model_paths=[resolved_model_path])

        self._running = False
        self._stop_event = Event()
        self._resume_event = Event()
        self._thread: Optional[Thread] = None
        self._callback: Optional[Callable[[], None]] = None
        self._paused = False
        self._audio_queue: Queue[bytes] = Queue()
        self._gain = 4.0

        if self.capture_backend == "sounddevice":
            log.info("Wake word mic: device %s (%s)", self.mic_device, self.mic_name)
        else:
            log.info(
                "Wake word mic using ALSA fallback: %s (requested '%s')",
                self.alsa_capture_device,
                self.mic_name,
            )
        log.info("Wake model loaded: %s", resolved_model_path)

    def start(self, callback: Callable[[], None]):
        """Start listening for wake-word detections."""
        self._callback = callback
        self._running = True
        self._paused = False
        self._stop_event.clear()
        self._resume_event.set()

        self._thread = Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop listening and join the background thread."""
        self._running = False
        self._stop_event.set()
        self._resume_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def pause(self):
        """Pause detection and release mic stream."""
        self._paused = True
        self._resume_event.clear()

    def resume(self):
        """Resume detection after pause."""
        self._paused = False
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except Empty:
                break
        self._resume_event.set()

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Apply adaptive gain normalization for weak USB mics."""
        peak = np.max(np.abs(audio))
        if peak < 50:
            return audio.astype(np.int16)

        target = self.gain_target_peak * 32767
        desired_gain = target / peak
        desired_gain = min(desired_gain, 15.0)

        self._gain = 0.3 * desired_gain + 0.7 * self._gain
        self._gain = min(self._gain, 15.0)

        gained = np.clip(audio * self._gain, -32768, 32767)
        return gained.astype(np.int16)

    def _listen_loop(self):
        """Main listening loop, reopening stream each pause/resume cycle."""
        while self._running:
            self._resume_event.wait()
            if not self._running:
                break

            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except Empty:
                    break

            if self.capture_backend == "sounddevice":
                detected = self._listen_with_sounddevice()
            else:
                detected = self._listen_with_arecord()

            if detected and self._callback:
                self._paused = True
                self._resume_event.clear()
                self.model.reset()
                try:
                    self._callback()
                except Exception:
                    log.exception("Wake-word callback failed")

    def _detect_from_raw(self, raw: bytes) -> bool:
        """Run wake-word prediction for one raw PCM chunk."""
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        normalized = self._normalize(audio)
        decimated = normalized[:: self.downsample_factor]

        predictions = self.model.predict(decimated)
        for model_name, score in predictions.items():
            if score >= self.threshold:
                log.info("Wake word detected: %s (score=%.3f)", model_name, score)
                return True
        return False

    def _listen_with_sounddevice(self) -> bool:
        """Listen for wake words using sounddevice RawInputStream."""
        def audio_callback(indata, frames, time_info, status):
            if status:
                log.debug("Wake-word audio status: %s", status)
            if self._paused or not self._running:
                return
            if self._is_muted and self._is_muted():
                return
            self._audio_queue.put(bytes(indata))

        try:
            stream = sd.RawInputStream(
                device=self.mic_device,
                samplerate=self.mic_sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.mic_chunk_size,
                latency="high",
                callback=audio_callback,
            )
            stream.start()
        except Exception as exc:
            log.error("Wake-word stream error: %s", exc)
            if self._running:
                self._stop_event.wait(timeout=1.0)
            return False

        detected = False
        try:
            while self._running and not self._paused:
                try:
                    raw = self._audio_queue.get(timeout=0.1)
                except Empty:
                    continue
                if self._detect_from_raw(raw):
                    detected = True
                    break
        finally:
            stream.stop()
            stream.close()

        return detected

    def _listen_with_arecord(self) -> bool:
        """Listen for wake words using ALSA arecord raw stream fallback."""
        if not self.alsa_capture_device:
            log.error("ALSA fallback selected but no capture device was resolved")
            return False

        cmd = [
            "arecord",
            "-q",
            "-D",
            self.alsa_capture_device,
            "-f",
            "S16_LE",
            "-c",
            "1",
            "-r",
            str(self.mic_sample_rate),
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
            log.error("Wake-word arecord start failed: %s", exc)
            if self._running:
                self._stop_event.wait(timeout=1.0)
            return False

        detected = False
        bytes_per_chunk = self.mic_chunk_size * 2
        try:
            while self._running and not self._paused:
                if self._is_muted and self._is_muted():
                    self._stop_event.wait(timeout=0.05)
                    continue

                raw = proc.stdout.read(bytes_per_chunk) if proc.stdout else b""
                if not raw:
                    self._stop_event.wait(timeout=0.05)
                    continue

                if self._detect_from_raw(raw):
                    detected = True
                    break
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                proc.kill()

        return detected
