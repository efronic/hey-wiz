"""Whisper.cpp STT wrapper adapted from pibot_local_agent."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import wave
from pathlib import Path


log = logging.getLogger(__name__)


class WhisperSTT:
    """Whisper.cpp speech-to-text engine."""

    def __init__(
        self,
        whisper_path: str = "/usr/local/bin/whisper-cpp",
        model_path: str = "",
        language: str = "en",
        threads: int = 4,
        beam_size: int = 5,
    ):
        self.whisper_path = whisper_path
        self.model_path = model_path
        self.language = language
        self.threads = threads
        self.beam_size = beam_size

        if not Path(self.whisper_path).exists():
            alt_paths = [
                "/usr/local/bin/whisper-cli",
                "/usr/bin/whisper-cpp",
                "/usr/bin/whisper-cli",
            ]
            for alt in alt_paths:
                if Path(alt).exists():
                    self.whisper_path = alt
                    break
            else:
                raise FileNotFoundError(f"Whisper not found at {whisper_path}")

        self.model_path = self._resolve_model_path(self.model_path)

    def _resolve_model_path(self, configured_path: str) -> str:
        """Resolve a usable Whisper model path from configured and common local locations."""
        configured = Path(configured_path) if configured_path else None
        if configured and configured.exists():
            return str(configured)

        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            project_root / "models" / "ggml-small.en-q8_0.bin",
            project_root / "models" / "ggml-small.en.bin",
            project_root / "models" / "ggml-base.en.bin",
            project_root / "whisper.cpp" / "models" / "for-tests-ggml-small.en.bin",
            project_root / "whisper.cpp" / "models" / "for-tests-ggml-base.en.bin",
        ]

        for candidate in candidates:
            if candidate.exists():
                if configured_path:
                    log.warning(
                        "Whisper model not found at %s; using %s",
                        configured_path,
                        candidate,
                    )
                else:
                    log.info("Using Whisper model: %s", candidate)
                return str(candidate)

        raise FileNotFoundError(
            "Model not found at {} and no fallback model was found in local models/ or whisper.cpp/models/."
            .format(configured_path)
        )

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a 16 kHz mono WAV file into text."""
        process = subprocess.run(
            [
                self.whisper_path,
                "-m",
                self.model_path,
                "-f",
                audio_path,
                "-l",
                self.language,
                "-t",
                str(self.threads),
                "-bs",
                str(self.beam_size),
                "--no-timestamps",
                "-np",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if process.returncode != 0:
            raise RuntimeError(f"Whisper failed: {process.stderr}")

        text = process.stdout.strip()
        text = text.replace("[BLANK_AUDIO]", "").strip()
        return text

    def transcribe_audio_array(self, audio, sample_rate: int = 16000) -> str:
        """Transcribe a numpy int16 audio array by writing a temporary WAV."""
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio.tobytes())

            return self.transcribe(temp_path)
        finally:
            os.unlink(temp_path)
