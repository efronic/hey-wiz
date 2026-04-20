"""Voice output: text-to-speech via Piper and playback via ALSA."""

import logging
import re
import subprocess

import config

log = logging.getLogger(__name__)


def _sanitize(text: str) -> str:
    """Strip markdown artefacts and cap length."""
    text = re.sub(r"[*_`#\[\]]", "", text)
    text = text.strip()
    if len(text) > config.MAX_TTS_CHARS:
        text = text[: config.MAX_TTS_CHARS] + "…"
    return text


def speak(text: str) -> None:
    """Synthesise *text* with Piper and play via the default ALSA device.

    Blocks until playback completes.
    """
    text = _sanitize(text)
    if not text:
        log.warning("Nothing to speak.")
        return

    log.info("Speaking: %s", text[:80])

    cmd = (
        f'echo {_shell_quote(text)} | '
        f'{config.PIPER_BINARY} --model {config.PIPER_MODEL_PATH} --output_raw | '
        f'aplay -D plughw:2,0 -r 22050 -f S16_LE -t raw -q'
    )
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=30)
    except subprocess.TimeoutExpired:
        log.error("TTS playback timed out.")
    except subprocess.CalledProcessError as exc:
        log.error("TTS pipeline failed: %s", exc)


def _shell_quote(text: str) -> str:
    """Safely quote text for shell interpolation."""
    import shlex
    return shlex.quote(text)
