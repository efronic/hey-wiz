"""wiz-voice — Headless AI voice assistant for Raspberry Pi 5.

Orchestrates: wake word → record → transcribe → brain bridge → speak.
"""

import asyncio
import logging
import signal
import sys

import audio_pipeline
import brain_bridge
import voice_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-18s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wiz-voice")

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    log.info("Received signal %s — shutting down.", signum)
    _shutdown = True


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("=== wiz-voice starting ===")

    # One-time model loading
    audio_pipeline.init()
    audio_pipeline.start_listening()

    log.info("Listening for wake word. Say 'Hey Jarvis' to begin.")

    while not _shutdown:
        audio_pipeline.wait_for_wake()
        if _shutdown:
            break

        log.info("Wake word heard — recording command…")

        try:
            audio = audio_pipeline.record_command()
            text = audio_pipeline.transcribe(audio)

            if not text:
                voice_output.speak("Sorry, I didn't catch that.")
                continue

            log.info("User said: %s", text)
            response = asyncio.run(brain_bridge.process(text))
            voice_output.speak(response)

        except Exception:
            log.exception("Pipeline error")
            voice_output.speak("Something went wrong. Please try again.")

    log.info("=== wiz-voice stopped ===")


if __name__ == "__main__":
    main()
