"""wiz-voice — Headless AI voice assistant for Raspberry Pi 5.

Orchestrates: wake word → record → transcribe → brain bridge → speak.
"""

import asyncio
import logging
import signal

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
    voice_output.init()

    # Speak greeting BEFORE starting wake word detection so the
    # assistant's own voice doesn't trigger the detector.
    voice_output.speak("Hello! I'm Wiz. Say hey Jarvis to get my attention.")

    audio_pipeline.start_listening()
    log.info("Listening for wake word. Say 'Hey Jarvis' to begin.")

    while not _shutdown:
        audio_pipeline.wait_for_wake()
        if _shutdown:
            break

        log.info("Wake word heard — recording command…")

        try:
            audio = audio_pipeline.record_command()

            if len(audio) == 0:
                voice_output.speak("Sorry, I didn't catch that.")
                audio_pipeline.resume_listening()
                continue

            # Play filler while processing to reduce perceived latency
            voice_output.speak_filler()

            text = audio_pipeline.transcribe(audio)

            if not text:
                voice_output.speak("Sorry, I didn't catch that.")
                audio_pipeline.resume_listening()
                continue

            log.info("User said: %s", text)
            response = asyncio.run(brain_bridge.process(text))
            voice_output.speak(response)

        except Exception:
            log.exception("Pipeline error")
            voice_output.speak("Something went wrong. Please try again.")

        # Resume wake-word detection after the full cycle completes
        audio_pipeline.resume_listening()

    log.info("=== wiz-voice stopped ===")


if __name__ == "__main__":
    main()
