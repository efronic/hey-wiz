"""wiz-voice — Headless AI voice assistant for Raspberry Pi 5.

Orchestrates: wake word → record → transcribe → brain bridge → speak.
"""

import asyncio
import logging
import signal

import screen_output

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


def _speak_with_ui(text: str) -> None:
    screen_output.show_state("speaking")
    voice_output.speak(text)


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("=== wiz-voice starting ===")

    # One-time model loading
    audio_pipeline.init()
    voice_output.init()
    screen_output.init()

    # Speak greeting BEFORE starting wake word detection so the
    # assistant's own voice doesn't trigger the detector.
    _speak_with_ui("Hello! I'm Wiz. Say your wake word to get my attention.")

    audio_pipeline.start_listening()
    screen_output.show_state("idle")
    log.info("Listening for wake word.")

    try:
        while not _shutdown:
            screen_output.show_state("idle")
            audio_pipeline.wait_for_wake()
            if _shutdown:
                break

            log.info("Wake word heard — recording command…")
            screen_output.show_state("listening")

            try:
                audio = audio_pipeline.record_command()

                if len(audio) == 0:
                    _speak_with_ui("Sorry, I didn't catch that.")
                    continue

                # Play filler while processing to reduce perceived latency
                screen_output.show_state("speaking")
                voice_output.speak_filler()

                screen_output.show_state("thinking")
                text = audio_pipeline.transcribe(audio)

                if not text:
                    _speak_with_ui("Sorry, I didn't catch that.")
                    continue

                log.info("User said: %s", text)
                response = asyncio.run(brain_bridge.process(text))
                _speak_with_ui(response)

            except Exception:
                screen_output.show_state("error")
                log.exception("Pipeline error")
                _speak_with_ui("Something went wrong. Please try again.")
            finally:
                # Resume wake-word detection after the full cycle completes
                audio_pipeline.resume_listening()
                screen_output.show_state("idle")
    finally:
        audio_pipeline.shutdown()
        screen_output.shutdown()

    log.info("=== wiz-voice stopped ===")


if __name__ == "__main__":
    main()
