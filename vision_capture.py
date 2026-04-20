"""Vision capture via Pi Camera Module 3 (picamera2 + libcamera)."""

import logging
import time

import config

log = logging.getLogger(__name__)


def capture_image() -> str:
    """Capture a high-res JPEG using Picamera2 with continuous autofocus.

    Returns the file path to the captured JPEG image.
    """
    from picamera2 import Picamera2
    from libcamera import controls  # type: ignore[import-untyped]

    picam2 = Picamera2()
    try:
        still_config = picam2.create_still_configuration(
            main={"size": (1920, 1080), "format": "RGB888"},
        )
        picam2.configure(still_config)
        picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        picam2.start()

        # Allow autofocus to settle
        time.sleep(1.0)

        picam2.capture_file(config.CAPTURE_PATH)
        log.info("Image captured: %s", config.CAPTURE_PATH)
        return config.CAPTURE_PATH
    finally:
        picam2.stop()
        picam2.close()
        log.debug("Camera released.")
