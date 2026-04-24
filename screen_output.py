"""Screen output facade for UI state transitions."""

from __future__ import annotations

import logging

from ui.ui_manager import UIManager, UIState

import config

log = logging.getLogger(__name__)

_ui: UIManager | None = None


def init() -> None:
    """Initialize and start the touchscreen UI if enabled."""
    global _ui
    if not config.ENABLE_UI:
        log.info("UI disabled (ENABLE_UI=false)")
        return

    try:
        _ui = UIManager(
            width=config.DISPLAY_WIDTH,
            height=config.DISPLAY_HEIGHT,
            assets_path=str(config.UI_ASSETS_PATH),
            fps=config.UI_FPS,
            backend=config.UI_BACKEND,
            use_framebuffer_fallback=config.UI_FBCON_FALLBACK,
        )
        _ui.start()
        _ui.set_state(UIState.IDLE)
        log.info("Touchscreen UI started")
    except Exception as exc:
        _ui = None
        log.warning("UI unavailable; running headless: %s", exc)


def show_state(state_name: str) -> None:
    """Set UI state by string name."""
    if not _ui:
        return
    mapping = {
        "idle": UIState.IDLE,
        "listening": UIState.LISTENING,
        "thinking": UIState.THINKING,
        "speaking": UIState.SPEAKING,
        "error": UIState.ERROR,
    }
    state = mapping.get(state_name.lower())
    if state is not None:
        _ui.set_state(state)


def set_audio_amplitude(amplitude: float) -> None:
    if _ui:
        _ui.set_audio_amplitude(amplitude)


def shutdown() -> None:
    """Stop the UI render thread if running."""
    global _ui
    if _ui:
        _ui.stop()
        _ui = None
        log.info("Touchscreen UI stopped")
