"""PyGame-based UI manager for Raspberry Pi touch displays."""

from __future__ import annotations

import math
import os
from enum import Enum
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Dict, Optional

import pygame


class UIState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


class UIManager:
    """Manages animated assistant UI on a 5-inch touchscreen."""

    def __init__(
        self,
        width: int = 800,
        height: int = 480,
        assets_path: str = "",
        fps: int = 30,
        backend: str = "wayland",
        use_framebuffer_fallback: bool = True,
    ):
        self.width = width
        self.height = height
        self.assets_path = Path(assets_path) if assets_path else Path("assets/face")
        self.fps = fps
        self.backend = backend
        self.use_framebuffer_fallback = use_framebuffer_fallback

        self._state = UIState.IDLE
        self._state_lock = Lock()
        self._running = False
        self._ready = Event()
        self._thread: Optional[Thread] = None

        self._audio_amplitude = 0.0
        self._frame_count = 0

        self.bg_color = (10, 10, 20)
        self.accent_color = (0, 200, 255)

        self._faces: Dict[str, pygame.Surface] = {}
        self._face_rects: Dict[str, pygame.Rect] = {}

    def set_state(self, state: UIState):
        """Set current UI state in a thread-safe way."""
        with self._state_lock:
            self._state = state

    def set_audio_amplitude(self, amplitude: float):
        """Set speaking/listening amplitude in range [0.0, 1.0]."""
        self._audio_amplitude = max(0.0, min(1.0, amplitude))

    def start(self):
        """Start the render loop in a background thread."""
        self._running = True
        self._thread = Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self):
        """Stop the render loop and join the thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def _init_display(self, driver: str):
        """Initialize pygame display with a specific SDL video driver."""
        os.environ["SDL_VIDEODRIVER"] = driver

        if driver == "wayland":
            os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
            os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
        elif driver == "fbcon":
            os.environ.setdefault("SDL_FBDEV", "/dev/fb0")

        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF,
        )
        pygame.display.set_caption("Wiz Voice")
        try:
            pygame.mouse.set_visible(False)
        except Exception:
            pass
        return screen

    def _render_loop(self):
        """Main render loop that owns pygame context."""
        screen = None
        used_driver = None

        drivers = [self.backend]
        if self.backend != "fbcon" and self.use_framebuffer_fallback:
            drivers.append("fbcon")

        for driver in drivers:
            try:
                screen = self._init_display(driver)
                used_driver = driver
                break
            except pygame.error:
                pygame.display.quit()
                pygame.font.quit()

        if screen is None:
            self._ready.set()
            return

        font_small = pygame.font.Font(None, 24)
        font_medium = pygame.font.Font(None, 36)

        self._load_faces()
        self._ready.set()

        clock = pygame.time.Clock()

        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False
                    break

            if not self._running:
                break

            screen.fill(self.bg_color)
            with self._state_lock:
                state = self._state

            if state == UIState.IDLE:
                self._render_idle(screen)
            elif state == UIState.LISTENING:
                self._render_listening(screen)
            elif state == UIState.THINKING:
                self._render_thinking(screen)
            elif state == UIState.SPEAKING:
                self._render_speaking(screen)
            elif state == UIState.ERROR:
                self._render_error(screen, font_medium)

            status = f"Status: {state.value.upper()} ({used_driver})"
            text_surface = font_small.render(status, True, (100, 100, 100))
            text_rect = text_surface.get_rect(center=(self.width // 2, self.height - 20))
            screen.blit(text_surface, text_rect)

            try:
                pygame.display.flip()
            except pygame.error:
                break

            self._frame_count += 1
            clock.tick(self.fps)

        pygame.display.quit()
        pygame.font.quit()

    def _load_faces(self):
        """Load PNG face assets and scale to the display height."""
        target_height = int(self.height * 0.8)
        for png_file in self.assets_path.glob("*.png"):
            stem = png_file.stem
            try:
                surface = pygame.image.load(str(png_file)).convert_alpha()
                orig_w, orig_h = surface.get_size()
                scale = target_height / max(1, orig_h)
                new_w = int(orig_w * scale)
                surface = pygame.transform.smoothscale(surface, (new_w, target_height))
                rect = surface.get_rect(center=(self.width // 2, self.height // 2))
                self._faces[stem] = surface
                self._face_rects[stem] = rect
            except Exception:
                continue

    def _blit_face(self, screen, name: str) -> bool:
        if name in self._faces:
            screen.blit(self._faces[name], self._face_rects[name])
            return True
        return False

    def _render_idle(self, screen):
        blink_interval = self.fps * 3
        blink_duration = self.fps // 3
        is_blinking = (self._frame_count % blink_interval) < blink_duration

        if is_blinking and "winking" in self._faces:
            self._blit_face(screen, "winking")
        elif not self._blit_face(screen, "happy"):
            self._draw_procedural_face(screen, eyes_closed=is_blinking, mouth_openness=0.0)

    def _render_listening(self, screen):
        if not self._blit_face(screen, "happy_eye_glistening"):
            self._draw_procedural_face(screen, eyes_closed=False, mouth_openness=0.0, eye_scale=1.3)

    def _render_thinking(self, screen):
        if not self._blit_face(screen, "thinking"):
            self._draw_procedural_face(screen, eyes_closed=False, mouth_openness=0.0)

        cx, cy = self.width // 2, self.height - 60
        num_dots = 8
        angle_offset = self._frame_count * 5
        for i in range(num_dots):
            angle = math.radians(angle_offset + i * (360 / num_dots))
            x = cx + int(25 * math.cos(angle))
            y = cy + int(25 * math.sin(angle))
            dot_size = max(2, 6 - int((i / num_dots) * 3))
            pygame.draw.circle(screen, self.accent_color, (x, y), dot_size)

    def _render_speaking(self, screen):
        if not self._blit_face(screen, "happy"):
            mouth = self._audio_amplitude * 0.8 + 0.1
            self._draw_procedural_face(screen, eyes_closed=False, mouth_openness=mouth)

    def _render_error(self, screen, font):
        if not self._blit_face(screen, "irritated"):
            self._draw_procedural_face(screen, eyes_closed=True, mouth_openness=0.3)

        overlay = pygame.Surface((self.width, self.height))
        overlay.fill((255, 0, 0))
        overlay.set_alpha(30)
        screen.blit(overlay, (0, 0))

        text = font.render("Error", True, (255, 100, 100))
        screen.blit(text, text.get_rect(center=(self.width // 2, self.height // 2 + 150)))

    def _draw_procedural_face(
        self,
        screen,
        eyes_closed: bool = False,
        mouth_openness: float = 0.0,
        eye_scale: float = 1.0,
    ):
        """Draw a fallback face when PNG assets are unavailable."""
        cx = self.width // 2
        cy = self.height // 2 - 30
        eye_width = int(80 * eye_scale)
        eye_height = int(50 * eye_scale) if not eyes_closed else 8
        spacing = 120
        eye_y = cy - 30

        pygame.draw.ellipse(
            screen,
            self.accent_color,
            (cx - spacing - eye_width // 2, eye_y - eye_height // 2, eye_width, eye_height),
            3,
        )
        pygame.draw.ellipse(
            screen,
            self.accent_color,
            (cx + spacing - eye_width // 2, eye_y - eye_height // 2, eye_width, eye_height),
            3,
        )

        if not eyes_closed:
            pupil_size = int(15 * eye_scale)
            pygame.draw.circle(screen, self.accent_color, (cx - spacing, eye_y), pupil_size)
            pygame.draw.circle(screen, self.accent_color, (cx + spacing, eye_y), pupil_size)

        mouth_y = cy + 60
        mouth_w = 100
        mouth_h = int(20 + mouth_openness * 40)
        if mouth_openness > 0.1:
            pygame.draw.ellipse(
                screen,
                self.accent_color,
                (cx - mouth_w // 2, mouth_y - mouth_h // 2, mouth_w, mouth_h),
                3,
            )
        else:
            pygame.draw.line(
                screen,
                self.accent_color,
                (cx - mouth_w // 2, mouth_y),
                (cx + mouth_w // 2, mouth_y),
                3,
            )
