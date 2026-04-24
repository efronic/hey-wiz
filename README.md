# wiz-voice

Offline-first Raspberry Pi voice assistant with:
- Custom openWakeWord wake model (preferred)
- Local Whisper.cpp speech-to-text
- OpenClaw gateway for response generation
- Piper TTS for spoken output
- Touchscreen UI state display (5-inch, Wayland-first)

## What Changed in feat/new-voice-detection

This branch migrates wake-word, STT, and screen logic to the pibot_local_agent style while keeping the existing OpenClaw flow in place.

### Migrated Components
- Wake-word detector: [senses/wake_word_detector.py](senses/wake_word_detector.py)
- STT wrapper: [audio/stt_engine.py](audio/stt_engine.py)
- Touchscreen UI manager: [ui/ui_manager.py](ui/ui_manager.py)
- UI facade: [screen_output.py](screen_output.py)
- Pipeline integration: [audio_pipeline.py](audio_pipeline.py)
- Main loop state transitions: [main.py](main.py)

## Runtime Flow

1. Wake detector listens continuously in a background thread.
2. On wake event, recording starts and UI enters LISTENING.
3. Audio is downsampled to 16 kHz and transcribed via Whisper.cpp.
4. Transcribed text is routed through OpenClaw in [brain_bridge.py](brain_bridge.py).
5. Piper TTS speaks the response and UI enters SPEAKING.
6. UI returns to IDLE and wake listening resumes.

UI states:
- IDLE
- LISTENING
- THINKING
- SPEAKING
- ERROR

## Prerequisites

- Raspberry Pi 5 (recommended)
- USB mic and speaker
- 5-inch touch display
- Wayland session preferred (fallback to fbcon supported)
- Custom wake model ONNX file (optional if bundled fallback is enabled)

## Setup

Run:

```bash
chmod +x setup_pi.sh
./setup_pi.sh
```

Then:

1. Copy .env.example to .env.
2. Set OPENCLAW credentials.
3. Set WAKE_WORD_MODEL_PATH to your custom wake model ONNX file (or keep bundled fallback enabled).
4. Activate venv and run.

## Environment Variables

Required:
- OPENCLAW_TOKEN

Wake model controls:
- WAKE_WORD_MODEL_PATH
- WAKE_WORD_ALLOW_BUNDLED_FALLBACK=true
- WAKE_WORD_BUNDLED_MODEL_NAME=hey_jarvis

Key optional UI variables:
- ENABLE_UI=true
- DISPLAY_WIDTH=800
- DISPLAY_HEIGHT=480
- UI_BACKEND=wayland
- UI_FBCON_FALLBACK=true
- UI_ASSETS_PATH=./assets/face

Key optional audio device variables:
- MIC_NAME=USB PnP Sound Device
- SPEAKER_NAME=UACDemoV1.0

## Run

```bash
source venv/bin/activate
python main.py
```

## Troubleshooting

- Error: "Gateway auth failed: unauthorized: gateway token mismatch"
	- Cause: OPENCLAW_TOKEN is missing, still set to placeholder, or stale.
	- Fix: Open the OpenClaw dashboard URL, copy the gateway token from Control UI settings, and paste it into OPENCLAW_TOKEN in .env.

- Error: "Mic 'USB PnP Sound Device' not found"
	- Cause: MIC_NAME does not match the current USB audio device label.
	- Fix: Set MIC_NAME in .env to your actual input device name (for example MIC_NAME=UACDemoV1.0).

## Worktree

Implementation was done in worktree:
- path: /home/efron/projects/wiz-voice-2
- branch: feat/new-voice-detection
