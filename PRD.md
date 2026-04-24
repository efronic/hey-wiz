# wiz-voice Product Requirements Document

## 1. Overview

wiz-voice is an always-on Raspberry Pi assistant that now uses:
- pibot-style wake-word detection logic
- pibot-style Whisper.cpp STT wrapper logic
- a touchscreen UI state layer for a 5-inch display
- existing OpenClaw routing and response flow (unchanged)

This document reflects the migration implemented in branch feat/new-voice-detection in worktree /home/efron/projects/wiz-voice-2.

## 2. Goals

1. Replace prior wake-word internals with pibot_local_agent wake detector behavior.
2. Replace prior transcription internals with pibot_local_agent STT wrapper behavior.
3. Add visual state feedback on the connected 5-inch touch display.
4. Keep current OpenClaw gateway and command-tag brain behavior intact.

## 3. Non-Goals

1. No migration to pibot Ollama router/toolchain.
2. No change to the current OpenClaw WebSocket response model.
3. No requirement for cloud-only STT or wake services.

## 4. System Architecture

```text
Microphone (48kHz)
  -> WakeWordDetector (threaded, queue-based, custom ONNX model)
  -> wake event
  -> audio_pipeline.record_command()
  -> WhisperSTT.transcribe_audio_array() (16kHz)
  -> brain_bridge.process() (OpenClaw flow)
  -> voice_output.speak()
  -> Speaker

Touchscreen UI (5-inch)
  <- screen_output.show_state()
  <- main loop + error handling
```

### 4.1 Runtime State Machine

- IDLE: waiting for wake word
- LISTENING: recording user command
- THINKING: transcribing and processing
- SPEAKING: TTS playback
- ERROR: temporary failure state

## 5. Module Design

### 5.1 Wake-Word Module

File: [senses/wake_word_detector.py](senses/wake_word_detector.py)

Requirements:
1. Use openWakeWord model API.
2. Require custom model path (no bundled fallback mode).
3. Run in background thread.
4. Support pause/resume to free mic during capture and speech.
5. Use adaptive gain normalization and mic queue buffering.

### 5.2 STT Module

File: [audio/stt_engine.py](audio/stt_engine.py)

Requirements:
1. Wrap whisper-cpp execution.
2. Validate binary and model paths at startup.
3. Support file-based transcription and numpy-array transcription.
4. Remove known artifacts such as [BLANK_AUDIO] from output.

### 5.3 Pipeline Integration

File: [audio_pipeline.py](audio_pipeline.py)

Requirements:
1. Keep existing callable surface for main loop:
   - init()
   - start_listening()
   - wait_for_wake()
   - record_command()
   - transcribe()
   - resume_listening()
   - shutdown()
2. Delegate wake detection to WakeWordDetector.
3. Delegate STT to WhisperSTT.
4. Maintain mic mute/unmute interoperability with TTS.

### 5.4 Touchscreen UI Module

Files:
- [ui/ui_manager.py](ui/ui_manager.py)
- [screen_output.py](screen_output.py)

Requirements:
1. Provide UIState enum: IDLE, LISTENING, THINKING, SPEAKING, ERROR.
2. Render full-screen at 800x480 by default.
3. Use Wayland driver first.
4. Allow fbcon fallback when configured.
5. Use PNG face assets when present, procedural fallback otherwise.
6. Run safely in background thread and degrade to headless if init fails.

### 5.5 Main Orchestration

File: [main.py](main.py)

Requirements:
1. Keep existing OpenClaw request/response behavior.
2. Add UI state transitions around each stage.
3. Ensure clean shutdown of wake detector and UI thread.
4. Keep signal handling and loop resilience.

## 6. Configuration

File: [config.py](config.py)

### 6.1 Required Configuration

- OPENCLAW_TOKEN is required.
- WAKE_WORD_MODEL_PATH is recommended for custom wake behavior.
- When WAKE_WORD_MODEL_PATH is invalid or missing, bundled fallback can be used.

### 6.2 Primary Runtime Configuration

Wake/STT:
- WAKE_WORD_MODEL_PATH
- WAKE_WORD_THRESHOLD
- WAKE_SAMPLE_RATE
- WAKE_WORD_ALLOW_BUNDLED_FALLBACK
- WAKE_WORD_BUNDLED_MODEL_NAME
- SAMPLE_RATE
- WHISPER_BINARY_PATH
- WHISPER_MODEL_PATH
- WHISPER_THREADS
- WHISPER_BEAM_SIZE

Display/UI:
- ENABLE_UI
- DISPLAY_WIDTH
- DISPLAY_HEIGHT
- UI_FPS
- UI_BACKEND
- UI_FBCON_FALLBACK
- UI_ASSETS_PATH

OpenClaw/Vision/TTS settings remain in existing config scope.

## 7. Setup and Deployment

### 7.1 Installer

File: [setup_pi.sh](setup_pi.sh)

Requirements:
1. Install camera/audio dependencies.
2. Install SDL runtime dependencies for pygame UI.
3. Build/install whisper-cpp.
4. Install Python requirements.
5. Create wake_word model directory and face assets directory.
6. Warn user to set WAKE_WORD_MODEL_PATH in .env.

### 7.2 Environment Template

File: [.env.example](.env.example)

Requirements:
1. Include wake model path variable.
2. Include bundled wake fallback controls.
3. Include UI variables for touchscreen behavior.
4. Keep OpenClaw and Vision templates intact.

## 8. Documentation

### 8.1 Repository Documentation

File: [README.md](README.md)

Requirements:
1. Describe migrated architecture and unchanged OpenClaw boundary.
2. Document setup and run instructions.
3. Document wake model custom path and bundled fallback behavior.
4. Document touchscreen UI behavior and backend options.
5. Record worktree path and branch used for migration.

### 8.2 Product Document

File: [PRD.md](PRD.md)

Requirement:
- This PRD must remain aligned with implemented behavior in feat/new-voice-detection.

## 9. Acceptance Criteria

1. App starts with custom wake model when available, or bundled fallback when enabled.
2. Wake detection triggers recording reliably.
3. STT returns text from recorded speech through Whisper.cpp wrapper.
4. OpenClaw processing remains functional.
5. UI transitions correctly through IDLE, LISTENING, THINKING, SPEAKING.
6. Error path sets ERROR state and recovers to IDLE.
7. Clean shutdown stops wake detector and UI thread.

## 10. Verification Checklist

1. Static checks report no Python errors in workspace.
2. Manual flow test:
   - wake word
   - record
   - transcribe
   - OpenClaw response
   - spoken output
3. Touchscreen UI visible during run and reflects lifecycle states.
4. If display init fails, assistant still runs headless.
5. If wake model path is invalid and fallback is disabled, startup fails with clear message.

## 11. Risks and Mitigations

1. Missing custom wake model file:
   - Mitigation: fail fast at startup with explicit path error.
2. Wayland display init failure:
   - Mitigation: optional fbcon fallback and headless-safe UI facade.
3. Mic/speaker hardware name mismatch:
   - Mitigation: device name config remains tunable via environment.
4. Thread shutdown edge cases:
   - Mitigation: explicit shutdown hooks in main loop.

## 12. Future Enhancements

1. Touch interaction controls (tap-to-mute, tap-to-retry).
2. Live amplitude visualization from mic and TTS signals.
3. Optional conversational memory layer.
4. Additional face assets tuned for state transitions.
