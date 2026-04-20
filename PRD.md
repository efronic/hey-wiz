# wiz-voice — Product Requirements Document

## 1. Overview

**wiz-voice** is a headless, always-on AI voice assistant running on a
Raspberry Pi 5. It listens for a wake word, records a voice command,
transcribes it locally, and routes it to a local OpenClaw LLM instance. When
the user's intent involves vision, the system captures an image with a Pi
Camera Module 3, identifies the subject via a cloud Vision API, and delegates
data retrieval (e.g. IMDB ratings) to OpenClaw's agent-browser skill. The
final answer is spoken back through local Text-to-Speech.

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Raspberry Pi 5  (headless)                                  │
│                                                              │
│  ┌────────────┐   wake    ┌────────────┐   audio             │
│  │ USB Mic    │──────────▶│ audio_     │──────────┐          │
│  │ (48 kHz)   │           │ pipeline   │          │          │
│  └────────────┘           │            │◀─ oww ── │          │
│                           │  record +  │          │          │
│                           │  whisper   │   text   │          │
│                           └─────┬──────┘          │          │
│                                 │                 │          │
│                                 ▼                 │          │
│                           ┌─────────────┐         │          │
│                           │ brain_      │         │          │
│                           │ bridge      │         │          │
│                           │             │         │          │
│            ┌──────────────│ intent ──▶  │         │          │
│            │  vision?     │ vision API  │         │          │
│            ▼              │ openclaw    │         │          │
│  ┌────────────┐           └─────┬───────┘         │          │
│  │ Pi Camera  │                 │                 │          │
│  │ Module 3   │                 │ response        │          │
│  │ (picamera2)│                 ▼                 │          │
│  └────────────┘           ┌─────────────┐         │          │
│                           │ voice_      │         │          │
│                           │ output      │         │          │
│                           │ (piper+aplay│────────▶│          │
│                           └─────────────┘  USB    │          │
│                                            Spkr   │          │
└──────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  LAN / localhost:18789  │
                    │  OpenClaw (OpenAI API)  │
                    │  agent-browser skill    │
                    └─────────────────────────┘
```

## 3. Hardware BOM

| Component                  | Notes                                    |
|----------------------------|------------------------------------------|
| Raspberry Pi 5 (4 GB+)    | Headless Raspberry Pi OS (Bookworm 64-bit) |
| USB Microphone             | 48 kHz capture, ALSA-compatible          |
| USB Speaker                | ALSA playback                            |
| Pi Camera Module 3         | libcamera / picamera2 (NOT legacy)       |
| MicroSD 32 GB+             | OS + models (~500 MB)                    |
| Power Supply 27 W USB-C   | Official Pi 5 PSU recommended            |

## 4. Software Dependencies

| Library / Tool      | Purpose                          | Install Method    |
|----------------------|----------------------------------|-------------------|
| openwakeword         | Offline wake-word detection      | pip               |
| sounddevice          | Audio I/O via PortAudio          | pip               |
| numpy                | Audio buffer manipulation        | pip               |
| whispercpp           | Local STT (ONNX, base.en)       | pip               |
| picamera2            | Pi Camera Module 3 capture       | apt + pip         |
| httpx                | Async HTTP client                | pip               |
| python-dotenv        | .env file loading                | pip               |
| piper-tts            | Local offline TTS                | pip / binary      |
| libcamera            | Camera backend for picamera2     | apt               |
| alsa-utils           | aplay / arecord                  | apt               |

### Models

| Model                          | Size   | Location                   |
|--------------------------------|--------|----------------------------|
| ggml-base.en.bin (Whisper)     | ~148 MB| `models/`                  |
| en_US-lessac-medium.onnx (Piper)| ~64 MB| `models/`                  |
| hey_jarvis (openwakeword)      | bundled| openwakeword package       |

## 5. Module Specifications

### 5.1 `config.py`

Central configuration loaded from `.env` + environment variables.

| Variable               | Default                                          |
|------------------------|--------------------------------------------------|
| WAKE_WORD_MODEL        | `hey_jarvis`                                     |
| WAKE_WORD_THRESHOLD    | `0.5`                                            |
| SAMPLE_RATE            | `48000`                                          |
| WHISPER_SAMPLE_RATE    | `16000`                                          |
| SILENCE_THRESHOLD      | `500` (RMS)                                      |
| SILENCE_DURATION       | `1.5` s                                          |
| MAX_RECORD_SECONDS     | `30`                                             |
| WHISPER_MODEL_PATH     | `models/ggml-base.en.bin`                        |
| VISION_PROVIDER        | `anthropic`                                      |
| ANTHROPIC_API_KEY      | (from .env)                                      |
| OPENAI_API_KEY         | (from .env)                                      |
| OPENCLAW_URL           | `http://127.0.0.1:18789/v1/chat/completions`     |
| OPENCLAW_TOKEN         | (from .env)                                      |
| OPENCLAW_MODEL         | `openai-codex/gpt-5.4`                           |
| PIPER_MODEL_PATH       | `models/en_US-lessac-medium.onnx`                |

### 5.2 `audio_pipeline.py`

| Function            | Description                                        |
|---------------------|----------------------------------------------------|
| `init()`            | Pre-load openwakeword model                        |
| `start_listening()` | Open mic stream, begin wake-word detection (bg)    |
| `wait_for_wake()`   | Block until wake word fires                        |
| `record_command()`  | Record until 1.5 s silence; return int16 array     |
| `transcribe(audio)` | Downsample 48→16 kHz, run Whisper, return text     |

### 5.3 `vision_capture.py`

| Function          | Description                                          |
|-------------------|------------------------------------------------------|
| `capture_image()` | Init Picamera2, autofocus, capture JPEG, release cam |

Camera constraints:
- Must use `picamera2` + `libcamera` (NOT legacy `cv2` or `raspistill`)
- Must enable `AfModeEnum.Continuous` before capture
- Must release camera in `finally` block

### 5.4 `brain_bridge.py`

| Function                    | Description                                  |
|-----------------------------|----------------------------------------------|
| `_detect_vision_intent(t)`  | Keyword match → bool                         |
| `_call_vision_api(path)`    | Base64 image → Anthropic/OpenAI → title      |
| `_call_openclaw(prompt)`    | POST to OpenClaw with Bearer auth            |
| `process(transcription)`    | Orchestrate intent → vision → OpenClaw → text|

OpenClaw integration:
- Endpoint: `http://127.0.0.1:18789/v1/chat/completions`
- Auth: `Authorization: Bearer {OPENCLAW_TOKEN}`
- Model: `openai-codex/gpt-5.4`
- No web-scraping code — delegates to OpenClaw's agent-browser skill

### 5.5 `voice_output.py`

| Function      | Description                                          |
|---------------|------------------------------------------------------|
| `speak(text)` | Sanitize → Piper TTS → aplay pipeline (blocks)      |

### 5.6 `main.py`

Orchestrator loop:
1. Load `.env`, init models
2. Start wake-word listener
3. Wait for wake → record → transcribe → process → speak
4. Repeat; handle SIGINT/SIGTERM gracefully

## 6. Data Flow

```
[USB Mic 48kHz] → sounddevice → openwakeword (wake detection)
                                      │
                                wake event
                                      │
                               record audio
                                      │
                          downsample 48→16 kHz
                                      │
                            Whisper STT (local)
                                      │
                              transcribed text
                                      │
                        ┌─────────────┴──────────────┐
                   vision intent?                 no vision
                        │                             │
                  capture_image()                      │
                        │                             │
                   Vision API                         │
                  (Anthropic/OpenAI)                   │
                        │                             │
                   extract title                      │
                        │                             │
                  construct prompt                    │
                  (agent-browser)                     │
                        │                             │
                        └─────────────┬──────────────┘
                                      │
                         OpenClaw POST (Bearer auth)
                         model: openai-codex/gpt-5.4
                                      │
                              response text
                                      │
                          Piper TTS → aplay
                                      │
                              [USB Speaker]
```

## 7. Known Limitations (v1)

- **Wake word**: Using pre-trained `hey_jarvis`; custom `hey_wiz` requires
  openwakeword training pipeline (deferred).
- **Single-turn only**: No conversation memory between queries.
- **No streaming TTS**: Full response is synthesised before playback begins.
- **Downsampling**: Simple numpy decimation (factor 3). May introduce aliasing
  in edge cases; swap to `scipy.signal.resample` if quality degrades.
- **Camera latency**: ~1 s autofocus delay per capture.
- **Vision scope**: Cloud API call required (not offline).

## 8. Future Roadmap

- [ ] Train custom `hey_wiz` wake word via openwakeword training pipeline
- [ ] Streaming TTS for lower perceived latency
- [ ] Multi-turn conversation context
- [ ] On-device vision model (e.g. MobileNet) for offline fallback
- [ ] LED/GPIO feedback for listening/processing/speaking states
- [ ] Web dashboard for configuration and logs
