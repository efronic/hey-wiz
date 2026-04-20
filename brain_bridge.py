"""Brain bridge: intent routing, Vision API calls, and OpenClaw handoff."""

import base64
import logging

import httpx

import config

log = logging.getLogger(__name__)

VISION_API_TIMEOUT = 30.0
OPENCLAW_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

def _detect_vision_intent(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in config.VISION_INTENT_KEYWORDS)


# ---------------------------------------------------------------------------
# Vision API
# ---------------------------------------------------------------------------

async def _call_vision_api(image_path: str) -> str:
    """Send image to the configured Vision API and return the description."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    if config.VISION_PROVIDER == "anthropic":
        return await _call_anthropic(image_b64)
    elif config.VISION_PROVIDER == "openai":
        return await _call_openai(image_b64)
    else:
        raise ValueError(f"Unknown VISION_PROVIDER: {config.VISION_PROVIDER}")


async def _call_anthropic(image_b64: str) -> str:
    payload = {
        "model": config.VISION_MODEL_ANTHROPIC,
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": config.VISION_PROMPT},
                ],
            }
        ],
    }
    async with httpx.AsyncClient(timeout=VISION_API_TIMEOUT) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": config.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    return data["content"][0]["text"].strip()


async def _call_openai(image_b64: str) -> str:
    payload = {
        "model": config.VISION_MODEL_OPENAI,
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                    {"type": "text", "text": config.VISION_PROMPT},
                ],
            }
        ],
    }
    async with httpx.AsyncClient(timeout=VISION_API_TIMEOUT) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# OpenClaw handoff
# ---------------------------------------------------------------------------

async def _call_openclaw(prompt: str) -> str:
    """POST to the local OpenClaw instance with Bearer auth."""
    payload = {
        "model": config.OPENCLAW_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {config.OPENCLAW_TOKEN}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=OPENCLAW_TIMEOUT) as client:
        resp = await client.post(config.OPENCLAW_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def process(transcription: str) -> str:
    """Route a transcription through vision (if needed) and OpenClaw.

    Returns the final text response to be spoken.
    """
    log.info("Processing: %s", transcription)

    if _detect_vision_intent(transcription):
        log.info("Vision intent detected — capturing image.")
        from vision_capture import capture_image

        image_path = capture_image()
        title = await _call_vision_api(image_path)
        log.info("Vision API returned: %s", title)

        prompt = (
            f"Using the agent-browser skill, go to imdb.com, search for "
            f"{title}, read the accessibility tree, and return only the "
            f"IMDB rating and Metascore as a natural language string."
        )
    else:
        prompt = transcription

    response = await _call_openclaw(prompt)
    log.info("OpenClaw response: %s", response)
    return response
