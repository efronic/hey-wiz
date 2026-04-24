"""Brain bridge: tag-based command routing, Vision API, and OpenClaw handoff."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import uuid

import httpx
import websockets

import config

log = logging.getLogger(__name__)

VISION_API_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Tag matching
# ---------------------------------------------------------------------------


def _match_command_tag(text: str) -> tuple[dict, str] | tuple[None, str]:
    """Check whether *text* starts with a known command-tag trigger.

    Returns ``(tag_config, remaining_text)`` on match, or
    ``(None, original_text)`` when no tag is recognised.
    """
    lower = text.lower().strip()
    for tag_name, tag_cfg in config.COMMAND_TAGS.items():
        for trigger in tag_cfg["triggers"]:
            pattern = re.compile(rf"^{re.escape(trigger)}\b[,:\s]*", re.IGNORECASE)
            m = pattern.match(lower)
            if m:
                remainder = text[m.end() :].strip()
                log.info("Matched command tag '%s' (trigger='%s')", tag_name, trigger)
                return tag_cfg, remainder
    return None, text


# ---------------------------------------------------------------------------
# Vision API
# ---------------------------------------------------------------------------


async def _call_vision_api(image_path: str, vision_prompt: str) -> str:
    """Send image to the configured Vision API and return the description."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    if config.VISION_PROVIDER == "anthropic":
        return await _call_anthropic(image_b64, vision_prompt)
    elif config.VISION_PROVIDER == "openai":
        return await _call_openai(image_b64, vision_prompt)
    else:
        raise ValueError(f"Unknown VISION_PROVIDER: {config.VISION_PROVIDER}")


async def _call_anthropic(image_b64: str, vision_prompt: str) -> str:
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
                    {"type": "text", "text": vision_prompt},
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


async def _call_openai(image_b64: str, vision_prompt: str) -> str:
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
                    {"type": "text", "text": vision_prompt},
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
# OpenClaw gateway WebSocket client
# ---------------------------------------------------------------------------


async def _gw_connect() -> websockets.WebSocketClientProtocol:
    """Open an authenticated WebSocket to the OpenClaw gateway."""
    ws = await websockets.connect(
        config.OPENCLAW_GW_URL,
        open_timeout=5,
        additional_headers={"Origin": config.OPENCLAW_GW_ORIGIN},
    )
    # Consume challenge
    challenge = json.loads(await ws.recv())
    if challenge.get("event") != "connect.challenge":
        raise RuntimeError(f"Unexpected gateway greeting: {challenge}")

    req_id = str(uuid.uuid4())
    await ws.send(
        json.dumps(
            {
                "type": "req",
                "id": req_id,
                "method": "connect",
                "params": {
                    "minProtocol": 3,
                    "maxProtocol": 3,
                    "client": {
                        "id": "openclaw-control-ui",
                        "version": "1.0.0",
                        "platform": "linux-aarch64",
                        "mode": "webchat",
                    },
                    "role": "operator",
                    "scopes": ["operator.read", "operator.write", "operator.admin"],
                    "auth": {"token": config.OPENCLAW_TOKEN},
                    "caps": [],
                },
            }
        )
    )
    resp = json.loads(await ws.recv())
    if not resp.get("ok"):
        err = resp.get("error", {}).get("message", "unknown")
        raise RuntimeError(f"Gateway auth failed: {err}")
    return ws


async def _call_openclaw(prompt: str) -> str:
    """Send a message via the OpenClaw gateway WebSocket and return the reply."""
    log.info("Calling OpenClaw gateway (session=%s)…", config.OPENCLAW_SESSION_ID)
    try:
        ws = await _gw_connect()
    except Exception as exc:
        log.error("Gateway connect failed: %s", exc)
        return "Sorry, I couldn't connect to OpenClaw."

    try:
        chat_id = str(uuid.uuid4())
        await ws.send(
            json.dumps(
                {
                    "type": "req",
                    "id": chat_id,
                    "method": "chat.send",
                    "params": {
                        "sessionKey": f"agent:main:explicit:{config.OPENCLAW_SESSION_ID}",
                        "message": prompt,
                        "deliver": False,
                        "idempotencyKey": str(uuid.uuid4()),
                    },
                }
            )
        )

        # Listen for the assistant's final message
        deadline = asyncio.get_event_loop().time() + config.OPENCLAW_TIMEOUT
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                log.error("OpenClaw timed out after %ds", config.OPENCLAW_TIMEOUT)
                return "Sorry, I timed out waiting for a response."

            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            msg = json.loads(raw)

            if msg.get("type") != "event":
                continue

            payload = msg.get("payload", {})
            state = payload.get("state", "")
            message = payload.get("message", {})

            if state == "final" and message.get("role") == "assistant":
                content = message.get("content", "")
                # content is either a string or a list of blocks
                if isinstance(content, list):
                    parts = [b["text"] for b in content if b.get("type") == "text"]
                    text = " ".join(parts).strip()
                else:
                    text = str(content).strip()
                log.info("OpenClaw responded (%d chars)", len(text))
                return text or "No response from OpenClaw."

    except asyncio.TimeoutError:
        log.error("OpenClaw timed out after %ds", config.OPENCLAW_TIMEOUT)
        return "Sorry, I timed out waiting for a response."
    except Exception as exc:
        log.error("OpenClaw error: %s", exc)
        return "Sorry, something went wrong talking to OpenClaw."
    finally:
        await ws.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def process(transcription: str) -> str:
    """Route a transcription through the tag system and OpenClaw.

    1. Match a command tag  →  run its pipeline (vision + templated prompt).
    2. No tag matched       →  forward verbatim to OpenClaw (general assistant).

    Returns the final text response to be spoken.
    """
    log.info("Processing: %s", transcription)

    tag_cfg, remainder = _match_command_tag(transcription)

    if tag_cfg is not None:
        result = ""

        if tag_cfg["needs_vision"]:
            log.info("Tag requires vision — capturing image.")
            from vision_capture import capture_image

            image_path = capture_image()
            result = await _call_vision_api(image_path, tag_cfg["vision_prompt"])
            log.info("Vision API returned: %s", result)

        prompt = tag_cfg["openclaw_template"].format(
            result=result,
            remainder=remainder,
        )
    else:
        prompt = transcription

    response = await _call_openclaw(prompt)
    log.info("OpenClaw response: %s", response)
    return response
