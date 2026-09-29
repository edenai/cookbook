"""The Eden AI calls behind Director's Cut: Claude (chat completions) and PixVerse (video generation). One key."""
import asyncio
import os
import time

import httpx

try:  # a .env file for local runs
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE = os.getenv("EDENAI_BASE_URL", "https://api.edenai.run/v3")
KEY = os.getenv("EDENAI_API_KEY", "")
TAGS = {"demo": "directors-cut"}  # this demo's cost, split out in the Eden AI dashboard
client = httpx.AsyncClient(base_url=BASE, timeout=120, headers={"Authorization": f"Bearer {KEY or 'missing-key'}"})


async def call(method: str, path: str, **kwargs) -> dict:
    r = await client.request(method, path, **kwargs)
    if r.status_code >= 400:
        raise RuntimeError(f"Eden AI {path}: HTTP {r.status_code}: {r.text[:300]}")
    body = r.json()
    if body.get("status") == "fail":  # provider failures come back as HTTP 200
        error = body.get("error") or {}
        raise RuntimeError(f"{body.get('provider')} failed: {error.get('message', error) if isinstance(error, dict) else error}")
    return body


async def chat(messages: list, model: str, fallbacks: list, schema: dict) -> dict:
    return await call("POST", "/chat/completions", json={
        "model": model, "fallbacks": fallbacks, "messages": messages, "temperature": 0.7, "max_tokens": 4000, "tags": TAGS,
        "response_format": {"type": "json_schema", "json_schema": {"name": "film", "strict": True, "schema": schema}}})


async def upload(data: bytes, filename: str, content_type: str) -> dict:
    return await call("POST", "/upload", files={"file": (filename, data, content_type)}, data={"expires_in_days": "1"})


async def video(model: str, prompt: str, seconds: int, provider_params: dict, image: str | None = None, timeout: int = 420) -> dict:
    """Start a PixVerse job (text-to-video, or image-to-video when `image` is a file id) and poll until it's done."""
    inputs = {"text": prompt, "duration": seconds, **({"file": image} if image else {})}
    job = await call("POST", "/universal-ai/async", json={"model": model, "input": inputs, "provider_params": provider_params, "tags": TAGS})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = await call("GET", f"/universal-ai/async/{job['public_id']}")
        if body["status"] == "success":
            return body
        await asyncio.sleep(2)
    raise RuntimeError(f"video job {job['public_id']} still running after {timeout}s")


async def download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120) as plain:  # no Authorization header: these are CDN links
        r = await plain.get(url)
        r.raise_for_status()
        return r.content
