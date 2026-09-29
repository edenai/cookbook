"""The Eden AI calls behind news-to-video: web search, an LLM, video generation and text-to-speech. One key."""
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
TAGS = {"demo": "news-to-video"}  # cost of this demo, split out in the Eden AI dashboard
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


async def search(query: str) -> dict:
    return await call("POST", "/universal-ai", json={"model": "web/search/tavily", "fallbacks": ["web/search/linkup"],
                                                     "input": {"query": query, "max_results": 6}, "tags": TAGS})


async def chat(messages: list, model: str, fallbacks: list, schema: dict) -> dict:
    return await call("POST", "/chat/completions", json={
        "model": model, "fallbacks": fallbacks, "messages": messages, "temperature": 0.4, "tags": TAGS,
        "response_format": {"type": "json_schema", "json_schema": {"name": "script", "strict": True, "schema": schema}}})


async def video(prompt: str, model: str, dimension: str, seconds: int, timeout: int = 300) -> dict:
    """Start a video job and poll it until it's done (video generation is async on Eden AI)."""
    job = await call("POST", "/universal-ai/async", json={
        "model": model, "input": {"text": prompt, "duration": seconds, "dimension": dimension}, "tags": TAGS})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = await call("GET", f"/universal-ai/async/{job['public_id']}")
        if body["status"] == "success":
            return body
        await asyncio.sleep(2)
    raise RuntimeError(f"video job {job['public_id']} still running after {timeout}s")


async def speak(text: str, model: str) -> dict:
    audio_format = "wav" if model.startswith("audio/tts/gradium") else "mp3"  # Gradium has no mp3
    return await call("POST", "/universal-ai", json={"model": model, "input": {"text": text, "audio_format": audio_format}, "tags": TAGS})


async def download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120) as plain:  # no Authorization header: these are CDN links
        r = await plain.get(url)
        r.raise_for_status()
        return r.content
