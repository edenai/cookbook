"""The five Eden AI calls this demo makes. One key, one base URL."""
import asyncio
import os

import httpx
from dotenv import load_dotenv

load_dotenv()
BASE = os.getenv("EDENAI_BASE_URL", "https://api.edenai.run/v3")
KEY = os.getenv("EDENAI_API_KEY", "")
client = httpx.AsyncClient(base_url=BASE, timeout=30, headers={"Authorization": f"Bearer {KEY or 'missing-key'}"})


async def call(method, path, **kwargs):
    r = await client.request(method, path, **kwargs)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    body = r.json()
    if body.get("status") == "fail":  # provider failures come back as HTTP 200
        error = body.get("error") or {}
        raise RuntimeError(f"{body.get('provider')} failed: {error.get('message', error) if isinstance(error, dict) else error}")
    return body


async def upload(data: bytes, filename: str, content_type: str) -> dict:
    return await call("POST", "/upload", files={"file": (filename, data, content_type)}, data={"expires_in_days": "1"})


async def transcribe(file_id: str, model: str) -> dict:
    job = await call("POST", "/universal-ai/async", json={"model": model, "input": {"file": file_id, "language": "en"}})
    for _ in range(30):
        body = await call("GET", f"/universal-ai/async/{job['public_id']}")
        if body["status"] == "success":
            return body
        await asyncio.sleep(1)
    raise RuntimeError("transcription took longer than 30s")


async def search(query: str, model: str, fallbacks: list[str]) -> dict:
    return await call("POST", "/universal-ai", json={
        "model": model, "fallbacks": fallbacks, "input": {"query": query, "max_results": 6}})


async def chat(messages: list, tools: list, model: str, fallbacks: list[str], tool_choice: str) -> dict:
    return await call("POST", "/chat/completions", json={
        "model": model, "fallbacks": fallbacks, "messages": messages, "tools": tools, "tool_choice": tool_choice})


async def speak(text: str, model: str) -> dict:
    audio_format = "wav" if model.startswith("audio/tts/gradium") else "mp3"  # Gradium has no mp3
    return await call("POST", "/universal-ai", json={"model": model, "input": {"text": text, "audio_format": audio_format}})


async def download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30) as plain:  # no Authorization header: this is a CDN link
        return (await plain.get(url)).content


def in_catalog(model: str) -> bool:
    """Is this model string live? Uses the public catalog endpoints (no key needed)."""
    if model.startswith(("web/", "audio/")):  # expert model: feature/subfeature/provider[/model]
        feature = "/".join(model.split("/")[:2])
        return model in {m["model"] for m in httpx.get(f"{BASE}/info/{feature}", timeout=30).json()["models"]}
    return model in {m["id"] for m in httpx.get(f"{BASE}/models", timeout=30).json()["data"]}
