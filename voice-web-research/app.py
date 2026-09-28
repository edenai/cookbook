"""Ask the web, out loud: speech -> LLM + web search -> speech, all through Eden AI."""
import asyncio
import base64
import datetime
import json
import re
import struct
import time

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import edenai

STT = "audio/speech_to_text_async/gradium"
CHOICES = {  # first entry = default; the presenter can swap them live in the UI
    "search": ["web/search/tavily", "web/search/linkup", "web/search/firecrawl"],
    "tts": ["audio/tts/gradium", "audio/tts/elevenlabs", "audio/tts/deepgram/aura-2"],
    "llm": ["nebius/Qwen/Qwen3-235B-A22B-Instruct-2507", "nebius/openai/gpt-oss-120b", "mistral/mistral-large-latest"],
}
MAX_SEARCHES = 2  # per question: models happily fire ten searches at once, which is slow and costly on stage
SYSTEM = ("You are a research assistant. Today is {today}. Use web_search (at most twice) for anything time-sensitive "
          "or factual, and put the month and year in queries about recent events. Answer in at most 80 words of plain "
          "prose that sounds natural read aloud, then a line 'Sources:' with the numbered URLs you used.")
TOOLS = [{"type": "function", "function": {
    "name": "web_search", "description": "Search the web. Returns titles, URLs and page content.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]

print(f"\nEden AI base URL: {edenai.BASE}" + ("" if edenai.KEY else "   ! EDENAI_API_KEY is not set: copy .env.example to .env"))
try:  # startup check: keep only model strings the live catalog knows
    for kind, models in CHOICES.items():
        CHOICES[kind] = [m for m in models if edenai.in_catalog(m)] or models
        print(f"  {kind:<6} {' | '.join(CHOICES[kind])}")
    print(f"  stt    {STT}{'' if edenai.in_catalog(STT) else '  (NOT in catalog!)'}\n")
except Exception as e:
    print(f"  catalog check skipped ({e})\n")

app = FastAPI()
app.mount("/samples", StaticFiles(directory="samples"), name="samples")


class StepError(Exception):
    def __init__(self, step, error):
        super().__init__(str(error))
        self.step = step


async def timed(steps, name, model, call):
    """Run one Eden AI call and record who served it, how long it took and what it cost."""
    start = time.perf_counter()
    try:
        body = await call
    except Exception as e:
        raise StepError(name, e) from e
    steps.append({"name": name, "model": model, "provider": body.get("provider"),
                  "ms": round((time.perf_counter() - start) * 1000), "cost": float(body.get("cost") or 0)})
    return body


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/providers")
def providers():
    return CHOICES


@app.post("/api/ask")
async def ask(audio: UploadFile | None = None, text: str = Form(""),
              search: str = Form(CHOICES["search"][0]), llm: str = Form(CHOICES["llm"][0])):
    steps = []

    async def pipeline():
        # 1. Speech to text (or use the typed question)
        transcript = text.strip()
        if audio:
            content_type = (audio.content_type or "audio/wav").split(";")[0]  # say "audio/webm", or Eden AI sees video/webm
            file = await timed(steps, "upload", "/v3/upload", edenai.upload(await audio.read(), audio.filename or "question.webm", content_type))
            transcript = (await timed(steps, "speech-to-text", STT, edenai.transcribe(file["file_id"], STT)))["output"]["text"].strip()
        if not transcript:
            raise StepError("input", "no question heard: try again, or type it")

        # 2. LLM with one tool, web_search. Turn 1 must search; after MAX_SEARCHES (or turn 4) it must answer.
        messages = [{"role": "system", "content": SYSTEM.format(today=datetime.date.today().strftime("%B %d, %Y"))},
                    {"role": "user", "content": transcript}]
        sources, searches = {}, 0
        for turn in range(4):
            choice = "required" if turn == 0 else "none" if turn == 3 or searches >= MAX_SEARCHES else "auto"
            reply = await timed(steps, "llm", llm, edenai.chat(messages, TOOLS, llm, [m for m in CHOICES["llm"] if m != llm], choice))
            msg = reply["choices"][0]["message"]
            if not msg.get("tool_calls"):
                break
            calls = msg["tool_calls"]
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": [
                {"id": c["id"], "type": "function",  # echo back only the spec fields
                 "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}} for c in calls]})
            allowed = calls[:max(0, MAX_SEARCHES - searches)]
            searches += len(allowed)
            queries = [json.loads(c["function"]["arguments"] or "{}").get("query") or transcript for c in allowed]
            found = await asyncio.gather(*(timed(steps, "web search", search, edenai.search(q, search, [m for m in CHOICES["search"] if m != search]))
                                           for q in queries))  # the searches of one turn run in parallel
            for i, c in enumerate(calls):  # every tool call needs an answer, even the ones over the limit
                results = (found[i]["output"].get("results") or []) if i < len(found) else None
                sources.update({r["url"]: r.get("title") or r["url"] for r in results or []})
                content = json.dumps(results)[:6000] if results is not None else "Search limit reached: answer with what you have."
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": content})
        answer = (msg.get("content") or "").strip()
        cited = [(u, t) for u, t in sources.items() if u in answer] or list(sources.items())[:5]

        return {"transcript": transcript, "answer": answer, "spoken": re.sub(r"[*_#`]", "", answer.split("Sources:")[0]).strip() or answer,  # no markdown aloud
                "sources": [{"title": t, "url": u} for u, t in cited], "steps": steps}

    return await respond(pipeline(), steps)


@app.post("/api/speak")
async def speak(text: str = Form(...), tts: str = Form(CHOICES["tts"][0])):
    """Step 3, separate so the answer is on screen while the audio is being made."""
    steps = []

    async def pipeline():
        body = await timed(steps, "text-to-speech", tts, edenai.speak(text, tts))
        return {"audio_url": as_data_url(await edenai.download(body["output"]["audio_resource_url"])), "steps": steps}

    return await respond(pipeline(), steps)


def as_data_url(audio: bytes) -> str:
    """Hand the browser the audio itself: the right MIME type, no expiring CDN link, and a WAV header players accept
    (Gradium streams WAVs whose lengths are unknown, 0xFFFFFFFF, so write the real ones)."""
    if audio[:4] == b"RIFF" and audio[4:8] == b"\xff\xff\xff\xff":
        i = audio.find(b"data", 12)
        audio = audio[:4] + struct.pack("<I", len(audio) - 8) + audio[8:i + 4] + struct.pack("<I", len(audio) - i - 8) + audio[i + 8:]
    return f"data:{'audio/wav' if audio[:4] == b'RIFF' else 'audio/mpeg'};base64,{base64.b64encode(audio).decode()}"


async def respond(pipeline, steps):
    try:
        return await asyncio.wait_for(pipeline, timeout=60)
    except StepError as e:
        return JSONResponse({"error": str(e), "step": e.step, "steps": steps}, status_code=502)
    except TimeoutError:
        return JSONResponse({"error": "took longer than 60s", "step": "timeout", "steps": steps}, status_code=502)
