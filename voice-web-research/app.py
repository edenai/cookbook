"""Ask the web, out loud: speech -> LLM + web search -> speech, all through Eden AI."""
import asyncio
import json
import time

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import edenai

STT = "audio/speech_to_text_async/gradium"
CHOICES = {  # first entry = default; the presenter can swap them live in the UI
    "search": ["web/search/tavily", "web/search/linkup", "web/search/firecrawl"],
    "tts": ["audio/tts/gradium", "audio/tts/elevenlabs", "audio/tts/openai/tts-1"],
    "llm": ["nebius/Qwen/Qwen3-235B-A22B-Instruct-2507", "nebius/openai/gpt-oss-120b", "mistral/mistral-large-latest"],
}
SYSTEM = ("You are a research assistant. Use web_search for anything time-sensitive or factual. "
          "Answer in at most 120 words of plain prose that sounds natural read aloud, "
          "then a line 'Sources:' with numbered URLs.")
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


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/providers")
def providers():
    return CHOICES


@app.post("/api/ask")
async def ask(audio: UploadFile | None = None, text: str = Form(""),
              search: str = Form(CHOICES["search"][0]), tts: str = Form(CHOICES["tts"][0]),
              llm: str = Form(CHOICES["llm"][0])):
    steps, current = [], {"step": "start"}

    async def run(name, model, call):  # time one Eden AI call and record who served it and what it cost
        current["step"] = name
        start = time.perf_counter()
        body = await call
        steps.append({"name": name, "model": model, "provider": body.get("provider"),
                      "ms": round((time.perf_counter() - start) * 1000), "cost": float(body.get("cost") or 0)})
        return body

    async def pipeline():
        # 1. Speech to text (or use the typed question)
        transcript = text.strip()
        if audio:
            file = await run("upload", "/v3/upload", edenai.upload(await audio.read(), audio.filename or "question.webm"))
            transcript = (await run("speech-to-text", STT, edenai.transcribe(file["file_id"], STT)))["output"]["text"].strip()
        if not transcript:
            raise RuntimeError("no question heard: try again, or type it")

        # 2. LLM with one tool: web_search. Turn 1 must search (grounded answers); the last turn must answer.
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": transcript}]
        sources = {}
        for turn in range(4):
            choice = "required" if turn == 0 else "none" if turn == 3 else "auto"
            reply = await run("llm", llm, edenai.chat(messages, TOOLS, llm, [m for m in CHOICES["llm"] if m != llm], choice))
            msg = reply["choices"][0]["message"]
            if not msg.get("tool_calls"):
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": [
                {"id": c["id"], "type": "function",  # echo back only the spec fields
                 "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}
                for c in msg["tool_calls"]]})
            for c in msg["tool_calls"]:
                query = json.loads(c["function"]["arguments"] or "{}").get("query") or transcript
                found = await run("web search", search, edenai.search(query, search, [m for m in CHOICES["search"] if m != search]))
                results = found["output"].get("results") or []
                sources.update({r["url"]: r.get("title") or r["url"] for r in results})
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(results)[:6000]})
        answer = (msg.get("content") or "").strip()

        # 3. Read the answer aloud (without the Sources: block). If this fails, still show the answer.
        spoken = answer.split("Sources:")[0].strip() or answer
        audio_url, warning = None, None
        try:
            audio_url = (await run("text-to-speech", tts, edenai.speak(spoken, tts)))["output"]["audio_resource_url"]
        except Exception as e:
            warning = f"text-to-speech failed: {e}"

        return {"transcript": transcript, "answer": answer, "audio_url": audio_url, "warning": warning,
                "sources": [{"title": t, "url": u} for u, t in sources.items()],
                "steps": steps, "total_cost": round(sum(s["cost"] for s in steps), 6)}

    try:
        return await asyncio.wait_for(pipeline(), timeout=60)
    except Exception as e:
        message = "took longer than 60s" if isinstance(e, TimeoutError) else str(e)
        return JSONResponse({"error": message, "step": current["step"], "steps": steps}, status_code=502)
