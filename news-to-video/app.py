"""Web view for news-to-video: type a topic, watch the 3 Pruna clips arrive, then play the narrated result."""
import asyncio
import json
import shutil
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import edenai
from news_to_video import VIDEO, make_video

OUT = Path("out")
OUT.mkdir(exist_ok=True)
MODELS = [VIDEO, "video/generation_async/pruna/p-video-2"]  # p-video-2: higher quality, much slower
RUNS = set()  # keep a reference to running pipelines

print(f"\nEden AI base URL: {edenai.BASE}" + ("" if edenai.KEY else "   ! EDENAI_API_KEY is not set: copy .env.example to .env")
      + ("" if shutil.which("ffmpeg") else "   ! ffmpeg is not installed") + "\n")

app = FastAPI()
app.mount("/out", StaticFiles(directory=OUT), name="out")  # the clips and the final video


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/models")
def models():
    return MODELS


@app.post("/api/run")
async def run(topic: str = Form(...), video_model: str = Form(VIDEO)):
    """Make a video, streaming one JSON event per line as each step finishes (see make_video)."""
    if video_model not in MODELS or not topic.strip():
        raise HTTPException(400, "unknown video model or empty topic")
    events: asyncio.Queue = asyncio.Queue()

    async def pipeline():
        try:
            await make_video(topic.strip(), OUT, video_model=video_model, emit=events.put_nowait)
        except Exception as e:  # shown in the page instead of a broken stream
            events.put_nowait({"event": "error", "message": str(e)})
        events.put_nowait(None)

    task = asyncio.create_task(pipeline())  # runs to the end even if the page is closed: the clips are already paid for
    RUNS.add(task)
    task.add_done_callback(RUNS.discard)

    async def stream():
        while (event := await events.get()) is not None:
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
