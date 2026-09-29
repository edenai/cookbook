"""Web view for Director's Cut: describe a film, watch Claude write the shot list and PixVerse shoot it, then reshoot."""
import asyncio
import json
import shutil
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import edenai
from director import DIRECTORS, MODELS, MULTI_SHOT, SIZES, make_film, reshoot

OUT = Path("out")
OUT.mkdir(exist_ok=True)
RUNS = set()  # keep a reference to running pipelines

print(f"\nEden AI base URL: {edenai.BASE}" + ("" if edenai.KEY else "   ! EDENAI_API_KEY is not set: copy .env.example to .env")
      + ("" if shutil.which("ffmpeg") else "   ! ffmpeg is not installed") + "\n")

app = FastAPI()
app.mount("/out", StaticFiles(directory=OUT), name="out")  # the shots and the cut


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/options")
def options():
    return {"models": MODELS, "multishot": sorted(MULTI_SHOT), "aspects": list(SIZES),
            "directors": [{"id": k, "label": v[0], "group": v[1]} for k, v in DIRECTORS.items()]}


def stream(job) -> StreamingResponse:
    """Run a pipeline in the background and stream its events, one JSON object per line."""
    events: asyncio.Queue = asyncio.Queue()

    async def run():
        try:
            await job(events.put_nowait)
        except Exception as e:  # shown in the page instead of a broken stream
            events.put_nowait({"event": "error", "message": str(e)})
        events.put_nowait(None)

    task = asyncio.create_task(run())  # finishes even if the page is closed: the shots are already paid for
    RUNS.add(task)
    task.add_done_callback(RUNS.discard)

    async def lines():
        while (event := await events.get()) is not None:
            yield json.dumps(event) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.post("/api/film")
async def film(premise: str = Form(...), model: str = Form(...), aspect: str = Form("16:9"), shots: int = Form(4),
               mode: str = Form("shots"), director: str = Form("anthropic/claude-sonnet-latest")):
    if (not premise.strip() or model not in MODELS or aspect not in SIZES or director not in DIRECTORS
            or mode not in ("shots", "chain", "multishot") or not 2 <= shots <= 6 or (mode == "multishot" and model not in MULTI_SHOT)):
        raise HTTPException(400, "invalid options")
    return stream(lambda emit: make_film(premise.strip(), OUT, model, aspect, shots, mode, director, emit))


@app.post("/api/reshoot")
async def reshoot_shot(folder: str = Form(...), index: int = Form(...), action: str = Form("")):
    if "/" in folder or ".." in folder or not (OUT / folder / "plan.json").exists():
        raise HTTPException(404, "unknown film")
    return stream(lambda emit: reshoot(OUT, folder, index, action, emit))
