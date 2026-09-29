"""Director's Cut: Claude directs, PixVerse shoots, ffmpeg edits. Every AI call goes through Eden AI with one key.

premise -> Claude writes the film: a style bible plus N shots (action, camera move, sound, length)
        -> PixVerse renders each shot with native audio (in parallel, or chained frame to frame for continuity,
           or as one multi-shot take that PixVerse cuts itself)
        -> ffmpeg cuts the shots together behind a title card
"""
import asyncio
import datetime
import json
import re
import subprocess
import time
from pathlib import Path

import edenai

MODELS = {"video/generation_async/pixverse/v6": "PixVerse V6", "video/generation_async/pixverse/c1": "PixVerse C1",
          "video/generation_async/pixverse/v5.5": "PixVerse V5.5"}
MULTI_SHOT = {"video/generation_async/pixverse/v6", "video/generation_async/pixverse/v5.5"}  # native multi-shot takes
DIRECTORS = {"anthropic/claude-sonnet-latest": "Claude Sonnet", "anthropic/claude-opus-latest": "Claude Opus"}
SIZES = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (720, 720), "21:9": (1680, 720)}  # the edit's frame size

FILM = {"type": "object", "additionalProperties": False, "required": ["title", "logline", "style", "shots"],
        "properties": {
            "title": {"type": "string"}, "logline": {"type": "string"}, "style": {"type": "string"},
            "shots": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                 "required": ["name", "action", "sound", "seconds"],
                                                 "properties": {"name": {"type": "string"}, "action": {"type": "string"},
                                                                "sound": {"type": "string"}, "seconds": {"type": "integer"}}}}}}
DIRECTOR = """You are a film director and editor. Turn the premise into a short film of exactly {n} shots, to be rendered by
an AI video model (PixVerse) in {aspect} and cut together in order.
- "title": at most 6 words. "logline": one sentence.
- "style": one paragraph that locks the look of the whole film: describe each character identically every time (age,
  build, hair, clothing), the setting, the colour palette, the lens and the film stock. It is prepended to every shot so
  the shots match, so put everything that must stay consistent here.
- "shots": cut like an editor: establish, build, pay off. Vary the shot sizes, and match action across the cuts.
  "name" is 2-4 words. "action" is one clear moment in 1-2 sentences, with the shot size (wide, medium, close-up),
  one camera movement (dolly in, tracking, pan, crane up, handheld, orbit...) and the lighting.
  "sound" is what we hear: ambience, sound effects and music mood, with at most one short line of dialogue.
  "seconds" is 3 to 5.
- No on-screen text, logos or real, named people."""


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "film"


def shot_prompt(style: str, shot: dict) -> str:
    return f"{style}\n\n{shot['action']}\nSound: {shot['sound']}"


def seconds(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def has_audio(path: Path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True).stdout
    return bool(out.strip())


def last_frame(clip: Path, out: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.2", "-i", str(clip), "-frames:v", "1", "-update", "1", str(out)],
                   check=True)


def edit(work: Path, clips: list[str], title: str, aspect: str, out: str = "film.mp4") -> None:
    """Cut the clips together at one frame size, with a title card over the first seconds and an AI-generated label."""
    w, h = SIZES[aspect]
    (work / "title.txt").write_text(title)
    (work / "credit.txt").write_text("directed by Claude  ·  shot on PixVerse  ·  via Eden AI")
    fade = "alpha='if(lt(t,2.2),1,max(0,1-(t-2.2)*2))'"
    text = "expansion=none:fontcolor=white:borderw=3:bordercolor=black@0.6:x=(w-text_w)/2"
    parts, joined = [], ""
    for i, clip in enumerate(clips):
        video = f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24[v{i}]"
        audio = (f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo[a{i}]" if has_audio(work / clip)
                 else f"anullsrc=r=48000:cl=stereo,atrim=duration={seconds(work / clip):.3f}[a{i}]")  # a silent shot
        parts += [video, audio]
        joined += f"[v{i}][a{i}]"
    parts.append(f"{joined}concat=n={len(clips)}:v=1:a=1[cut][a]")
    parts.append(f"[cut]drawtext=textfile=title.txt:fontsize={h // 11}:{text}:y=(h-text_h)/2-{h // 20}:{fade},"
                 f"drawtext=textfile=credit.txt:fontsize={h // 34}:{text}:y=(h/2)+{h // 14}:{fade},"
                 f"drawtext=text='AI-generated':fontsize={h // 40}:{text}:y={h // 30}[v]")
    cmd = (["ffmpeg", "-y", "-loglevel", "error"] + [a for c in clips for a in ("-i", c)]
           + ["-filter_complex", ";".join(parts), "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
              "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out])
    result = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-800:]}")


class Film:
    """One film's folder: plan.json (the settings and Claude's shot list), shotN.mp4, film.mp4."""

    def __init__(self, work: Path, emit):
        self.work, self.emit, self.steps = work, emit, []

    @property
    def meta(self) -> dict:
        return json.loads((self.work / "plan.json").read_text())

    def save(self, meta: dict) -> None:
        (self.work / "plan.json").write_text(json.dumps(meta, indent=1))

    async def timed(self, name: str, model: str, call):
        t = time.perf_counter()
        body = await call
        step = {"name": name, "model": model, "seconds": round(time.perf_counter() - t, 1), "cost": float(body.get("cost") or 0)}
        self.steps.append(step)
        return body, step

    async def shoot(self, meta: dict, i: int, image: str | None = None) -> str:
        """Render shot i (image-to-video from `image` when chaining), save it and announce it."""
        plan, shot = meta["plan"], meta["plan"]["shots"][i]
        params = {"aspect_ratio": meta["aspect"], "generate_audio_switch": True, "quality": "720p"}
        self.emit({"event": "shooting", "index": i})
        body, step = await self.timed(f"shot {i + 1}", meta["model"], edenai.video(
            meta["model"], shot_prompt(plan["style"], shot), shot["seconds"], params, image))
        name = f"shot{i + 1}.mp4"
        (self.work / name).write_bytes(await edenai.download(body["output"]["video_resource_url"]))
        self.emit({"event": "shot", "index": i, "file": name, **step})
        return name

    async def chained_from(self, i: int) -> str:
        """Upload the last frame of shot i, so the next shot starts exactly where this one ends."""
        last_frame(self.work / f"shot{i + 1}.mp4", self.work / f"last{i + 1}.png")
        return (await edenai.upload((self.work / f"last{i + 1}.png").read_bytes(), f"last{i + 1}.png", "image/png"))["file_id"]

    async def cut(self, meta: dict, clips: list[str], started: float) -> dict:
        self.emit({"event": "editing"})
        t = time.perf_counter()
        await asyncio.to_thread(edit, self.work, clips, meta["plan"]["title"], meta["aspect"])
        self.steps.append({"name": "edit", "model": "ffmpeg", "seconds": round(time.perf_counter() - t, 1), "cost": 0.0})
        done = {"event": "done", "folder": self.work.name, "video": "film.mp4", "length": round(seconds(self.work / "film.mp4"), 1),
                "steps": self.steps, "cost": round(sum(s["cost"] for s in self.steps), 4), "seconds": round(time.perf_counter() - started)}
        self.emit(done)
        return done


async def make_film(premise: str, out: Path, model: str, aspect: str, shots: int = 4, mode: str = "shots",
                    director: str = "anthropic/claude-sonnet-latest", emit=lambda event: None) -> dict:
    """mode: "shots" (each shot separately, in parallel), "chain" (each shot starts from the previous one's last frame)
    or "multishot" (one PixVerse take that cuts between the shots itself)."""
    started = time.perf_counter()
    work = out / f"{datetime.datetime.now():%Y-%m-%d-%H%M%S}-{slug(premise)}"
    work.mkdir(parents=True, exist_ok=True)
    film = Film(work, emit)
    emit({"event": "directing", "folder": work.name, "director": DIRECTORS.get(director, director)})

    # 1. Claude directs: a style bible and the shot list, as structured JSON
    reply, _ = await film.timed("direction", director, edenai.chat(
        [{"role": "system", "content": DIRECTOR.format(n=shots, aspect=aspect)}, {"role": "user", "content": premise}],
        director, [d for d in DIRECTORS if d != director], FILM))
    plan = json.loads(re.search(r"\{.*\}", reply["choices"][0]["message"]["content"], re.S).group())
    plan["shots"] = plan["shots"][:shots]
    for s in plan["shots"]:
        s["seconds"] = max(3, min(5, int(s["seconds"])))
    meta = {"premise": premise, "model": model, "aspect": aspect, "mode": mode, "director": director, "plan": plan}
    film.save(meta)
    emit({"event": "plan", **plan})

    # 2. PixVerse shoots
    if mode == "multishot":  # one generation; PixVerse makes the cuts
        total = min(15, sum(s["seconds"] for s in plan["shots"]))
        prompt = plan["style"] + "\n\n" + "\n".join(
            f"Shot {i + 1} ({s['seconds']} s, {s['name']}): {s['action']} Sound: {s['sound']}" for i, s in enumerate(plan["shots"]))
        params = {"aspect_ratio": aspect, "generate_audio_switch": True, "generate_multi_clip_switch": True, "quality": "720p"}
        emit({"event": "shooting", "index": -1})
        body, step = await film.timed("multi-shot take", model, edenai.video(model, prompt, total, params))
        (work / "take.mp4").write_bytes(await edenai.download(body["output"]["video_resource_url"]))
        emit({"event": "shot", "index": -1, "file": "take.mp4", **step})
        clips = ["take.mp4"]
    elif mode == "chain":  # continuity: each shot starts from the last frame of the one before
        clips, image = [], None
        for i in range(len(plan["shots"])):
            clips.append(await film.shoot(meta, i, image))
            if i + 1 < len(plan["shots"]):
                image = await film.chained_from(i)
    else:  # every shot at once
        clips = list(await asyncio.gather(*(film.shoot(meta, i) for i in range(len(plan["shots"])))))

    # 3. The edit
    return await film.cut(meta, clips, started)


async def reshoot(out: Path, folder: str, index: int, action: str, emit=lambda event: None) -> dict:
    """The director's note: re-render one shot with new action, then re-cut the film."""
    started = time.perf_counter()
    film = Film(out / folder, emit)
    meta = film.meta
    if meta["mode"] == "multishot":
        raise RuntimeError("a multi-shot take is one generation: make a new film instead")
    if not 0 <= index < len(meta["plan"]["shots"]):
        raise RuntimeError(f"there is no shot {index + 1}")
    meta["plan"]["shots"][index]["action"] = action.strip() or meta["plan"]["shots"][index]["action"]
    film.save(meta)
    image = await film.chained_from(index - 1) if meta["mode"] == "chain" and index > 0 else None
    await film.shoot(meta, index, image)
    return await film.cut(meta, [f"shot{i + 1}.mp4" for i in range(len(meta["plan"]["shots"]))], started)
