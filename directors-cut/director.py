"""Director's Cut: an LLM directs, Gemini draws the storyboard, PixVerse shoots, ffmpeg edits. One Eden AI key.

premise -> the director writes the film: a cast (one fixed look per character), a world (place, time of day, light),
           a style, and N shots (first frame, one action, one camera move, sound, length)
        -> Gemini draws a reference sheet per character, then the first frame of every shot from those sheets
        -> the director reviews the storyboard and has any frame with a defect redrawn, once
        -> PixVerse animates every frame with native audio (image-to-video), or one multi-shot take from the first frame
        -> ffmpeg levels the sound and cuts the shots together behind a title card
"""
import asyncio
import base64
import datetime
import json
import re
import subprocess
import time
from pathlib import Path

import edenai

MODELS = {"video/generation_async/pixverse/v6": "PixVerse V6", "video/generation_async/pixverse/c1": "PixVerse C1"}
MULTI_SHOT = {"video/generation_async/pixverse/v6"}  # native multi-shot takes
ARTIST = "image/generation/google/gemini-2.5-flash-image"  # draws the storyboard from reference sheets
REVIEWER = "anthropic/claude-sonnet-latest"  # reviews the storyboard for directors that can't see images
# Who writes the shot list. Each is pinned to an endpoint that supports JSON-schema output, with a fallback serving the
# same model elsewhere. GPT-6 is served by Azure: OpenAI's own endpoint returned 401s on our account when we tested.
DIRECTORS = {  # id: (label, group, extra request fields, fallbacks, can see images)
    "anthropic/claude-sonnet-latest": ("Claude Sonnet", "Anthropic", {}, ["anthropic/claude-opus-latest"], True),
    "anthropic/claude-opus-latest": ("Claude Opus", "Anthropic", {}, ["anthropic/claude-sonnet-latest"], True),
    "azure/gpt-6-astra": ("GPT-6 Astra", "OpenAI", {}, ["openai/gpt-6-astra"], True),
    "azure/gpt-6-sol": ("GPT-6 Sol", "OpenAI", {}, ["openai/gpt-6-sol"], True),
    "azure/gpt-6-luna": ("GPT-6 Luna", "OpenAI", {}, ["openai/gpt-6-luna"], True),
    "moonshot/kimi-k3": ("Kimi K3", "Open weights", {"reasoning_effort": "low"}, ["together_ai/moonshotai/Kimi-K3"], True),
    "zai/glm-5.3": ("GLM 5.3 (always thinks: 2-4 min)", "Open weights", {"reasoning_effort": "low", "max_tokens": 16000},
                    ["deepinfra/zai-org/GLM-5.3"], False),  # it thinks for 6,000+ tokens before it writes
    "nebius/deepseek-ai/DeepSeek-V4-Pro": ("DeepSeek V4 Pro", "Open weights", {}, ["deepinfra/deepseek-ai/DeepSeek-V4-Pro"], False),
}
SIZES = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (720, 720), "21:9": (1680, 720)}  # the edit's frame size
DRAW = {"16:9": "1344x768", "9:16": "768x1344", "1:1": "1024x1024", "21:9": "1536x672"}  # the storyboard's frame size


def obj(**fields) -> dict:  # a strict JSON-schema object
    return {"type": "object", "additionalProperties": False, "required": list(fields), "properties": fields}


TEXT, NUMBER, YES_NO = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
FILM = obj(title=TEXT, logline=TEXT, cast={"type": "array", "items": obj(tag=TEXT, look=TEXT)}, world=TEXT, style=TEXT,
           shots={"type": "array", "items": obj(name=TEXT, cast={"type": "array", "items": TEXT}, frame=TEXT, action=TEXT,
                                                camera=TEXT, sound=TEXT, seconds=NUMBER)})
REVIEW = obj(shots={"type": "array", "items": obj(shot=NUMBER, ok=YES_NO, problem=TEXT, frame=TEXT)})

# The rules below come from the PixVerse, Veo, Sora and Runway prompting guides and from research on AI film pipelines:
# one visible action per shot, written first; the frame sets the action up; looks and light pasted word for word.
DIRECTOR = """You direct a short film made by AI. An image model draws the first frame of every shot, a video model
(PixVerse) animates each frame for a few seconds with its own sound, and the shots are cut together in order, in {aspect}.
Each shot is generated on its own by models that know nothing about the story and take every word literally.

- "title": at most 6 words. "logline": one sentence.
- "cast": one or two characters (a person, an animal, a robot, a car...). "tag" is a 2-4 word visual handle, used
  everywhere instead of a name ("the old keeper", "the brass robot"). "look" is 20-40 words of appearance only: species or
  age, build, face and hair, clothes or materials, colours. It is pasted word for word wherever the character appears.
- "world": 30-50 words that fix the place, the time of day, the weather, the key light (its source and direction) and
  3-5 palette colours. It is pasted word for word into every frame, so the light never changes between cuts.
- "style": at most 20 words: medium, lens and film stock (or an animation style).
- "shots": exactly {n}. Tell the story so it reads with the sound off: establish, build, pay off. Move between wide,
  medium and close-up, and make every cut show something new. For each shot:
  - "cast": the tags of the characters visible in the frame (it can be empty).
  - "frame": the still first frame in 25-50 words: shot size and angle, who is where, their pose just before the action,
    what they hold. It sets up the action: if a character will pour, the can is already in their hand. Leave out the
    looks, the world and the style: they are added to every frame for you.
  - "action": 12-30 words, present tense, starting with the subject: ONE large, visible physical action and its visible
    result, like "The brass robot tips the watering can and a stream of water splashes onto the dry soil." Show feelings
    through the body ("her shoulders drop"), never with "realises", "hopes" or "decides". Small things need a close shot:
    a single drop is an extreme close-up. Cause and consequence are two shots, not one.
  - "camera": one move with a speed word: "slow push-in", "steady tracking shot to the left", "slow crane up",
    "locked camera", "gentle handheld".
  - "sound": 2-4 concrete sounds from the scene, tied to what happens ("a hollow clank as the can hits the rim"), plus
    at most one short line of dialogue written like: the old keeper (whispering): 'Who sent this?'. No music.
  - "seconds": 4 to 6. Five suits most actions.
- Write what is there, not what is missing. Keep the picture free of writing (signs, captions, logos) and of real,
  named people."""
CRITIC = """You directed this film and are checking its storyboard before it is animated, the expensive step. You get the
plan, the character reference sheets and the drawn first frame of shots {shots}. For each frame, look only for defects
that would ruin the shot:
- a character doesn't match their sheet, is missing, is doubled, or someone who isn't in the shot's cast appears;
- the place, time of day or light contradicts the world;
- the frame doesn't set up the action (the prop the character must use isn't there, or they face the wrong way);
- there is writing in the picture, or a clearly broken hand, face or object.
Taste is not a defect. For a good frame set "ok" to true and leave "problem" and "frame" empty. Otherwise set "ok" to
false, say what is wrong in "problem" (under 12 words), and rewrite the shot's "frame" description so a redraw fixes it."""


def key(tag: str) -> str:  # "The brass robot" and "brass robot" are the same character
    return re.sub(r"^(the|a|an) ", "", tag.lower().strip())


def in_shot(plan: dict, shot: dict) -> list[int]:
    """Which cast members (by index) appear in a shot."""
    names = {key(c) for c in shot["cast"]}
    return [k for k, c in enumerate(plan["cast"]) if key(c["tag"]) in names]


def looks(plan: dict, shot: dict | None = None) -> str:
    """The cast's fixed descriptions, word for word: everyone in `shot`, or the whole cast."""
    who = in_shot(plan, shot) if shot else range(len(plan["cast"]))
    return " ".join(f"{plan['cast'][k]['tag'][0].upper() + plan['cast'][k]['tag'][1:]}: {plan['cast'][k]['look'].rstrip('.')}." for k in who)


def sheet_prompt(plan: dict, who: dict) -> str:
    return (f"Character reference sheet for a film: {who['tag']}, {who['look'].rstrip('.')}. Full body, three-quarter view, "
            f"neutral pose, plain light grey background, soft even light. {plan['style']}. One character, no text.")


def frame_prompt(plan: dict, i: int, aspect: str) -> str:
    shot = plan["shots"][i]
    same = "Draw the characters exactly as in their reference sheets" + (
        ", in the same place and light as the establishing frame" if i else "") + "."
    return (f"Cinematic film still, {aspect} frame. {shot['frame']}\n{looks(plan, shot)}\n{plan['world']}\n"
            f"{plan['style']}. {same} No text, captions or watermark.")


def motion_prompt(shot: dict) -> str:
    """Image-to-video: the frame already shows who and where, so the prompt says only what moves, and what we hear.
    The action comes first: PixVerse follows the first sentence most closely."""
    return (f"{shot['action']} {shot['camera'][0].upper() + shot['camera'][1:].rstrip('.')}. Keep every character exactly "
            f"as in the image, moving naturally.\nAudio: {shot['sound'].rstrip('.')}. No music, no subtitles.")


def take_prompt(plan: dict, total: int) -> str:
    """A multi-shot take, in the timecoded format PixVerse documents, with the looks repeated for every cut."""
    lines, start, scale = [], 0, total / sum(s["seconds"] for s in plan["shots"])
    for i, s in enumerate(plan["shots"]):
        end = total if i == len(plan["shots"]) - 1 else round(start + s["seconds"] * scale)
        lines.append(f"Shot {i + 1}: 00:{start:02d} - 00:{end:02d}. {s['camera'][0].upper() + s['camera'][1:].rstrip('.')}. {s['action']}")
        start = end
    sounds = "; ".join(s["sound"].rstrip(".") for s in plan["shots"])
    return "\n".join(lines) + f"\nThe same characters in every shot. {looks(plan)}\n{plan['world']}\nAudio: {sounds}. No music, no subtitles."


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "film"


def seconds(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def loudness(path: Path) -> float | None:
    """Integrated loudness in LUFS, or None when the clip has no sound."""
    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "loudnorm=print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    found = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", err)
    value = json.loads(found.group())["input_i"] if found else "-inf"
    return None if "inf" in value else float(value)


def edit(work: Path, clips: list[str], title: str, credit: str, aspect: str, out: str = "film.mp4") -> None:
    """Cut the clips together at one frame size and one loudness, with a title card and an AI-generated label.
    PixVerse renders every shot's sound on its own, so their levels differ by up to 15 LU: each shot is levelled to the
    same loudness first (with at most +30 dB: PixVerse's tracks can be very quiet), then the whole cut is normalized."""
    w, h = SIZES[aspect]
    (work / "title.txt").write_text(title)
    (work / "credit.txt").write_text(credit)
    fade = "alpha='if(lt(t,2.2),1,max(0,1-(t-2.2)*2))'"
    text = "expansion=none:fontcolor=white:borderw=3:bordercolor=black@0.6:x=(w-text_w)/2"
    parts, joined = [], ""
    for i, clip in enumerate(clips):
        parts.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24[v{i}]")
        level = loudness(work / clip)
        if level is None:  # a silent shot
            parts.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={seconds(work / clip):.3f}[a{i}]")
        else:  # level it, and fade 30 ms at both ends so the cuts don't click
            parts.append(f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={max(-10, min(30, -20 - level)):.1f}dB,"
                         f"afade=t=in:d=0.03,areverse,afade=t=in:d=0.03,areverse[a{i}]")
        joined += f"[v{i}][a{i}]"
    parts.append(f"{joined}concat=n={len(clips)}:v=1:a=1[cut][mix]")
    parts.append("[mix]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]")
    parts.append(f"[cut]drawtext=textfile=title.txt:fontsize={h // 11}:{text}:y=(h-text_h)/2-{h // 20}:{fade},"
                 f"drawtext=textfile=credit.txt:fontsize={h // 34}:{text}:y=(h/2)+{h // 14}:{fade},"
                 f"drawtext=text='AI-generated':fontsize={h // 40}:{text}:y={h // 30}[v]")
    cmd = (["ffmpeg", "-y", "-loglevel", "error"] + [a for c in clips for a in ("-i", c)]
           + ["-filter_complex", ";".join(parts), "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
              "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out])
    result = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-800:]}")


def thumbnail(path: Path, width: int) -> str:
    """A small JPEG of an image as a data URL, for the director to look at."""
    jpeg = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"scale={width}:-2", "-frames:v", "1", "-q:v", "4",
                           "-f", "mjpeg", "-"],
                          capture_output=True, check=True).stdout
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


class Film:
    """One film's folder: plan.json (the settings and the director's plan), castN.png, frameN.png, shotN.mp4, film.mp4."""

    def __init__(self, work: Path, emit):
        self.work, self.emit, self.steps, self.ids = work, emit, [], {}

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
        self.emit({"event": "step", **step})
        return body

    async def file_id(self, name: str) -> str:
        """Upload a drawing once per run, so Gemini and PixVerse can use it."""
        if name not in self.ids:
            self.ids[name] = (await edenai.upload((self.work / name).read_bytes(), name, "image/png"))["file_id"]
        return self.ids[name]

    async def draw(self, name: str, step: str, prompt: str, references: list[str], size: str) -> None:
        body = await self.timed(step, ARTIST, edenai.image(ARTIST, prompt, [await self.file_id(r) for r in references], size))
        if not body["output"].get("items") and references:  # Gemini sometimes returns no image for a reference photo of a
            body = await self.timed(f"{step}, retried without references", ARTIST, edenai.image(ARTIST, prompt, [], size))  # person
        if not body["output"].get("items"):
            raise RuntimeError(f"Gemini returned no image for {step}: try rewording the premise")
        item = body["output"]["items"][0]
        (self.work / name).write_bytes(base64.b64decode(item["image"]) if item.get("image") else await edenai.download(item["image_resource_url"]))
        self.ids.pop(name, None)  # a redraw needs a fresh upload

    async def draw_frame(self, meta: dict, i: int) -> None:
        """Shot i's first frame, drawn from the sheets of the characters in it and, after shot 1, the establishing frame."""
        plan = meta["plan"]
        sheets = [f"cast{k + 1}.png" for k in in_shot(plan, plan["shots"][i])]
        await self.draw(f"frame{i + 1}.png", f"frame {i + 1}", frame_prompt(plan, i, meta["aspect"]),
                        sheets + (["frame1.png"] if i else []), DRAW[meta["aspect"]])
        self.emit({"event": "frame", "index": i, "file": f"frame{i + 1}.png"})

    async def storyboard(self, meta: dict, frames: list[int]) -> None:
        plan = meta["plan"]
        self.emit({"event": "drawing"})

        async def sheet(k: int, who: dict):
            await self.draw(f"cast{k + 1}.png", f"cast sheet {k + 1}", sheet_prompt(plan, who), [], "1024x1024")
            self.emit({"event": "cast", "index": k, "file": f"cast{k + 1}.png"})
        await asyncio.gather(*(sheet(k, who) for k, who in enumerate(plan["cast"])))
        await self.draw_frame(meta, frames[0])  # the establishing frame fixes the place and the light for the others
        await asyncio.gather(*(self.file_id(f"cast{k + 1}.png") for k in range(len(plan["cast"]))), self.file_id("frame1.png"))
        await asyncio.gather(*(self.draw_frame(meta, i) for i in frames[1:]))

    async def review(self, meta: dict, frames: list[int]) -> None:
        """The director looks at the storyboard and has frames with a defect redrawn, once. Cheap frames are checked
        before anything is animated."""
        plan, director = meta["plan"], meta["director"]
        _, _, extra, fallbacks, sees = DIRECTORS[director]
        model, extra, fallbacks = (director, extra, fallbacks) if sees else (REVIEWER, {}, DIRECTORS[REVIEWER][3])
        self.emit({"event": "reviewing", "model": model})
        content = [{"type": "text", "text": json.dumps(plan)}]
        for k, who in enumerate(plan["cast"]):
            content += [{"type": "text", "text": f"Reference sheet: {who['tag']}"},
                        {"type": "image_url", "image_url": {"url": thumbnail(self.work / f"cast{k + 1}.png", 384)}}]
        for i in frames:
            content += [{"type": "text", "text": f"Shot {i + 1}, first frame"},
                        {"type": "image_url", "image_url": {"url": thumbnail(self.work / f"frame{i + 1}.png", 640)}}]
        shots = ", ".join(str(i + 1) for i in frames)
        reply = await self.timed("storyboard review", model, edenai.chat(
            [{"role": "system", "content": CRITIC.format(shots=shots)}, {"role": "user", "content": content}],
            model, fallbacks, REVIEW, extra, name="review"))
        notes = json.loads(re.search(r"\{.*\}", reply["choices"][0]["message"]["content"], re.S).group())["shots"]
        redraw = [n for n in notes if not n["ok"] and n["frame"].strip() and n["shot"] - 1 in frames]
        for n in redraw:
            plan["shots"][n["shot"] - 1]["frame"] = n["frame"].strip()
        self.save(meta)
        self.emit({"event": "review", "notes": [{"index": n["shot"] - 1, "problem": n["problem"], "frame": n["frame"]} for n in redraw]})
        await asyncio.gather(*(self.draw_frame(meta, n["shot"] - 1) for n in redraw))

    async def shoot(self, meta: dict, i: int) -> str:
        """Animate shot i from its first frame, with native audio."""
        shot = meta["plan"]["shots"][i]
        params = {"aspect_ratio": meta["aspect"], "generate_audio_switch": True, "quality": "720p"}
        self.emit({"event": "shooting", "index": i})
        body = await self.timed(f"shot {i + 1}", meta["model"], edenai.video(
            meta["model"], motion_prompt(shot), shot["seconds"], params, await self.file_id(f"frame{i + 1}.png")))
        name = f"shot{i + 1}.mp4"
        (self.work / name).write_bytes(await edenai.download(body["output"]["video_resource_url"]))
        self.emit({"event": "shot", "index": i, "file": name})
        return name

    async def take(self, meta: dict) -> str:
        """One multi-shot generation from the first frame: PixVerse makes the cuts."""
        plan = meta["plan"]
        total = min(15, sum(s["seconds"] for s in plan["shots"]))
        params = {"aspect_ratio": meta["aspect"], "generate_audio_switch": True, "generate_multi_clip_switch": True, "quality": "720p"}
        self.emit({"event": "shooting", "index": -1})
        body = await self.timed("multi-shot take", meta["model"], edenai.video(
            meta["model"], take_prompt(plan, total), total, params, await self.file_id("frame1.png")))
        (self.work / "take.mp4").write_bytes(await edenai.download(body["output"]["video_resource_url"]))
        self.emit({"event": "shot", "index": -1, "file": "take.mp4"})
        return "take.mp4"

    async def cut(self, meta: dict, clips: list[str], started: float) -> dict:
        self.emit({"event": "editing"})
        t = time.perf_counter()
        credit = f"directed by {DIRECTORS[meta['director']][0].split(' (')[0]}  ·  storyboard by Gemini  ·  shot on PixVerse  ·  via Eden AI"
        await asyncio.to_thread(edit, self.work, clips, meta["plan"]["title"], credit, meta["aspect"])
        step = {"name": "edit", "model": "ffmpeg", "seconds": round(time.perf_counter() - t, 1), "cost": 0.0}
        self.steps.append(step)
        self.emit({"event": "step", **step})
        done = {"event": "done", "folder": self.work.name, "video": "film.mp4", "length": round(seconds(self.work / "film.mp4"), 1),
                "cost": round(sum(s["cost"] for s in self.steps), 4), "seconds": round(time.perf_counter() - started)}
        self.emit(done)
        return done


async def make_film(premise: str, out: Path, model: str, aspect: str, shots: int = 4, mode: str = "storyboard",
                    director: str = "anthropic/claude-sonnet-latest", emit=lambda event: None) -> dict:
    """mode: "storyboard" (every frame drawn, reviewed and animated, in parallel) or "multishot" (one PixVerse take,
    started from the first frame, that cuts between the shots itself)."""
    started = time.perf_counter()
    work = out / f"{datetime.datetime.now():%Y-%m-%d-%H%M%S}-{slug(premise)}"
    work.mkdir(parents=True, exist_ok=True)
    film = Film(work, emit)
    label, _, extra, fallbacks, _ = DIRECTORS[director]
    emit({"event": "directing", "folder": work.name, "director": label})

    # 1. The director writes the film as structured JSON: cast, world, style and shots
    reply = await film.timed("direction", director, edenai.chat(
        [{"role": "system", "content": DIRECTOR.format(n=shots, aspect=aspect)}, {"role": "user", "content": premise}],
        director, fallbacks, FILM, extra))
    plan = json.loads(re.search(r"\{.*\}", reply["choices"][0]["message"]["content"], re.S).group())
    plan["cast"], plan["shots"] = plan["cast"][:2], plan["shots"][:shots]
    for s in plan["shots"]:
        s["seconds"] = max(4, min(6, int(s["seconds"])))
    meta = {"premise": premise, "model": model, "aspect": aspect, "mode": mode, "director": director, "plan": plan}
    film.save(meta)
    emit({"event": "plan", **plan})

    # 2. Gemini draws the storyboard, and 3. the director reviews it
    frames = [0] if mode == "multishot" else list(range(len(plan["shots"])))
    await film.storyboard(meta, frames)
    await film.review(meta, frames)

    # 4. PixVerse animates it
    if mode == "multishot":
        clips = [await film.take(meta)]
    else:
        clips = list(await asyncio.gather(*(film.shoot(meta, i) for i in frames)))

    # 5. The edit
    return await film.cut(meta, clips, started)


async def reshoot(out: Path, folder: str, index: int, frame: str, action: str, emit=lambda event: None) -> dict:
    """The director's note: re-render one shot with a new action (and a redrawn first frame if its description changed),
    then re-cut the film."""
    started = time.perf_counter()
    film = Film(out / folder, emit)
    meta = film.meta
    if meta["mode"] == "multishot":
        raise RuntimeError("a multi-shot take is one generation: make a new film instead")
    if not 0 <= index < len(meta["plan"]["shots"]):
        raise RuntimeError(f"there is no shot {index + 1}")
    shot = meta["plan"]["shots"][index]
    redraw = bool(frame.strip()) and frame.strip() != shot["frame"]
    shot["frame"], shot["action"] = frame.strip() or shot["frame"], action.strip() or shot["action"]
    film.save(meta)
    if redraw:
        await film.draw_frame(meta, index)
    await film.shoot(meta, index)
    return await film.cut(meta, [f"shot{i + 1}.mp4" for i in range(len(meta["plan"]["shots"]))], started)
