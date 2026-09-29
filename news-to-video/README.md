# News to video

**Give it a topic. About 30 seconds later you have a narrated 9:16 news clip made from today's web.**

```
topic → Tavily web search → Nebius LLM writes a headline, a narration and 3 shots
      → Pruna renders the 3 clips in parallel, while Gradium reads the narration
      → ffmpeg joins them into a vertical video with captions
```

Four providers, one Eden AI key. Pruna's fast video models are what make the whole thing fit in about half a minute. Each clip comes with its own soundtrack, which plays at 20% under the narration.

## Run it

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org) (`brew install ffmpeg` or `sudo apt install ffmpeg`).

```bash
cd news-to-video
pip install -r requirements.txt
cp .env.example .env                      # then paste your Eden AI key
python news_to_video.py "European AI startups"
```

A real run on 2026-09-29:

```
Searching the web for: European AI startups
Writing the script from 6 results
  "Europe's AI Startup Boom"
  narration (33 words): European AI startups are surging, with Finland's Verda hitting unicorn status ...
Rendering 3 clips with video/generation_async/pruna/p-video while audio/tts/gradium reads the narration
Stitching with ffmpeg

  web search  web/search/tavily                               0.5 s  $0.0080
  script      nebius/openai/gpt-oss-120b                      3.0 s  $0.0008
  narration   audio/tts/gradium                               5.7 s  $0.0121
  clip 2      video/generation_async/pruna/p-video           18.0 s  $0.1000
  clip 1      video/generation_async/pruna/p-video           20.6 s  $0.1000
  clip 3      video/generation_async/pruna/p-video           20.9 s  $0.1000
  stitch      ffmpeg                                          4.3 s  $0.0000

Done in 31 s for $0.3209 via Eden AI
Video:  out/2026-09-29-european-ai-startups/news.mp4
Script: out/2026-09-29-european-ai-startups/script.md
```

The output folder holds:
- **`news.mp4`:** 720×1280, 24 fps, H.264 plus AAC, about 15 s. It shows the captions "Finland's Verda Unicorn", "London Nscale $2B" and "AI Race Across Europe", with an *AI-generated* label on every frame.
- **`script.md`:** the headline, the narration, the shot prompts, the sources, and the timing and cost table.
- **The raw pieces:** `shot1-3.mp4` and the narration audio.

## Demo script (about 1 minute)

1. Ask the audience for a topic, and run `python news_to_video.py "<topic>"` in a terminal at a large font size.
2. While it runs, point at the output: the search, then the script, then **three Pruna clips rendering at the same time**.
3. Play `news.mp4`, then open `script.md` to show the sources and the cost line: about 30 s and about $0.32, on one bill.

## How it works

| File | What it does |
|---|---|
| `news_to_video.py` | Runs the pipeline: search → script (structured JSON) → 3 clips and the narration in parallel → `ffmpeg` stitch → `script.md` |
| `edenai.py` | The Eden AI calls: `web/search/tavily` (falls back to linkup), `/v3/chat/completions` with a JSON schema, `video/generation_async/pruna/p-video` (async job plus polling), `audio/tts/gradium`, and a plain download of the results |

The details that keep it reliable:
- **One search:** its results go to the LLM as untrusted data, and the LLM may only state facts found in them.
- **No real people:** shot prompts can't contain text, logos, real faces or named people. News is about real people, and a video model shouldn't invent their likeness.
- **Narration fits the clips:** it's capped at 34 words, about 14 s at Gradium's pace, by keeping whole sentences. If it still runs long, the last frame holds until it ends.
- **Captions fit the frame:** the font shrinks for long captions, and `ffmpeg` reads them from files, so `%`, `:` and quotes need no escaping.
- **Any video model:** if one returns clips without sound, the video keeps just the narration.

## Options

| Flag | Default | Notes |
|---|---|---|
| `--video-model` | `video/generation_async/pruna/p-video` | `.../pruna/p-video-2` has higher quality but was much slower in our tests (about 96 s for one clip) |
| `--llm` | `nebius/openai/gpt-oss-120b` | Falls back to `nebius/Qwen/Qwen3-235B-A22B-Instruct-2507`. Qwen took about 9 s here against about 3 s for gpt-oss |
| `--tts` | `audio/tts/gradium` | Any Eden AI TTS, e.g. `audio/tts/deepgram/aura-2` or `audio/tts/elevenlabs` |
| `--no-captions` | captions on | |
| `--out` | `out/` | One folder per run, named `<date>-<topic>` |

## Cost and timing

About **$0.32 per video**. Almost all of it is the three 5-second Pruna clips ($0.02 per second, so $0.10 each); the search, LLM and narration add about $0.02 together. End to end it takes **30–65 s**. The clips dominate, because Pruna's latency through Eden AI varied from about 16 s to 45 s per clip across our runs. The three render in parallel, so a run waits for the slowest one.

**P-Video-2-Pro**, Pruna's newest model (a 5 s clip in about 2 s in Speed mode), isn't on Eden AI yet. When it is, pass it as `--video-model` and the clip step should shrink to a few seconds.

## Responsible use

Every frame is labelled *AI-generated*, and `script.md` lists the sources the narration is based on. It's a demo: check the facts before publishing anything it makes.
