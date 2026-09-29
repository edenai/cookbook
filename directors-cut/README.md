# Director's Cut

**Describe a film. An LLM director (Claude, GPT-6 or an open-weight model) writes and cuts the shot list, PixVerse shoots every shot with its own sound, and you get the edit in about a minute.**

```
premise → the director (Claude by default) writes a style bible + N shots (action, camera move, sound, length)
        → PixVerse renders every shot with native audio: all at once, chained frame to frame, or as one multi-shot take
        → ffmpeg cuts them together behind a title card → reshoot any shot with a new note
```

| At a glance | |
|---|---|
| Providers | Director: Claude, GPT-6, Kimi K3, GLM 5.3 or DeepSeek V4 Pro. Camera: PixVerse V6, C1 or V5.5 |
| One film | 4 shots, 16 s, 720p with sound: about 60–65 s and $0.78 (3 shots: about $0.59). A reshoot takes about 45 s and costs $0.19 |
| Runs as | A local web page (FastAPI plus one HTML file) |
| Eden AI endpoints | `/v3/chat/completions` (JSON schema), `/v3/universal-ai/async`, `/v3/upload` |

## Why these providers

| Step | Provider, via Eden AI | Why it's a great fit here |
|---|---|---|
| Direct | **Claude** `anthropic/claude-sonnet-latest` | Directing is a writing job, and Claude's structured output returns the whole film as one strict JSON shot list. It also writes a "style bible" that describes the character identically in every shot. On the e-bike ad that kept the same rider, kit and bike across all four shots. About 12–17 s and $0.01–0.02. Seven other directors are one dropdown away (below). |
| Shoot | **PixVerse V6** `video/generation_async/pixverse/v6` | V6 is PixVerse's general-purpose flagship, and this demo uses the strengths that matter for short films. **Native audio** (`generate_audio_switch`) means every shot comes with its own ambience and effects, so there's no separate sound step. It follows camera language written into the prompt (push-in, tracking, handheld, crane up). **Image-to-video** makes continuity mode possible. **Multi-shot takes** (`generate_multi_clip_switch`) let PixVerse cut a whole sequence in one generation. A 4 s, 720p shot with sound cost $0.19 and took 40–46 s. |
| Shoot, cinematic | **PixVerse C1** | PixVerse's cinematic, action-oriented model, for motion-heavy scenes. It can't do multi-shot takes, so the page disables that mode for C1. |

### Pick your director

The director can be any of eight LLMs, all through the same Eden AI key, and all tested with the same premise ("a lighthouse keeper finds a glowing message in a bottle during a storm"). Each returned a valid 4-shot plan with a style bible.

| Director | Eden AI model | Time to write the film | Cost |
|---|---|---|---|
| Claude Sonnet (default) | `anthropic/claude-sonnet-latest` | 12–17 s | $0.01–0.02 |
| Claude Opus | `anthropic/claude-opus-latest` | 18 s | $0.026 |
| GPT-6 Astra | `azure/gpt-6-astra` | 22 s | $0.044 |
| GPT-6 Sol | `azure/gpt-6-sol` | 10–12 s | $0.009–0.011 |
| GPT-6 Luna | `azure/gpt-6-luna` | 6.5 s | $0.0004 |
| Kimi K3 (open weights) | `moonshot/kimi-k3`, with `reasoning_effort: low` | 23 s (58 s at default effort) | $0.009 |
| GLM 5.3 (open weights) | `zai/glm-5.3` | 44–90 s: it always thinks, and Z.ai refuses to turn that off | $0.012–0.015 |
| DeepSeek V4 Pro (open weights) | `nebius/deepseek-ai/DeepSeek-V4-Pro` | 4.5 s | $0.003 |

Each director is pinned to an endpoint that supports JSON-schema output, with a fallback that serves the same model elsewhere:
- **GPT-6 goes through Azure.** OpenAI's own endpoint returned `401 Incorrect API key` on our account, which is a problem with Eden AI's platform key. OpenAI stays as the fallback.
- **DeepSeek V4 Pro goes through Nebius.** DeepSeek's own endpoint rejects the JSON-schema response format.

**Why Eden AI ties it together:**
- **One key** covers all eight directors and the camera.
- **Swappable models:** each director or PixVerse model is one string, with a fallback serving the same model on another provider.
- **Visible cost:** every call is tagged `demo=directors-cut`, and its time and cost appear in the page's table.

## Run it

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org).

```bash
cd directors-cut
pip install -r requirements.txt
cp .env.example .env              # then paste your Eden AI key
uvicorn app:app                   # open http://localhost:8000
```

## Demo script (about 2 minutes)

1. **Pitch a film.** Pick an example or type a premise, for example "a lighthouse keeper finds a glowing message in a bottle during a storm". Leave *Every shot at once* selected and press **Action!**.
2. **The director writes the film** (about 5–20 s, depending on the director). The title, logline and style bible appear, then a storyboard card per shot, each with its camera move and its sound. Try the same premise with Claude, GPT-6 and Kimi K3 to compare directors.
3. **PixVerse shoots.** All four shots render at once and pop into their cards as they land, each with its own sound.
4. **The cut** plays on the right, with the title card and the time and cost of every call.
5. **Director's note.** Edit one card's direction, for example "slow-motion close-up, camera tilts up to her face", and press **Reshoot this shot**. Only that shot is re-rendered, and the film is re-cut in about 45 s.

## Shooting modes

| Mode | What happens | Time for 4 shots |
|---|---|---|
| Every shot at once | Each shot renders in parallel; the style bible keeps them consistent | about 65 s |
| Continuity | Each shot starts from the last frame of the one before (image-to-video), so the cuts flow into each other | about 3–4 min (one after another) |
| One multi-shot take | Claude's shot list becomes one prompt, and V6 or V5.5 renders a single take (up to 15 s) that cuts between the shots itself | about 1–1.5 min |

## How it works

| File | What it does |
|---|---|
| `director.py` | `DIRECTORS` (the eight LLMs and their settings) and `make_film()`: the director's shot list (JSON schema), the three shooting modes, and the edit. `reshoot()`: re-renders one shot and re-cuts. Each step is reported as it finishes |
| `app.py` | `POST /api/film` and `POST /api/reshoot` stream those step updates as one JSON object per line. `/out/` serves the shots and the film |
| `edenai.py` | The Eden AI calls: chat completions, file upload, PixVerse async jobs with polling, and a plain download of the results |
| `static/index.html` | The page: brief, pipeline, storyboard cards with a reshoot each, the final cut and the cost table |

Each film is a folder, `out/<date>-<time>-<premise>/`. It holds `plan.json` (the settings and Claude's shot list), `shot1…N.mp4` and `film.mp4`. The edit normalizes every shot to one frame size, because image-to-video can come back 1280×704 rather than 1280×720. It also fills in silence for any shot without sound, and draws the title and an *AI-generated* label.

## Worth knowing

- **Why the render step doesn't use Eden AI's MCP server:** its `video_generation` tool takes no `provider_params`, so it can't switch on PixVerse's audio or multi-shot mode. That's why this demo calls the REST API directly.
- **PixVerse options on Eden AI:** Eden AI passes some PixVerse options through `provider_params`: `generate_audio_switch`, `generate_multi_clip_switch`, `aspect_ratio`, `quality`, `motion_mode`, `negative_prompt` and `template_id`. It refuses `camera_movement`, `style` and `last_frame_image`, so camera moves are written into the prompt. For V6 audio, use `generate_audio_switch`: `sound_effect_switch` is refused, because it only applies to V5 and earlier.
- **PixVerse R2**, the real-time model, isn't on Eden AI yet.
- **Costs are per second of video:** about $0.048 per second at 720p with sound on V6.
