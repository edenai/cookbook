# Ask the web, out loud

**Speak a question. It's searched on the live web, answered with sources, and read back to you.**

```
you speak → Gradium transcribes → Nebius LLM decides to search → Tavily searches the live web
          → Nebius writes a short cited answer (on screen) → Gradium reads it aloud
```

| At a glance | |
|---|---|
| Providers | Gradium (speech-to-text, text-to-speech), Tavily (web search), Nebius (LLM with tool calling) |
| One question | Answer on screen in about 5 s typed, or 10–14 s spoken; audio at about 17–27 s; about $0.04 |
| Runs as | A local web page (FastAPI plus one HTML file, no build step) |
| Eden AI endpoints | `/v3/upload`, `/v3/universal-ai` (+ async), `/v3/chat/completions` |

## Why these providers

| Step | Provider, via Eden AI | Why it's a great fit here |
|---|---|---|
| Speech-to-text | **Gradium** `audio/speech_to_text_async/gradium` | Voice is all Gradium does: it's the Paris voice company built by the team behind the [Kyutai](https://kyutai.org) research lab. It transcribed our browser recording of the stage question word for word in about 6 s, for $0.0005, and Eden AI lists it as EU-hosted. |
| Web search | **Tavily** `web/search/tavily` | A search API built for AI agents. It returns cleaned, ranked page content rather than a list of links, so the LLM can ground and cite its answer straight away. One search took 0.4–1.5 s for $0.008, and with today's date in the prompt it returned this month's news. |
| LLM | **Nebius** `nebius/Qwen/Qwen3-235B-A22B-Instruct-2507` | Nebius Token Factory serves open-weight models behind an OpenAI-compatible API, with the tool calling this one-tool agent loop depends on. Each LLM turn took 0.6–5 s for under $0.001. `nebius/openai/gpt-oss-120b` is one dropdown away. |
| Text-to-speech | **Gradium** `audio/tts/gradium` | The same voice provider on both ends, with a natural reading voice, also EU-hosted on Eden AI. Gradium reports 155 ms to first audio (P50) on the independent Coval benchmark with its streaming API. This demo uses the simpler, non-streaming call, so a whole 80-word answer takes 10–12 s. |

**Why Eden AI ties it together:**
- **One key:** four capabilities from three providers, on one bill.
- **Live swaps:** every provider can be changed from a dropdown during the talk (Linkup or Firecrawl for search, ElevenLabs or Deepgram for voice, Mistral for the LLM).
- **Fallbacks:** search and LLM calls pass `fallbacks`, so one provider hiccup doesn't end the demo.
- **Visible cost:** every call's time and cost appears in the Stack panel.

## Run it

```bash
cd voice-web-research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # then paste your Eden AI key into .env
python make_sample.py             # optional: regenerates samples/question.wav, the "Use sample" question
uvicorn app:app --reload          # open http://localhost:8000
```

On startup the terminal prints the base URL and every model string the app will use. The app checks each one against Eden AI's live catalog and skips any that are missing.

## Demo script (2 minutes)

1. **Ask.** Click the mic, ask "What are the biggest AI announcements this week?", and click again. If the room is noisy, use **Use sample** or type the question.
2. **Watch the pipeline.** *Listen* (Gradium), *Search* (Tavily), *Answer* (Nebius) and *Speak* (Gradium) light up in turn, each with its time. The text answer and source cards appear first, and the voice follows a few seconds later.
3. **Show the Stack panel.** Each row is one call: upload, speech-to-text, LLM, web search, LLM, text-to-speech. Each has its provider, latency and cost, and they add up to one total on one bill.
4. **Swap live.** Change *Web search* to linkup, *Text to speech* to deepgram, or *LLM* to mistral, then ask again. The pipeline labels follow, with no code change and no restart.

## How it works

| File | What it does |
|---|---|
| `edenai.py` | The Eden AI calls: upload, speech-to-text (async job plus polling), web search, chat completions with tools, text-to-speech, and a plain download of the audio |
| `app.py` | `POST /api/ask` (question → answer and sources) and `POST /api/speak` (answer → audio). Each records its steps' provider, latency and cost. Also `GET /api/providers` |
| `static/index.html` | The whole UI: mic, text box, live pipeline, answer, sources, Stack panel |
| `make_sample.py` | Makes the backup audio question with Eden AI TTS |

The LLM gets one tool, `web_search`, and today's date:
- **Grounded answers:** its first turn is forced to search (`tool_choice: "required"`).
- **Two searches at most** per question (`MAX_SEARCHES`), because models happily fire ten at once. After that it must answer.
- **Text before voice:** text-to-speech is a separate request, so the answer is already on screen while its audio is made. If text-to-speech fails, the answer stays up, with a warning.

`BRIEF.md` is the original demo brief, and `ARTICLE.md` is a short write-up of the demo.

## Cost and timing

Measured with real calls on 2026-09-28:

| | Typed question | Spoken question |
|---|---|---|
| Answer on screen | about 5 s | about 10–14 s (speech-to-text takes about 6 s) |
| Audio ready (Gradium) | about 17 s | about 21–27 s |
| Cost | about $0.04 | about $0.04 |

Most of the cost is Gradium text-to-speech (about $0.03 for an 80-word answer), then one Tavily search ($0.008). Speech-to-text and the LLM cost well under a cent. Gradium's time grows with answer length: about 4 s for one sentence, 10–12 s for 80 words. Deepgram `aura-2` costs about half as much per character.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `HTTP 401` in the UI | `EDENAI_API_KEY` is missing or wrong in `.env` |
| The mic does nothing | The browser needs mic permission, and it only works on `localhost` or HTTPS |
| "No sample yet" | Run `python make_sample.py` |
| `text-to-speech failed` | The answer still shows. Switch the TTS dropdown to another provider |
| "Press ▶ to hear the answer" | The browser blocked autoplay. Click anywhere on the page once before the demo |
| A model is skipped at startup | It's no longer in the Eden AI catalog. Edit `CHOICES` in `app.py` |

For EU data residency, set `EDENAI_BASE_URL=https://api.eu.edenai.run/v3` in `.env`. The EU catalog lists only EU-eligible models, so the startup check drops the others from the dropdowns. On 2026-09-28 that left linkup for search, gradium for TTS and `mistral/mistral-large-latest` as the LLM. Anything non-EU sent anyway is refused with HTTP 451.

## What real testing changed

These all passed against a mock and only broke on real calls:
- **Gradium text-to-speech has no `mp3` output**, only `wav`, `opus` or `pcm`. The app asks Gradium for `wav` and other providers for `mp3`.
- **Gradium's WAV is a streaming file** whose header lengths are `0xFFFFFFFF`, served as `binary/octet-stream`. The backend downloads it, writes the real lengths, and hands the browser a `data:audio/wav` URL, which also avoids expiring CDN links.
- **Browser recordings are webm**, which Eden AI's upload detects as `video/webm` and speech-to-text refuses. The app sends the browser's own `audio/webm` content type with the upload.
- **OpenAI text-to-speech (`tts-1`, `gpt-4o-mini-tts`) failed on the test account** with a provider-side 401, so the third voice is Deepgram `aura-2`.
