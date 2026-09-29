# Ask the web, out loud

Ask a question out loud. **Gradium** transcribes it, a **Nebius** LLM searches the web with **Tavily**, and **Gradium** reads the cited answer back. Every call goes through Eden AI with one API key.

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
2. **Show the answer.** The text and source cards appear first, and the answer is read aloud a few seconds later.
3. **Show the Stack panel.** Point at the rows: upload, speech-to-text (Gradium), LLM (Nebius), web search (Tavily), LLM, text-to-speech (Gradium). Each row has its latency and cost, with one total. That's five providers, one key and one bill.
4. **Swap live.** Change *Web search* to linkup, *TTS* to deepgram or *LLM* to mistral, then ask again. No code changes and no restart.

## How it works

| File | What it does |
|---|---|
| `edenai.py` | The five Eden AI calls: upload, speech-to-text (async job + polling), web search, chat completions with tools, text-to-speech |
| `app.py` | `POST /api/ask` (question → answer and sources) and `POST /api/speak` (answer → audio). Each records its steps' provider, latency and cost. Also `GET /api/providers` |
| `static/index.html` | The whole UI: mic, text box, answer, sources, Stack panel |
| `make_sample.py` | Makes the backup audio question with Eden AI TTS |

`BRIEF.md` is the original demo brief, and `ARTICLE.md` is a short write-up of the demo.

The LLM gets one tool, `web_search`, and today's date. Its first turn is forced to search (`tool_choice: "required"`), so answers are always grounded. It gets at most 2 searches per question (`MAX_SEARCHES`), because models happily fire ten at once, and it must answer after that. Search and LLM calls pass Eden AI `fallbacks`, so one provider hiccup doesn't end the demo. Text-to-speech is a separate request, so the answer is on screen while its audio is made. If text-to-speech fails, the answer stays up, with a warning.

## Cost and timing

Measured with real calls on 2026-09-28:

| | Typed question | Spoken question |
|---|---|---|
| Answer on screen | about 5 s | about 10–14 s (speech-to-text takes about 6 s) |
| Audio ready (Gradium) | about 17 s | about 21–27 s |
| Cost | about $0.04 | about $0.04 |

Most of the cost is Gradium text-to-speech (about $0.03 for an 80-word answer), then one Tavily search ($0.008). Speech-to-text and the LLM cost well under a cent. Gradium's time grows with answer length (about 4 s for one sentence, 10–12 s for 80 words). Deepgram `aura-2` costs about half as much per character. The Stack panel shows the real figures for every run.

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
- **Gradium's WAV is a streaming file** whose header lengths are `0xFFFFFFFF`, served as `binary/octet-stream`. The backend downloads it, writes the real lengths and hands the browser a `data:audio/wav` URL, which also avoids expiring CDN links.
- **Browser recordings are webm**, which Eden AI's upload detects as `video/webm` and speech-to-text refuses. The app sends the browser's own `audio/webm` content type with the upload.
- **OpenAI text-to-speech (`tts-1`, `gpt-4o-mini-tts`) failed on the test account** with a provider-side 401, so the third voice is Deepgram `aura-2`.
