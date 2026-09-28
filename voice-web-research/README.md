# Ask the web, out loud

Ask a question out loud. **Gradium** transcribes it, a **Nebius** LLM searches the web with **Tavily**, and **Gradium** reads the cited answer back. Every call goes through Eden AI with one API key.

## Run it

```bash
cd voice-web-research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # then paste your Eden AI key into .env
python make_sample.py             # once: creates samples/question.mp3 for the "Use sample" button
uvicorn app:app --reload          # open http://localhost:8000
```

On startup the terminal prints the base URL and every model string the app will use. The app checks each one against Eden AI's live catalog and skips any that are missing.

## Demo script (2 minutes)

1. **Ask.** Click the mic, ask "What are the biggest AI announcements this week?", and click again. If the room is noisy, use **Use sample** or type the question.
2. **Show the answer.** It's read aloud, with the source cards underneath.
3. **Show the Stack panel.** Point at the rows: upload, speech-to-text (Gradium), LLM (Nebius), web search (Tavily), LLM, text-to-speech (Gradium). Each row has its latency and cost, with one total. That's five providers, one key and one bill.
4. **Swap live.** Change *Web search* to linkup, *TTS* to elevenlabs or *LLM* to mistral, then ask again. No code changes and no restart.

## How it works

| File | What it does |
|---|---|
| `edenai.py` | The five Eden AI calls: upload, speech-to-text (async job + polling), web search, chat completions with tools, text-to-speech |
| `app.py` | `POST /api/ask`, which runs the pipeline and records each step's provider, latency and cost. Also `GET /api/providers` |
| `static/index.html` | The whole UI: mic, text box, answer, sources, Stack panel |
| `make_sample.py` | Makes the backup audio question with Eden AI TTS |

`BRIEF.md` is the original demo brief, and `ARTICLE.md` is a short write-up of the demo.

The LLM gets one tool, `web_search`. Its first turn is forced to search (`tool_choice: "required"`), so answers are always grounded, and its fourth turn is forced to answer. Search and LLM calls pass Eden AI `fallbacks`, so one provider hiccup doesn't end the demo. If text-to-speech fails, the answer still shows, with a warning.

## Cost

About **$0.05 per question** at the 2026-09-28 catalog prices. Text-to-speech is most of it (about $0.04 for a 120-word answer), then search (about $0.008). Speech-to-text and the LLM are under $0.001 each. The Stack panel shows the real figure for every run.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `HTTP 401` in the UI | `EDENAI_API_KEY` is missing or wrong in `.env` |
| The mic does nothing | The browser needs mic permission, and it only works on `localhost` or HTTPS |
| "No sample yet" | Run `python make_sample.py` |
| `text-to-speech failed` | The answer still shows. Switch the TTS dropdown to another provider |
| A model is skipped at startup | It's no longer in the Eden AI catalog. Edit `CHOICES` in `app.py` |

For EU data residency, set `EDENAI_BASE_URL=https://api.eu.edenai.run/v3` in `.env`. The EU catalog lists only EU-eligible models, so the startup check drops the others from the dropdowns. On 2026-09-28 that left linkup for search, gradium for TTS and `mistral/mistral-large-latest` as the LLM. Anything non-EU sent anyway is refused with HTTP 451.
