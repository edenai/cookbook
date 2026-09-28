# Ask the web, out loud

A single-page demo for an Eden AI event. You ask a question out loud. The app transcribes it, lets an LLM search the web, and reads a cited answer back. Every AI call goes through Eden AI with one API key.

This is a stage demo, so reliability and legibility matter more than features.

## Demo flow

1. The presenter speaks a question, or types it.
2. **Gradium** transcribes it.
3. A **Nebius** LLM searches the web with **Tavily** and writes a short answer with sources.
4. **Gradium** reads the answer aloud.
5. A "Stack" panel shows each step's provider, model, latency and cost. This is the Eden AI selling point.

## Stack

- **Backend:** Python + FastAPI.
- **Frontend:** one static HTML page with vanilla JS, no build step.
- **Config:** `.env` with `EDENAI_API_KEY`. Base URL `https://api.edenai.run/v3` by default; `https://api.eu.edenai.run/v3` for EU.
- **Not needed:** a database or a queue.

## Eden AI calls

| Step | Endpoint | Model |
|---|---|---|
| Upload the recording | `POST /v3/upload` returns `file_id` | none |
| Speech to text | `POST /v3/universal-ai/async`, then poll `GET /v3/universal-ai/async/{public_id}` until `status` is `success` or `fail` | `audio/speech_to_text_async/gradium` |
| Web search (the LLM's tool) | `POST /v3/universal-ai` | `web/search/tavily` |
| LLM | `POST /v3/chat/completions` (OpenAI format, with `tools`) | `nebius/Qwen/Qwen3-235B-A22B-Instruct-2507`, fallback `nebius/openai/gpt-oss-120b` |
| Text to speech | `POST /v3/universal-ai` | `audio/tts/gradium`, which returns a playable `audio_resource_url` |

Every response includes a `cost`, which the Stack panel should show. All model strings were checked against the live catalog on 2026-09-28. Check them again at startup with `GET /v3/info/{feature}/{subfeature}` and `GET /v3/models`, and print what the app will use.

## Backend

- `POST /api/ask` takes either an audio recording or typed text. It returns the transcript, the answer, the sources, the audio URL, and each step's provider, model, latency and cost.
- `GET /api/providers` lists the options for the provider dropdowns.
- The LLM gets one tool, `web_search`, and at most a few turns. It answers in about 120 words of speakable prose, followed by numbered sources. Only the prose is spoken.

## Frontend

- A big mic button, plus a text box and a "Use sample" button (`samples/question.wav`) for when the venue audio is bad.
- The answer in large type, source cards, and audio that plays automatically.
- The Stack panel, always visible.
- Three dropdowns to swap providers live, with no reload:
  - **Search:** tavily, linkup, firecrawl.
  - **TTS:** gradium, elevenlabs, openai/tts-1.
  - **LLM:** the two Nebius models, or `mistral/mistral-large-latest`.
- A dark, high-contrast theme that's readable on a projector.

## What breaks demos

- **Failures that look like success:** when a provider fails, Universal AI still answers HTTP 200, with `"status": "fail"`. Check `status` on every step.
- **Rejected tool calls:** when sending the model's tool calls back, keep only `id`, `type` and `function`, and use `""` rather than `null` for content. Cap each search result at a few thousand characters.
- **One error sinking the demo:** a failed step should show a clear message in the UI, not crash the server. Use Eden AI `fallbacks` so one provider hiccup doesn't end the demo.
- **Bad audio:** the typed input and the sample file must always work, with no mic.

## Deliverables

- `app.py`, `edenai.py` (a thin client for the calls above), `static/index.html`, `.env.example`, `samples/question.wav`.
- A `README.md` with setup, `uvicorn app:app --reload`, the demo flow, and how to swap providers.
- About 500 lines in total, written to be read on a projector.

## Out of scope

Auth, persistence, streaming audio, multi-turn memory, deployment.
