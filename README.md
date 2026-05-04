# Eden AI Cookbook

Practical, runnable recipes that show what you can build when 200+ AI models sit behind a single API key.

Each notebook is self-contained, uses only Eden AI's V3 endpoints, and is designed so you can **swap providers in one line** to compare results. That's the whole point.

## Recipes

### [Voice-to-Voice Agent](voice_to_voice_agent.ipynb)
A push-to-talk voice assistant chaining three Eden AI features in one pipeline:

**Speech-to-Text → LLM → Text-to-Speech**

Click record, talk, release — the audio is uploaded, transcribed, sent to an LLM with rolling conversation history, and the answer is spoken back. Includes a translator variant (any language → any language) using the same pipeline.

**Endpoints used:** `/v3/upload`, `/v3/universal-ai/async`, `/v3/llm/chat/completions`, `/v3/universal-ai`

### [Live LLM Arena](llm_arena.ipynb)
Same prompt → 4 LLMs → 4 streaming panes side-by-side → vote on the winner (or let an LLM judge pick).

Tracks first-token latency and total response time per model. Add or remove models by editing one list.

**Default lineup:** Claude, GPT, DeepSeek, GLM. **Endpoint:** `/v3/llm/chat/completions` (OpenAI-compatible, with SSE streaming).

## Prerequisites

- Python 3.10+
- An [Eden AI account](https://app.edenai.run) and API key
- A browser-based Jupyter frontend (Classic Notebook or JupyterLab — VS Code's Jupyter extension does not expose mic permissions reliably; required for the voice notebook only)
- A working microphone (voice notebook only)

## Setup

```bash
git clone <this-repo>
cd edenai-cookbook
export EDENAI_API_KEY="sk-..."   # or set in your shell rc
jupyter lab
```

Each notebook's first cell installs its own dependencies via `%pip install`, so there's no shared `requirements.txt` to track.

## Why these recipes

Both notebooks are designed around the moment that makes Eden AI obvious:

> "Wait — I just changed `openai/gpt-4` to `anthropic/claude-sonnet-4-5` and it still works?"

The voice agent makes that point across three different feature categories (STT, LLM, TTS). The arena makes it within a single category, four ways at once, in real time.

## API endpoints used

| Feature | Endpoint | Pattern |
|---|---|---|
| File upload | `POST /v3/upload` | multipart, returns `file_id` |
| Speech-to-text | `POST /v3/universal-ai/async` | async — launch + poll |
| LLM chat | `POST /v3/llm/chat/completions` | OpenAI-compatible, supports SSE |
| Text-to-speech | `POST /v3/universal-ai` | sync, returns `audio_resource_url` |

Auth on every call: `Authorization: Bearer $EDENAI_API_KEY`.

## Reference

- [Eden AI V3 docs](https://docs.edenai.co)
- [Chat completions reference](https://docs.edenai.co/v3/llms/chat-completions)
- [List of supported models](https://docs.edenai.co/v3/llms/listing-models)

## Contributing

Open a PR with a new notebook in this folder. Keep recipes:

- **Self-contained** — one notebook, runnable top-to-bottom
- **Provider-swappable** — provider strings as constants at the top
- **Focused** — one concept per recipe; link to others rather than duplicating
