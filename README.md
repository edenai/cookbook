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

## Coming soon

Recipes on the to-do list — same self-contained, swap-in-one-line philosophy. PRs welcome.

### 🔍 RAG over your PDFs
Upload a document → OCR/extract → chunk and embed → semantic search → LLM answers with citations. Compares 3 embedding providers (OpenAI, Cohere, Mistral) side-by-side to show how retrieval quality changes with one config swap.

**Endpoints:** `/v3/upload`, `/v3/ocr/async` (or equivalent), `/v3/llm/embeddings`, `/v3/llm/chat/completions`. **Pattern:** classic RAG, but every step is provider-swappable.

### 🖼️ Vision Arena
Upload an image → same question fired at 4 vision LLMs (GPT-4o, Claude Sonnet, Gemini, Pixtral) → compare descriptions + structured object detection side-by-side. Same arena UI as the LLM one, but with images as input and JSON-mode outputs for apples-to-apples comparison.

**Endpoints:** `/v3/llm/chat/completions` with image content blocks. **Pattern:** arena, extended to multimodal.

### 🌍 Real-time conversation translator
Two-person live interpretation: person A speaks French → transcribed → translated → spoken in English to person B → person B replies in English → translated back to French. UN-interpreter style. Builds on the voice agent pipeline but with a translation step and a language-detection front-end.

**Endpoints:** `/v3/universal-ai/async` (STT), `/v3/translation`, `/v3/universal-ai` (TTS). **Pattern:** bidirectional continuous pipeline.

## Prerequisites

- Python 3.10+
- An [Eden AI account](https://app.edenai.run) and API key
- A browser-based Jupyter frontend (Classic Notebook or JupyterLab — VS Code's Jupyter extension does not expose mic permissions reliably; required for the voice notebook only)
- A working microphone (voice notebook only)

## Setup

```bash
git clone <this-repo>
cd edenai-cookbook

# 1. Create a virtual environment + register it as a Jupyter kernel
python -m venv .venv
.venv\Scripts\python.exe -m pip install ipykernel jupyterlab requests ipywebrtc ipywidgets python-dotenv aiohttp nest_asyncio
.venv\Scripts\python.exe -m ipykernel install --user --name edenai-cookbook --display-name "Python (edenai-cookbook)"

# 2. Drop your key in a .env file
echo EDENAI_API_KEY="your-key-here" > .env

# 3. Launch JupyterLab and pick the "Python (edenai-cookbook)" kernel
.venv\Scripts\jupyter.exe lab
```

> **Don't use the XPython kernel.** It treats `--quiet` as a package name and breaks `%pip install`. The kernel registered above is `ipykernel`, which works.

The notebooks' first cell still runs `%pip install` as a safety net, but if you ran the setup above everything is already installed. The config cell calls `load_dotenv(override=True)`, so the `.env` file is the source of truth even if a stale `EDENAI_API_KEY` exists in your shell.

> **Sandbox vs production keys.** Eden AI issues *sandbox* tokens (free, mocked responses — every model returns the same fake text) and *production* tokens (real provider calls, billed). The cookbook works with both, but the arena comparison and the voice pipeline only show real differences with a production key.

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
