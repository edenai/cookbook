# Eden AI Cookbook

Practical, runnable recipes that show what you can build when 200+ AI models sit behind a single API key.

Each notebook is self-contained, uses only Eden AI's V3 endpoints, and is designed so you can **swap providers in one line** to compare results. That's the whole point.

## Recipes

### [Voice-to-Voice Agent](voice_to_voice_agent.ipynb)
A push-to-talk voice assistant chaining three Eden AI features in one pipeline:

**Speech-to-Text → LLM → Text-to-Speech**

Click record, talk, release — the audio is uploaded, transcribed, sent to an LLM with rolling conversation history, and the answer is spoken back.

**Endpoints used:** `/v3/upload`, `/v3/universal-ai/async`, `/v3/llm/chat/completions`, `/v3/universal-ai`

### [Live LLM Arena](llm_arena.ipynb)
Same prompt → 4 LLMs → 4 streaming panes side-by-side → vote on the winner (or let an LLM judge pick).

Tracks first-token latency and total response time per model. Add or remove models by editing one list.

**Default lineup:** Claude, GPT, DeepSeek, GLM. **Endpoint:** `/v3/llm/chat/completions` (OpenAI-compatible, with SSE streaming).

### [Vision Arena](vision_arena.ipynb)
The LLM Arena extended to multimodal. Drop an image, ask a question, four vision LLMs stream their answers side-by-side. Same OpenAI-compatible payload — only the `model` string and the `content` block shape change.

**Endpoint:** `/v3/llm/chat/completions` with image content blocks (base64 data URI or HTTPS URL).

### [Tool-calling Agent Arena](tool_calling_agent.ipynb)
Same prompt, same four tools (calculator, weather, time, web search), four LLMs. Watch which models call the right tool, which hallucinate instead of calling, and which invent data when a tool returns "no results". The OpenAI-compatible `tools` parameter works identically across providers — the interesting part is **how** each model uses them.

**Endpoint:** `/v3/llm/chat/completions` with `tools` parameter (multi-turn agent loop).

### [Document → Structured JSON](document_to_json.ipynb)
Drop a receipt, resume, or invoice. Four vision LLMs extract a JSON object matching the same schema, side-by-side. See who follows the schema, who hallucinates fields, and who refuses to guess. Three pre-defined schemas (Receipt / Resume / Invoice), hot-swappable.

**Endpoint:** `/v3/llm/chat/completions` with image content blocks + JSON schema validation post-call.

### [RAG Arena](rag_arena.ipynb)
Classic RAG in ~150 lines: text corpus → chunk → embed → retrieve top-k → LLM answers with citations. The interesting version: **swap the embedding provider in one line and see how retrieved chunks change**. Three embedding spaces (OpenAI / Mistral / Cohere) over the same query, one LLM for the final answer — so the only variable is *which chunks got retrieved*.

**Endpoints:** `/v3/llm/embeddings`, `/v3/llm/chat/completions`.

### [Real-time Voice Translator](realtime_translator.ipynb)
Two-person live interpretation. Person A speaks → STT → translate to target language → TTS plays the translated audio. One click switches direction for the reply. Same pipeline as the Voice-to-Voice Agent, but the LLM step is a translation prompt and there's a language picker. UN-interpreter style.

**Endpoints used:** `/v3/upload`, `/v3/universal-ai/async`, `/v3/llm/chat/completions`, `/v3/universal-ai`.

### [Marketing Copy Generator](marketing_copy.ipynb)
Drop a product brief → 4 LLMs each return 3 taglines, 2 LinkedIn posts, 5 SEO title variants. Pick a tone (punchy / professional / playful), see which model gets it. The interesting compare here isn't *correctness* but **style fidelity, length compliance, and originality**.

**Endpoint:** `/v3/llm/chat/completions` with structured JSON output + post-call length/tone validation.

### [PII Anonymization Arena](pii_anonymization.ipynb)
Paste text containing personal data → four LLMs return a redacted version + a structured list of what they found. See which models catch all PII, which over-redact, and which miss obvious entities. Trust & safety baseline before routing customer data through any provider.

**Endpoint:** `/v3/llm/chat/completions` with structured JSON output (redacted text + entities list).

### [Automation Webhook](automation_webhook.ipynb)
Three webhook-shaped payloads (HubSpot lead, support ticket, Slack message) → one LLM enrichment function → enriched JSON ready to send back into your automation. This is what's happening *under the hood* when you drop an Eden AI step into n8n / Zapier / Make. Fork it into your own backend before going no-code.

**Endpoint:** `/v3/llm/chat/completions`. Pattern: pure function `enrich(payload) → enriched_payload`, drop into a Flask route, an n8n Code node, or anywhere that speaks JSON.

## Coming soon

Recipes on the to-do list — same self-contained, swap-in-one-line philosophy. PRs welcome.

### 👥 Speaker-diarized meeting transcript
Upload a meeting recording → STT with diarization → per-speaker timeline with timestamps → LLM-generated summary + action items per speaker. Deepens the audio story beyond the voice agent and translator: same `/v3/universal-ai/async` endpoint, just turning on diarization and consuming the speaker labels.

**Endpoints:** `/v3/universal-ai/async` (with diarization), `/v3/llm/chat/completions`. **Pattern:** STT → structured output → LLM summarization, all over a single async upload.

### 🔎 Multimodal Search (CLIP-style)
One shared embedding space holding both images and text. Search a photo library with a sentence ("a person riding a bike at sunset"), or search a text corpus by uploading an image. Same swap-providers point as RAG Arena, but the embeddings cross modalities — and the failure modes are completely different.

**Endpoints:** `/v3/llm/embeddings` (multimodal variants). **Pattern:** cross-modal retrieval — same shape as RAG, different geometry.

### ⚙️ Async Batch Processor
1000-row CSV → concurrent LLM calls with backpressure, retries on transient failures, progress bar, and a cost ledger that tallies tokens-in / tokens-out / $$ per row. The production-grade sibling of the Automation Webhook: same `enrich(row)` shape, but built for scale.

**Endpoint:** `/v3/llm/chat/completions` with `asyncio.gather` + semaphore concurrency limits. **Pattern:** the batch-enrichment job you'd actually ship.

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

Every notebook is designed around the moment that makes Eden AI obvious:

> "Wait — I just changed `openai/gpt-4` to `anthropic/claude-sonnet-4-5` and it still works?"

The voice agent and the translator make that point *across* feature categories (STT, LLM, TTS chained). The arenas (LLM, vision, tool-calling, RAG, copy, PII) make it *within* a category, four ways at once, in real time. The webhook and JSON extraction recipes show the same swap inside the kind of glue code you'd actually ship.

## API endpoints used

| Feature | Endpoint | Pattern |
|---|---|---|
| File upload | `POST /v3/upload` | multipart, returns `file_id` |
| Speech-to-text | `POST /v3/universal-ai/async` | async — launch + poll |
| LLM chat | `POST /v3/llm/chat/completions` | OpenAI-compatible, supports SSE, tools, image content blocks |
| Embeddings | `POST /v3/llm/embeddings` | OpenAI-compatible, used by RAG |
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
