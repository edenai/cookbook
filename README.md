# Eden AI Cookbook

Runnable demos built on [Eden AI](https://www.edenai.co): one API key and one bill for 500+ models from many providers. Each demo is a self-contained folder. Copy it, add your key, and run it.

| Demo | What it shows | Providers | Eden AI endpoints |
|---|---|---|---|
| [voice-web-research](voice-web-research/) | Ask a question out loud and get a cited, spoken answer from the live web | Gradium (speech-to-text, text-to-speech), Nebius (LLM), Tavily (search) | `/v3/upload`, `/v3/universal-ai` (+ async), `/v3/chat/completions` |
| [news-to-video](news-to-video/) | Turns a topic into a narrated 9:16 news clip built from today's web, in 30–45 seconds | Tavily (search), Nebius (LLM), Pruna (video), Gradium (voice) | `/v3/universal-ai` (+ async), `/v3/chat/completions` (structured output) |
| [upstream-watch](upstream-watch/) | A daily GitHub Action where the web tells a repo its dependencies are stale: it opens cited bump PRs, plus issues for breaking changes | Tavily (search), Nebius (LLM) | `/v3/universal-ai`, `/v3/chat/completions` (structured output) |

## How the cookbook is organized

- **One folder per demo**, named after what it does. Providers are listed in the table, not used as folders, because most demos mix several of them and let you swap them.
- **Each demo is self-contained.** It has its own `README.md` (what it shows, how to run it, a demo script, cost per run), `requirements.txt` and `.env.example`, and no code is shared between demos.
- **Model strings are checked against the live catalog** (`GET /v3/models`, `GET /v3/info/{feature}/{subfeature}`). Look them up there rather than copying them from old code.

## Adding a demo

1. Create a folder named after the use case, e.g. `invoice-extraction/`.
2. Include a `README.md`, `.env.example` (`EDENAI_API_KEY=` only) and `requirements.txt`. Never commit a real key.
3. Add a row to the table above.

## License

MIT, see [LICENSE](LICENSE).
