# Eden AI Cookbook

Runnable demos built on [Eden AI](https://www.edenai.co): one API key and one bill for 500+ models from many providers. Each demo is a self-contained folder. Copy it, add your key, and run it.

| Demo | What it shows | Providers | Eden AI endpoints |
|---|---|---|---|
| [voice-web-research](voice-web-research/) | Ask a question out loud and get a cited, spoken answer from the live web | Gradium (speech-to-text, text-to-speech), Nebius (LLM), Tavily (search) | `/v3/upload`, `/v3/universal-ai` (+ async), `/v3/chat/completions` |
| [directors-cut](directors-cut/) | Describe a film: an LLM director writes it, Gemini draws a storyboard, the director checks it, PixVerse animates every shot with native audio, and you can reshoot any shot | Claude, GPT-6, Kimi K3, GLM 5.3 or DeepSeek V4 Pro (director), Gemini (storyboard), PixVerse (video) | `/v3/chat/completions` (structured output, image input), `/v3/universal-ai` (+ async) |
| [news-to-video](news-to-video/) | Turns a topic into a narrated 9:16 news clip built from today's web, in 30–45 seconds | Tavily (search), Nebius (LLM), Pruna (video), Gradium (voice) | `/v3/universal-ai` (+ async), `/v3/chat/completions` (structured output) |
| [upstream-watch](upstream-watch/) | A daily GitHub Action where the web tells a repo its dependencies are stale: it opens cited bump PRs, plus issues for breaking changes | Tavily (search), Nebius (LLM) | `/v3/universal-ai`, `/v3/chat/completions` (structured output) |

## Providers in this cookbook

Each demo picks the provider that's best at its step. Each demo's README has a **Why these providers** section with the numbers we measured.

| Provider | What it does here | Why it's a great fit | Demos |
|---|---|---|---|
| **Tavily** | Web search | A search API built for AI agents. It returns cleaned, ranked page content that an LLM can cite straight away, in about a second, for $0.008 per search | voice-web-research, news-to-video, upstream-watch |
| **Nebius** | LLMs: Qwen3-235B, gpt-oss-120b | Nebius Token Factory serves open-weight models behind an OpenAI-compatible API, with reliable tool calling and strict JSON-schema output, for under $0.001 per call in these demos | voice-web-research, news-to-video, upstream-watch |
| **Gradium** | Speech-to-text and text-to-speech | A voice specialist built by the team behind the Kyutai research lab, listed as EU-hosted on Eden AI. It gives accurate transcripts of browser recordings and a natural reading voice | voice-web-research, news-to-video |
| **Claude** | Director (the default of eight): writes the film, then reviews its storyboard | Strict JSON-schema output turns a premise into a plan of looks, light and one-action shots, and image input lets it check every drawn frame before it is animated. GPT-6, Kimi K3, GLM 5.3 and DeepSeek V4 Pro can direct too, for comparison | directors-cut |
| **Gemini** | Storyboard: character sheets and the first frame of every shot | Gemini 2.5 Flash Image draws from reference images, so a character looks the same in every shot, and a frame costs $0.039, much less than re-rendering video | directors-cut |
| **PixVerse** | Video generation with native audio | V6 animates each storyboard frame with its own sound and dialogue, follows camera language, and can cut a multi-shot sequence in one generation. C1 covers cinematic, action-heavy scenes | directors-cut |
| **Pruna** | Video generation | Built for speed. A 5 s vertical clip with its own soundtrack costs $0.10, so three render in parallel within a half-minute pipeline. Pruna reports that its newest model, P-Video-2-Pro, ranks #2 overall on the Design Arena video leaderboard | news-to-video |

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
