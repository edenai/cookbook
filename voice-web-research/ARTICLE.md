# Ask the web, out loud: a voice research agent in about 300 lines, with one API key

Voice agents make good demos, and they're painful to build. You need speech-to-text, an LLM that can call tools, a search API and text-to-speech. Those usually come from different vendors, each with its own SDK, key and bill, so most of the effort goes into the plumbing, not the idea.

For this demo we put all of it behind Eden AI. The result is *Ask the web, out loud*. You speak a question, and the app transcribes it with **Gradium**. A **Nebius**-hosted LLM searches the web with **Tavily**, and Gradium reads the cited answer back. The backend is about 200 lines of Python, the UI is one HTML file, and there's one environment variable: `EDENAI_API_KEY`.

## How it works

1. **The browser records** the question with `MediaRecorder` and posts it to the backend.
2. **Speech to text.** The backend uploads the audio to Eden AI (`POST /v3/upload`) and starts an async Gradium job (`audio/speech_to_text_async/gradium`). It polls until the transcript is ready.
3. **LLM plus search.** The transcript goes to `POST /v3/chat/completions`, Eden AI's OpenAI-compatible endpoint, running `nebius/Qwen/Qwen3-235B-A22B-Instruct-2507` with one tool, `web_search`. When the model calls it, the backend runs `web/search/tavily` through Eden AI's Universal AI endpoint and returns the results to the model.
4. **Text to speech.** The final answer, minus its source list, goes to `audio/tts/gradium`. The backend fetches the audio and hands it to the page, which plays it.

The page shows a **Stack** panel with one row per call: the provider, the model, the latency and the cost, plus a total. That panel is the point of the demo. Gradium, Nebius and Tavily each serve part of the answer, each call returns its own `cost`, and the total is one line on one bill.

## Why Gradium, Tavily and Nebius

Each provider is here because of what it's best at in this loop:

- **Gradium for both ends of the voice.** Voice is all Gradium does. It's the Paris company built by the team behind the Kyutai research lab, and it reports top latency results on the independent Coval voice benchmarks. In our runs it transcribed a browser recording word for word in about 6 seconds and read an 80-word answer in a natural voice for about three cents. Eden AI lists both of those Gradium models as EU-hosted.
- **Tavily for the web.** Tavily is a search API built for AI agents. It returns cleaned, ranked page content rather than links, which is exactly what an LLM needs in order to cite. A search took about a second and cost $0.008.
- **Nebius for the reasoning.** Nebius Token Factory serves open-weight models, here Qwen3-235B, behind an OpenAI-compatible API with reliable tool calling. That's the one feature this agent loop can't do without. Each turn took one to five seconds and cost under a tenth of a cent.

## One request shape for everything that isn't an LLM

Speech-to-text, search and text-to-speech come from two vendors here, but on Eden AI they all use the same call:

```python
await call("POST", "/universal-ai", json={
    "model": "web/search/tavily",            # feature/subfeature/provider
    "fallbacks": ["web/search/linkup"],      # tried in order if Tavily fails
    "input": {"query": query, "max_results": 6},
})
```

Swap `web/search/tavily` for `web/search/linkup`, or `audio/tts/gradium` for `audio/tts/elevenlabs`, and nothing else changes: the input and output shapes are normalized per feature. That's why the demo can offer dropdowns that swap providers **live, mid-talk**, without touching code.

The LLM side is plain OpenAI format. The same `messages`, `tools` and `tool_calls` code works whichever provider serves the model, with a `fallbacks` list in case one is down.

## Four things that make a stage demo reliable

These are the lessons worth taking away, because each one is a real way a live demo falls over.

- **Check `status`, not just the HTTP code.** When a provider fails on Universal AI, you still get HTTP 200, with `"status": "fail"`. The client raises on that, so a failed step shows up as a clear error rather than a blank answer.
- **Force the first search.** Models sometimes answer from memory, or narrate "I'll search for that…" without calling the tool. `tool_choice: "required"` on the first turn guarantees a grounded answer, and `"none"` on the last turn guarantees the loop ends with text.
- **Echo tool calls back minimally.** When the assistant's tool call goes back into the conversation, keep only `id`, `type` and `function`, and use `""` rather than `null` for content. Strict providers reject extra fields, such as the `index` a gateway can add.
- **Degrade, don't crash.** If text-to-speech fails, the answer still appears on screen, with a warning. A typed question and a pre-recorded sample always work, because venue audio never does.

## What only real calls caught

A mock of the API passed every test. The first real run failed in four places, and each was a one- or two-line fix:

- **Gradium's text-to-speech has no MP3 output.** It offers WAV, Opus or PCM, so the app asks Gradium for WAV and other voices for MP3.
- **Gradium streams its WAV** with "unknown length" in the header, which browser audio players can choke on. The backend writes the real lengths in before handing the audio to the page.
- **Browser recordings are WebM**, which gets detected as *video* on upload and refused by speech-to-text. Sending the browser's own `audio/webm` type fixes it.
- **The LLM fired ten searches for one question**, and the run took nearly 60 seconds. The app now allows two searches, puts today's date in the prompt so they target this week's news, and shows the text before the audio is ready. The answer is on screen in about 5 seconds.

## Check the catalog at startup

Our first brief for this demo had three wrong details: a model ID missing its version suffix, an LLM endpoint path that doesn't exist, and the wrong name for the async job ID field. Each would have failed on the first call. Model names change faster than documentation. So at startup the app checks every model string against Eden AI's public catalog (`GET /v3/info/{feature}/{subfeature}` and `GET /v3/models`, no key needed). It prints the ones it will use and drops any that have disappeared.

The same check gives EU data residency for free. Point `EDENAI_BASE_URL` at `api.eu.edenai.run`, and the catalog lists only EU-eligible models. The dropdowns follow: today that means Linkup for search, Gradium for voice and Mistral Large as the LLM.

## What it costs

Measured with real calls, one question costs about **four cents**. Most of that is Gradium text-to-speech on an 80-word answer (about $0.03), then one Tavily search ($0.008). Transcription and two LLM turns cost well under a cent. Nobody has to estimate it, because every run prints its exact cost in the Stack panel.

## Try it

```bash
cd voice-web-research && pip install -r requirements.txt
cp .env.example .env   # add your Eden AI key
python make_sample.py  # optional: regenerate the backup question
uvicorn app:app --reload
```

Open `http://localhost:8000`, click the mic, and ask something that happened this week. Then change the search provider and ask again.
