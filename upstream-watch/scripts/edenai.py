"""The two Eden AI calls Upstream Watch makes: web search and chat completions. One key: EDENAI_API_KEY."""
import os

import httpx

try:  # local runs can use a .env file; in GitHub Actions the key comes from a repo secret
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE = os.getenv("EDENAI_BASE_URL", "https://api.edenai.run/v3")
TAGS = {"repo": os.getenv("GITHUB_REPOSITORY", "local")}  # per-repo cost in the Eden AI dashboard
client = httpx.Client(base_url=BASE, timeout=90,
                      headers={"Authorization": f"Bearer {os.getenv('EDENAI_API_KEY') or 'missing-key'}"})


def call(path: str, body: dict) -> dict:
    r = client.post(path, json={**body, "tags": TAGS})
    if r.status_code >= 400:
        raise RuntimeError(f"Eden AI {path}: HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    if data.get("status") == "fail":  # provider failures come back as HTTP 200
        raise RuntimeError(f"Eden AI {path}: {data.get('provider')} failed: {data.get('error')}")
    return data


def search(query: str) -> dict:
    return call("/universal-ai", {"model": "web/search/tavily", "fallbacks": ["web/search/linkup"],
                                  "input": {"query": query, "max_results": 5}})


def chat(messages: list, model: str, fallbacks: list, schema: dict | None = None) -> dict:
    body = {"model": model, "fallbacks": fallbacks, "messages": messages, "temperature": 0}
    if schema:  # structured output: the reply is JSON that matches the schema
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": schema}}
    return call("/chat/completions", body)
