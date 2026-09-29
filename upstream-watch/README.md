# Upstream Watch

**The public web tells a repo that its dependencies are out of date, and the repo opens its own PRs.**

A daily GitHub Actions job reads `requirements.txt` (or `package.json`). For every pinned package, it runs one **Tavily** web search for release notes and advisories. A **Nebius** LLM classifies the result as `none`, `patch`, `security` or `breaking`:
- **Patch and security** updates become a PR that bumps the pin, with cited sources and the run cost in the body.
- **Breaking** updates become an issue with a migration plan.

Your CI validates the PR, and a human merges. Both AI calls go through Eden AI with one key.

```
Trigger (cron) → Signal (Tavily search per package) → Triage (LLM → JSON verdict)
     → patch/security: bump pin → PR → CI → human merges
     → breaking: issue + migration plan (code is never edited)
```

| At a glance | |
|---|---|
| Providers | Tavily (web search), Nebius (LLM with structured output) |
| One run | About 30 s and $0.06 for 7 packages; one PR or issue per real finding |
| Runs as | A GitHub Action on a daily schedule plus a "Run workflow" button, or a local dry run |
| Eden AI endpoints | `/v3/universal-ai`, `/v3/chat/completions` (JSON schema) |

## Why these providers

| Step | Provider, via Eden AI | Why it's a great fit here |
|---|---|---|
| Signal | **Tavily** `web/search/tavily` | It turns the public web into an event source. Tavily is a search API built for AI agents, so each package's search returns cleaned, ranked page content: release notes, changelogs and advisories the LLM can judge. On flaskr-tdd's real pins it surfaced the Flask 3.1.3 security fix (CVE-2026-27205) and the gunicorn and pytest CVEs, at $0.008 a package. |
| Triage and drafts | **Nebius** `nebius/Qwen/Qwen3-235B-A22B-Instruct-2507` | Triage has to be machine-readable every time, and this open-weight model on Nebius Token Factory returns strict JSON-schema verdicts. It also writes the migration plans for breaking issues. Each call cost about $0.0004, so the LLM is a rounding error on the bill. `nebius/openai/gpt-oss-120b` is the automatic fallback. |

**Why Eden AI ties it together:**
- **One repo secret** (`EDENAI_API_KEY`) covers both providers.
- **Fallbacks:** Tavily falls back to Linkup, and Qwen to gpt-oss, so a flaky provider doesn't turn the daily run red.
- **Per-repo cost:** every call is tagged `repo=<owner>/<repo>`, and the summed `cost` fields go into each PR and issue body and set the run's hard cost cap.

## Try it locally (dry run: prints, changes nothing)

```bash
cd upstream-watch
pip install -r requirements.txt
cp .env.example .env                   # then paste your Eden AI key
python scripts/upstream_watch.py --manifest example/requirements.txt
```

`example/requirements.txt` holds the real pins of [mjhea0/flaskr-tdd](https://github.com/mjhea0/flaskr-tdd), last bumped in October 2023, so the findings are real and nothing is staged.

## Add it to a repo

1. Fork the target repo. `mjhea0/flaskr-tdd` is small, stale and already has CI.
2. Copy `scripts/upstream_watch.py`, `scripts/edenai.py` and `.github/workflows/upstream-watch.yml` into it, keeping those paths.
3. Add the repo secret `EDENAI_API_KEY`.
4. Add the repo secret `UPSTREAM_WATCH_TOKEN`: a fine-grained token with *contents*, *pull requests* and *issues* read/write on the fork.
   - **Why it's needed:** GitHub doesn't run CI on PRs opened with the default `GITHUB_TOKEN`, and without CI the demo has no green check.
   - **Without it:** the workflow falls back to `GITHUB_TOKEN`. You then need to enable *Settings → Actions → General → Allow GitHub Actions to create and approve pull requests*.
5. **Actions → Upstream Watch → Run workflow.**

## Stage runbook (about 90 s)

1. Show the fork: `requirements.txt` with old pins and a green CI badge.
2. **Actions → Upstream Watch → Run workflow.** While it runs (about 30 s plus runner start-up), show the Eden AI dashboard filling with `web/search/tavily` and `nebius/…` calls, tagged `repo=<owner>/<repo>`.
3. The run page shows a verdict table, and PRs appear with a title, a one-line summary, cited sources and the run cost.
4. CI goes green on the PR.
5. Close with: *"Nobody asked for this PR. The web told the repo it was out of date. One key, one bill, and still a human on merge."*

As a fallback, pre-record a 60-second capture of steps 2–4.

## Guardrails

| Guardrail | How it works |
|---|---|
| No auto-merge | The script opens PRs and never merges |
| At most 3 PRs per run | `--max-prs 3`. Security first, then patches; the rest wait for the next run |
| Allowlist | `--allow flask,gunicorn`. Other packages get an issue, not a PR |
| Breaking never edits code | A new major version always becomes an issue, even if the model called it a patch |
| Hard cost cap | `--max-cost 0.10`. Packages are checked 4 at a time, and the summed `cost` is checked between batches. Once it reaches the cap, the rest wait for the next run and issues skip the LLM draft |
| Search results are data | The prompt says so. A proposed version must be newer than the pin **and** appear in the search results, and cited URLs must come from the results, or the verdict is dropped |
| No duplicates | A PR or issue title seen before, open or closed, is never opened again |
| Loud failures | If a package can't be checked (e.g. a bad key), the run goes red instead of silently green |

## How it works

| File | What it does |
|---|---|
| `scripts/upstream_watch.py` | Parses the manifest, then search → triage → check → PR or issue, using `git` and the `gh` CLI |
| `scripts/edenai.py` | The two Eden AI calls: `POST /v3/universal-ai` (`web/search/tavily`, falling back to linkup) and `POST /v3/chat/completions` (`nebius/Qwen/Qwen3-235B-A22B-Instruct-2507` with a JSON-schema response format, falling back to `nebius/openai/gpt-oss-120b`) |
| `.github/workflows/upstream-watch.yml` | Daily cron plus a manual trigger, with permissions to push, open PRs and open issues |

Every call is tagged `repo=<owner>/<repo>` for per-repo cost in the Eden AI dashboard. Model strings were checked against the live catalog (`/v3/models`, `/v3/info/web/search`) on 2026-09-28.

## Cost, timing and a real run

Each package costs one search (about $0.008 with Tavily) plus one small LLM call (well under $0.001). Each breaking issue adds one LLM call. Packages are checked 4 at a time.

Measured with real calls on 2026-09-28 against the 7 flaskr-tdd pins: **about 30 s and $0.06 per run.**

| Package | Pinned | Found | Verdict | What happens |
|---|---|---|---|---|
| Flask | 3.0.0 | 3.1.3 | security (CVE-2026-27205) | PR |
| Flask-SQLAlchemy | 3.1.1 | 3.1.2 | patch | PR |
| psycopg2-binary | 2.9.9 | 2.9.13 | patch | PR |
| gunicorn | 21.2.0 | 26.2.0 | breaking (major, also fixes request-smuggling CVEs) | issue |
| flake8 | 6.1.0 | 7.4.1 | breaking (major) | issue |
| pytest | 7.4.2 | 9.0.2 | breaking (major, also CVE-2025-71176) | issue, which names `tests/` files that import pytest |
| black | 23.10.0 | none found | none | nothing |

Results depend on what the web says on the day, so check the verdicts before a live demo. For example, black 25.x exists, but that day's search results didn't mention it, so the model correctly reported nothing newer. The exact cost is in every PR and issue body and on the run page.

## Limits (v1)

- Only the manifest is edited. Lockfiles (`package-lock.json`, `uv.lock`) are left to CI.
- No transitive dependencies, monorepos or multiple manifests.
- Only exact pins are checked: `pkg==1.2.3` in `requirements.txt`, and `1.2.3`, `^1.2.3` or `~1.2.3` in `package.json`.
- "Affected files" in a migration plan is a heuristic: files that import the package by name.
