# LLMTape — Statement of Work (rev 2, post-Opus review)

## Problem

Every LLM-backed application faces the same test tax: each test run makes live API calls.
Money burned, rate limits hit, 2–10s latency per call, non-deterministic output that makes
assertions fragile. Developers work around this with hand-written mocks — fake responses
that diverge from reality the moment a prompt changes.

The HTTP world solved this a decade ago with VCR/cassette libraries. `vcrpy` works today
against OpenAI HTTP calls, but cassettes are HTTP-shaped and unreadable for LLM payloads.
No library exists that records at the *function* level with cassettes optimized for
human review of LLM content.

**LLMTape's wedge:**
- Provider-agnostic function-level interception (works with any client, any transport)
- Cassettes that are readable, diffable, and reviewable in PRs as LLM content — not HTTP wire format
- Tooling around staleness and re-recording

## Solution

```python
import llmtape

@llmtape.tape
def answer(question: str) -> str:
    return openai_call(question)

@llmtape.tape
async def answer_async(question: str) -> str:
    return await openai_async_call(question)
```

Three modes via `LLMTAPE_MODE` env var:
- `record` — calls through, saves full provider response to cassette
- `replay` (default) — intercepts, returns saved response, no network
- `record-missing` — replay if cassette exists, record if it doesn't (the safe daily-driver mode)

---

## Scope — MVP

### Core decorator
- `@llmtape.tape` works on both `def` and `async def`
- Extracts and normalizes the LLM request (model, messages, temperature, max_tokens, tools)
  from function *return value context* — not function arguments (args may include client
  objects, loggers, etc. that break hashing)
- Hashing is on the **normalized request dict** sent to the LLM, not the Python function args
- Streaming (`stream=True`) is explicitly unsupported in v0 — raises `TapeStreamingError`
  with a clear message; streaming support is Week 2

### Cassette format

```yaml
cassette_version: 1
provider: openai          # detected from response object type
request:
  fingerprint: sha256:<hash>
  normalized:
    model: gpt-4o
    messages:
      - role: user
        content: |
          What is the capital of France?
    temperature: 0.7
response:
  raw:                    # full SDK response dict, verbatim
    id: chatcmpl-abc123
    object: chat.completion
    model: gpt-4o
    choices:
      - index: 0
        message:
          role: assistant
          content: The capital of France is Paris.
          tool_calls: null
        finish_reason: stop
    usage:
      prompt_tokens: 14
      completion_tokens: 9
      total_tokens: 23
metadata:
  recorded_at: 2026-04-28T14:22:11Z
  sdk_version: openai==1.55.0
  latency_ms: 340
  function_name: answer
```

Key decisions baked in:
- **`cassette_version: 1`** — schema versioning on day one
- **`response.raw`** is the full provider response dict — handles tool calls, structured
  outputs, logprobs, refusals, finish_reason without any special casing
- **YAML literal block scalars** (`|`) used automatically for strings > 80 chars — keeps
  multi-line prompts readable
- **`metadata.sdk_version`** recorded for rot detection
- **`provider`** field: `openai` or `anthropic`, detected from response type

### Cassette location
- Single project-level `.cassettes/` directory (cwd at import time) in MVP
- Configurable via `cassette_dir` in config
- Filename: `<function_name>_<fingerprint[:12]>.yaml`
- Pytest-adjacent cassettes (per-test-file) is Week 2 via the pytest plugin

### Cassette matching
- SHA-256 of normalized request dict (JSON-serialized, keys sorted, whitespace stripped)
- Normalization strips: None values, default parameters (temperature=1.0 if that's the default), ordering of messages is preserved
- On hash miss in replay mode: raises `CassetteNotFoundError` — error message includes
  expected filename, normalized request dict, and hint to run with `LLMTAPE_MODE=record-missing`

### Redaction
```toml
[llmtape]
redact = ["authorization", "api_key", "x-api-key", "openai-organization"]
```
- Applied to request dict before hashing and before saving
- Replaced with `"[REDACTED]"` in cassette
- README prominently warns: review cassettes before committing — system prompts,
  user messages, and metadata may contain PII or proprietary instructions

### Request extraction (`_extract.py`)
- Extracts normalized request from the **arguments passed to the underlying API call**
  by wrapping at the decorator level via inspection of the wrapped function's call to
  the provider SDK
- Alternatively (simpler for MVP): wraps the provider client methods directly when the
  function is decorated — i.e., temporarily monkey-patches `openai.chat.completions.create`
  for the duration of the decorated call, captures args, then restores
- Supports: `openai.chat.completions.create`, `anthropic.messages.create`,
  async variants of both
- Unsupported providers raise `TapeUnsupportedProviderError` with a clear message

### Configuration
Via `pyproject.toml` `[tool.llmtape]` section or `.llmtape.toml`:
```toml
[llmtape]
cassette_dir = ".cassettes"
mode = "replay"
max_age_days = 30
redact = ["authorization", "api_key"]
```
Priority: env var > config file > defaults

### CLI — `llmtape`
| Command | Description |
|---|---|
| `llmtape list` | List cassettes: name, age, function, provider, token counts |
| `llmtape show <name>` | Pretty-print a cassette (content readable, not raw YAML) |
| `llmtape check` | Flag cassettes older than `max_age_days` |
| `llmtape delete <name>` | Remove cassette by name or glob pattern |

### Tests
- `pytest` suite, no live API calls in CI
- Test suite dogfoods the library — all fixtures are `.cassettes/` files
- Covers: sync record, async record, sync replay, async replay, cassette miss,
  record-missing mode, hash stability, redaction, tool call round-trip, YAML block scalar output

---

## Scope — Week 2

### Streaming support
- `stream=True` calls: record the assembled final string, replay as a sync iterator
  of one chunk (transparent to most callers)
- Optional: `replay_chunks=true` to replay the original chunk sequence

### Fuzzy matching
- Sentence-embedding similarity on cache miss
- **Default: OFF** — fuzzy matches are logged as warnings, not silently used
- Use case: local iteration only, not CI

### Re-record workflow
- `llmtape rerecord <cassette>` — replays live, diffs new vs old before saving
- Shows: token delta, content diff, latency change
- Confirmation prompt before overwrite

### Pytest plugin
- `llmtape` fixture automatically sets `mode=replay` for the test scope
- Per-test cassette directories: `.cassettes/<test_module>/`
- `@pytest.mark.llmtape_record` to force record mode for one test

---

## What LLMTape Is NOT
- Not a proxy — no HTTP interception
- Not a mock generator — cassettes are real recordings
- Not an eval tool (that's llmgate)
- Not a production cache
- Not a replacement for vcrpy if you want HTTP-level recording

---

## Stack
| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Serialization | PyYAML (with literal block scalar post-processing) |
| Hashing | hashlib stdlib |
| CLI | Click + Rich |
| Build | hatchling + uv |
| Tests | pytest |

## File Layout

```
llmtape/
├── src/
│   └── llmtape/
│       ├── __init__.py        # tape decorator, public API, mode enum
│       ├── _cassette.py       # load/save/hash/redact cassette files
│       ├── _config.py         # config resolution (env > toml > defaults)
│       ├── _extract.py        # provider monkey-patch interceptor
│       ├── _errors.py         # CassetteNotFoundError, TapeStreamingError, etc.
│       ├── cli.py             # Click CLI
│       └── __main__.py
├── tests/
│   ├── .cassettes/
│   └── test_core.py
├── pyproject.toml
└── README.md
```

## Success Criteria
- `pip install llmtape` + two lines of code = zero-cost deterministic test replay
- Works with sync and async OpenAI and Anthropic SDK calls out of the box
- Tool calls and structured outputs round-trip correctly through cassette
- Cassettes are readable enough for a PR reviewer to understand what the LLM said
- `CassetteNotFoundError` tells you exactly what to do next
- README competitive section is honest about vcrpy and where LLMTape actually differs
