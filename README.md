# llmtape

**Record and replay LLM API calls in tests — zero cost, zero flakiness.**

```python
import llmtape

@llmtape.tape
def answer(question: str) -> str:
    return openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )
```

Record once. Replay forever. No API calls in CI.

---

## The problem

Every test run of an LLM app costs money, hits rate limits, takes 2–10s per call, and
produces non-deterministic output that makes assertions fragile. Hand-written mocks "fix"
this but diverge from reality the moment a prompt changes.

## How it works

`@llmtape.tape` intercepts calls to `openai.chat.completions.create` and
`anthropic.messages.create` made inside the decorated function, keyed by a hash of the
normalized request (model, messages, temperature, tools).

**Three modes via `LLMTAPE_MODE`:**

| Mode | Behavior |
|---|---|
| `replay` (default) | Return saved cassette. Raise `CassetteNotFoundError` on miss. |
| `record` | Call through, save cassette. |
| `record-missing` | Replay if cassette exists, record if not. Use this day-to-day. |

---

## Install

```bash
pip install llmtape
```

## Quickstart

```python
import os
import openai
import llmtape

client = openai.OpenAI()

@llmtape.tape
def classify(text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Classify as positive or negative: {text}"}],
    )
    return resp.choices[0].message.content

# First run: LLMTAPE_MODE=record python script.py
# Subsequent runs (CI): python script.py
result = classify("This movie was fantastic!")
```

Cassette saved to `.cassettes/classify_<fingerprint>.yaml`.

---

## Cassette format

```yaml
cassette_version: 1
provider: openai
request:
  fingerprint: sha256:7f5924ae...
  normalized:
    messages:
      - content: |
          Classify as positive or negative: This movie was fantastic!
        role: user
    model: gpt-4o-mini
response:
  raw:
    choices:
      - finish_reason: stop
        index: 0
        message:
          content: positive
          role: assistant
          tool_calls: null
    model: gpt-4o-mini
    usage:
      completion_tokens: 1
      prompt_tokens: 22
      total_tokens: 23
metadata:
  function_name: classify
  latency_ms: 412
  recorded_at: '2026-04-28T14:22:11+00:00'
  sdk_version: openai==1.55.0
```

Cassettes are human-readable, diffable in PRs, and safe to commit — **review them before
committing** since they contain your prompts and responses (which may include system
prompts, proprietary instructions, or user data).

---

## Async support

Works identically with `async def`:

```python
@llmtape.tape
async def answer_async(question: str):
    return await async_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )
```

---

## Anthropic support

```python
import anthropic
import llmtape

client = anthropic.Anthropic()

@llmtape.tape
def summarize(text: str):
    return client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=256,
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    )
```

---

## Tool calls and structured outputs

Full provider response objects are stored verbatim — tool calls, `finish_reason`,
`logprobs`, structured outputs all round-trip correctly through the cassette.

---

## Configuration

`pyproject.toml`:
```toml
[tool.llmtape]
cassette_dir = ".cassettes"   # where cassettes are stored
mode = "replay"               # default mode (overridden by LLMTAPE_MODE env var)
max_age_days = 30             # for llmtape check
redact = ["authorization", "api_key", "x-api-key"]  # keys to redact before saving
```

Or `.llmtape.toml` with the same keys (without the `[tool.]` wrapper).

---

## CLI

```
llmtape list                  # list all cassettes: age, function, model, tokens
llmtape show <name>           # pretty-print a cassette
llmtape check                 # flag cassettes older than max_age_days
llmtape delete <pattern>      # delete cassettes by name or glob
```

---

## GitHub Actions

```yaml
- name: Run tests
  run: pytest
  # No LLMTAPE_MODE set → defaults to replay
  # Cassettes are committed to the repo
  # Zero network calls, zero cost, runs in under 1s
```

---

## Streaming

`stream=True` is not supported in v0. LLMTape raises `TapeStreamingError` if a taped
function makes a streaming call. Use `vcrpy` for HTTP-level recording of streaming
responses, or remove `stream=True` from calls inside taped functions.

---

## Compared to vcrpy

`vcrpy` records at the HTTP level — cassettes are HTTP wire format (headers, status codes,
raw bytes). It works against any HTTP client but cassettes are hard to read for LLM content.

LLMTape records at the function level — cassettes are structured YAML with readable prompts
and responses. It's provider-agnostic (any client, any transport) and cassettes are
designed to be reviewed in pull requests as LLM content, not as HTTP traffic.

Use `vcrpy` if you need HTTP-level recording or streaming support.
Use LLMTape if you want cassettes that are readable, reviewable, and structured as LLM calls.
