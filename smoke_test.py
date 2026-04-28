"""
End-to-end smoke test for llmtape.
No live API calls — uses mocked provider responses.
Tests the full record → cassette-on-disk → replay cycle.
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Point cassettes at a temp dir for this test
tmpdir = tempfile.mkdtemp(prefix="llmtape_smoke_")
os.environ["LLMTAPE_CASSETTE_DIR"] = str(Path(tmpdir) / ".cassettes")
os.environ["LLMTAPE_MODE"] = "record"

import llmtape
from llmtape._config import reset_config
from llmtape._errors import CassetteNotFoundError, TapeStreamingError

reset_config()

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  [PASS] {name}")
    except Exception as e:
        results.append((FAIL, name))
        print(f"  [FAIL] {name}: {e}")

print("\n=== LLMTape Smoke Test ===\n")

# ── Build a fake OpenAI response ─────────────────────────────────────────────
def fake_openai_response(content="The capital of France is Paris."):
    resp = MagicMock()
    resp.__class__.__module__ = "openai.types.chat"
    resp.__class__.__name__ = "ChatCompletion"
    resp.model_dump.return_value = {
        "id": "chatcmpl-smoke",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content, "tool_calls": None},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 14, "completion_tokens": 9, "total_tokens": 23},
    }
    return resp

# ── 1. Record mode: cassette written to disk ──────────────────────────────────
print("1. Record mode")

import openai.resources.chat.completions as _oa

_real_create = _oa.Completions.create

def test_record():
    call_count = [0]
    def mock_create(self, **kwargs):
        call_count[0] += 1
        return fake_openai_response()

    with patch.object(_oa.Completions, "create", mock_create):
        import openai
        client = openai.OpenAI(api_key="fake")

        @llmtape.tape
        def ask_capital(country: str):
            return client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"Capital of {country}?"}],
            )

        result = ask_capital("France")

    assert call_count[0] == 1, "Live call should have been made in record mode"
    cassettes = list(Path(os.environ["LLMTAPE_CASSETTE_DIR"]).glob("*.yaml"))
    assert len(cassettes) == 1, f"Expected 1 cassette, got {len(cassettes)}"
    print(f"     Cassette written: {cassettes[0].name}")

check("Record makes live call and writes cassette", test_record)

# ── 2. Replay mode: no live call, returns cassette ────────────────────────────
print("\n2. Replay mode")

def test_replay():
    os.environ["LLMTAPE_MODE"] = "replay"
    reset_config()

    live_calls = [0]
    def mock_create_replay(self, **kwargs):
        live_calls[0] += 1
        raise RuntimeError("Should not make live call in replay mode")

    with patch.object(_oa.Completions, "create", mock_create_replay):
        import openai
        client = openai.OpenAI(api_key="fake")

        @llmtape.tape
        def ask_capital(country: str):
            return client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"Capital of {country}?"}],
            )

        result = ask_capital("France")

    assert live_calls[0] == 0, "No live calls should occur in replay mode"
    assert result is not None

check("Replay returns cassette without live call", test_replay)

# ── 3. Replay miss raises CassetteNotFoundError ───────────────────────────────
print("\n3. CassetteNotFoundError")

def test_cassette_miss():
    os.environ["LLMTAPE_MODE"] = "replay"
    reset_config()

    import openai
    client = openai.OpenAI(api_key="fake")

    @llmtape.tape
    def ask_something_new(q: str):
        return client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": q}],
        )

    try:
        ask_something_new("a question with no cassette")
        raise AssertionError("Should have raised CassetteNotFoundError")
    except CassetteNotFoundError as e:
        assert "ask_something_new" in str(e)
        assert "LLMTAPE_MODE=record-missing" in str(e)
        print(f"     Error message preview: {str(e)[:80].strip()}...")

check("CassetteNotFoundError raised on miss with actionable message", test_cassette_miss)

# ── 4. record-missing: replay existing, record new ───────────────────────────
print("\n4. record-missing mode")

def test_record_missing():
    os.environ["LLMTAPE_MODE"] = "record-missing"
    reset_config()

    live_calls = [0]
    def mock_create_new(self, **kwargs):
        live_calls[0] += 1
        return fake_openai_response("Berlin")

    with patch.object(_oa.Completions, "create", mock_create_new):
        import openai
        client = openai.OpenAI(api_key="fake")

        @llmtape.tape
        def ask_capital(country: str):
            return client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"Capital of {country}?"}],
            )

        # France cassette already exists — should NOT make live call
        result_france = ask_capital("France")
        assert live_calls[0] == 0, "France cassette should have been replayed"

        # Germany has no cassette — should make live call and record
        result_germany = ask_capital("Germany")
        assert live_calls[0] == 1, "Germany should have triggered a live call"

    cassettes = list(Path(os.environ["LLMTAPE_CASSETTE_DIR"]).glob("*.yaml"))
    assert len(cassettes) == 2, f"Expected 2 cassettes, got {len(cassettes)}"

check("record-missing replays existing, records new", test_record_missing)

# ── 5. Streaming raises TapeStreamingError ────────────────────────────────────
print("\n5. Streaming guard")

def test_streaming_blocked():
    os.environ["LLMTAPE_MODE"] = "record"
    reset_config()

    import openai
    client = openai.OpenAI(api_key="fake")

    @llmtape.tape
    def stream_fn(q: str):
        return client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": q}],
            stream=True,
        )

    try:
        stream_fn("hi")
        raise AssertionError("Should have raised TapeStreamingError")
    except TapeStreamingError as e:
        assert "stream=True" in str(e)

check("TapeStreamingError raised for stream=True calls", test_streaming_blocked)

# ── 6. Async replay ───────────────────────────────────────────────────────────
print("\n6. Async replay")

def test_async_replay():
    import asyncio
    import openai.resources.chat.completions as _oa_async

    async_live_calls = [0]

    async def mock_async_create(self, **kwargs):
        async_live_calls[0] += 1
        return fake_openai_response("Paris (async)")

    # Record first
    os.environ["LLMTAPE_MODE"] = "record"
    reset_config()

    import openai
    async_client = openai.AsyncOpenAI(api_key="fake")

    @llmtape.tape
    async def ask_async(country: str):
        return await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Capital of {country}?"}],
        )

    with patch.object(_oa_async.AsyncCompletions, "create", mock_async_create):
        asyncio.run(ask_async("France"))

    assert async_live_calls[0] == 1, "Should have made one live call during record"

    # Now replay — no live call
    os.environ["LLMTAPE_MODE"] = "replay"
    reset_config()

    with patch.object(_oa_async.AsyncCompletions, "create", mock_async_create):
        result = asyncio.run(ask_async("France"))

    assert result is not None
    assert async_live_calls[0] == 1, "No additional live calls during replay"

check("Async replay returns cassette", test_async_replay)

# ── 7. Tool call round-trip ───────────────────────────────────────────────────
print("\n7. Tool call round-trip")

def test_tool_calls():
    os.environ["LLMTAPE_MODE"] = "record"
    reset_config()

    tool_resp = MagicMock()
    tool_resp.__class__.__module__ = "openai.types.chat"
    tool_resp.__class__.__name__ = "ChatCompletion"
    tool_resp.model_dump.return_value = {
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    }

    def mock_tool_create(self, **kwargs):
        return tool_resp

    with patch.object(_oa.Completions, "create", mock_tool_create):
        import openai
        client = openai.OpenAI(api_key="fake")

        @llmtape.tape
        def get_weather_fn(city: str):
            return client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"Weather in {city}?"}],
                tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            )

        get_weather_fn("Paris")  # record

    os.environ["LLMTAPE_MODE"] = "replay"
    reset_config()

    with patch.object(_oa.Completions, "create", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no live calls"))):
        import openai
        client = openai.OpenAI(api_key="fake")

        @llmtape.tape
        def get_weather_fn(city: str):
            return client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"Weather in {city}?"}],
                tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            )

        replayed = get_weather_fn("Paris")

    raw = replayed.model_dump() if hasattr(replayed, "model_dump") else replayed._raw
    tool_calls = raw["choices"][0]["message"]["tool_calls"]
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "get_weather"

check("Tool calls preserved through record/replay cycle", test_tool_calls)

# ── 8. CLI smoke ──────────────────────────────────────────────────────────────
print("\n8. CLI")

def test_cli():
    from click.testing import CliRunner
    from llmtape.cli import cli

    runner = CliRunner()

    result = runner.invoke(cli, ["list", "--cassette-dir", str(Path(tmpdir) / ".cassettes")])
    assert result.exit_code == 0, f"list failed: {result.output}"

    cassettes = list(Path(tmpdir, ".cassettes").glob("*.yaml"))
    if cassettes:
        result = runner.invoke(cli, ["show", cassettes[0].name, "--cassette-dir", str(Path(tmpdir) / ".cassettes")])
        assert result.exit_code == 0, f"show failed: {result.output}"

    result = runner.invoke(cli, ["check", "--cassette-dir", str(Path(tmpdir) / ".cassettes"), "--max-age", "999"])
    assert result.exit_code == 0
    assert "fresh" in result.output

check("CLI list/show/check all exit 0", test_cli)

# ── Summary ───────────────────────────────────────────────────────────────────
shutil.rmtree(tmpdir, ignore_errors=True)

passed = sum(1 for s, _ in results if s == PASS)
failed = sum(1 for s, _ in results if s == FAIL)
print(f"\n{'='*40}")
print(f"  {passed} passed  {failed} failed  ({len(results)} total)")
print(f"{'='*40}\n")
sys.exit(0 if failed == 0 else 1)
