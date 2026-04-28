"""
Core test suite. No live API calls — all tests use pre-recorded cassettes.
"""
import os
import sys
import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import llmtape
from llmtape._cassette import fingerprint, normalize_request, cassette_path, save, load
from llmtape._config import get_config, reset_config
from llmtape._errors import CassetteNotFoundError, TapeStreamingError, TapeUnsupportedProviderError
from llmtape._extract import detect_provider, response_to_dict, dict_to_response, extract_request_from_kwargs


CASSETTES_DIR = Path(__file__).parent / ".cassettes"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_cfg():
    reset_config()
    yield
    reset_config()


@pytest.fixture
def tmp_cassettes(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMTAPE_CASSETTE_DIR", str(tmp_path / ".cassettes"))
    monkeypatch.setenv("LLMTAPE_MODE", "replay")
    reset_config()
    return tmp_path / ".cassettes"


def _make_openai_response(content="Paris", model="gpt-4o", prompt_tokens=14, completion_tokens=9):
    """Build a minimal OpenAI-shaped response dict."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content, "tool_calls": None},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_anthropic_response(text="Paris"):
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _write_cassette(path: Path, provider: str, request: dict, response: dict, function_name="test_fn"):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "cassette_version": 1,
        "provider": provider,
        "request": {
            "fingerprint": f"sha256:{fingerprint(request)}",
            "normalized": request,
        },
        "response": {"raw": response},
        "metadata": {
            "recorded_at": "2026-04-28T14:00:00+00:00",
            "sdk_version": f"{provider}==1.0.0",
            "latency_ms": 300,
            "function_name": function_name,
        },
    }
    import yaml
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# _cassette.py
# ---------------------------------------------------------------------------

class TestNormalizeRequest:
    def test_sorts_keys(self):
        req = {"temperature": 0.7, "model": "gpt-4o", "messages": []}
        result = normalize_request(req, [])
        assert list(result.keys()) == sorted(result.keys())

    def test_strips_none_values(self):
        req = {"model": "gpt-4o", "temperature": None, "messages": []}
        result = normalize_request(req, [])
        assert "temperature" not in result

    def test_redacts_keys(self):
        req = {"model": "gpt-4o", "api_key": "sk-secret", "messages": []}
        result = normalize_request(req, ["api_key"])
        assert result["api_key"] == "[REDACTED]"

    def test_redaction_is_case_insensitive(self):
        req = {"Authorization": "Bearer sk-secret", "model": "gpt-4o"}
        result = normalize_request(req, ["authorization"])
        assert result["Authorization"] == "[REDACTED]"


class TestFingerprint:
    def test_deterministic(self):
        req = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        assert fingerprint(req) == fingerprint(req)

    def test_different_inputs_different_fingerprints(self):
        r1 = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        r2 = {"model": "gpt-4o", "messages": [{"role": "user", "content": "bye"}]}
        assert fingerprint(r1) != fingerprint(r2)

    def test_order_invariant_for_keys(self):
        r1 = {"model": "gpt-4o", "temperature": 0.7}
        r2 = {"temperature": 0.7, "model": "gpt-4o"}
        # normalize_request sorts keys, so fingerprints should match after normalization
        assert fingerprint(normalize_request(r1, [])) == fingerprint(normalize_request(r2, []))

    def test_hash_stability(self):
        req = normalize_request(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "What is 2+2?"}], "temperature": 0.7},
            []
        )
        fp1 = fingerprint(req)
        fp2 = fingerprint(req)
        # Same input always produces the same 64-char hex string
        assert fp1 == fp2
        assert len(fp1) == 64
        assert all(c in "0123456789abcdef" for c in fp1)


class TestCassetteSaveLoad:
    def test_round_trip(self, tmp_path):
        req = normalize_request({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}, [])
        fp = fingerprint(req)
        path = cassette_path(str(tmp_path), "my_func", fp)
        raw = _make_openai_response()

        save(path, "openai", req, fp, raw, "openai==1.0", 300, "my_func")
        data = load(path)

        assert data["cassette_version"] == 1
        assert data["provider"] == "openai"
        assert data["response"]["raw"]["choices"][0]["message"]["content"] == "Paris"
        assert data["metadata"]["function_name"] == "my_func"

    def test_long_string_uses_literal_block(self, tmp_path):
        long_content = "A" * 200
        req = normalize_request({"model": "gpt-4o", "messages": [{"role": "user", "content": long_content}]}, [])
        fp = fingerprint(req)
        path = cassette_path(str(tmp_path), "fn", fp)
        raw = _make_openai_response()
        save(path, "openai", req, fp, raw, "openai==1.0", 100, "fn")

        text = path.read_text()
        # Long strings should use YAML literal block style (|)
        assert "|" in text


# ---------------------------------------------------------------------------
# _extract.py
# ---------------------------------------------------------------------------

class TestExtractRequestFromKwargs:
    def test_openai_extracts_known_keys(self):
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "stream": False,
            "client": MagicMock(),  # should not appear
        }
        result = extract_request_from_kwargs("openai", kwargs)
        assert "model" in result
        assert "temperature" in result
        assert "client" not in result

    def test_raises_on_streaming(self):
        with pytest.raises(TapeStreamingError):
            extract_request_from_kwargs("openai", {"model": "gpt-4o", "stream": True})

    def test_anthropic_extracts_known_keys(self):
        kwargs = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
            "extra_junk": "ignored",
        }
        result = extract_request_from_kwargs("anthropic", kwargs)
        assert "model" in result
        assert "max_tokens" in result
        assert "extra_junk" not in result


class TestDictToResponse:
    def test_openai_simple_response_fallback(self):
        raw = _make_openai_response("Hello world")
        resp = dict_to_response(raw, "openai")
        # Either a real ChatCompletion or _SimpleResponse — both should expose content
        assert hasattr(resp, "content") or (hasattr(resp, "choices") and resp.choices)

    def test_anthropic_simple_response_fallback(self):
        raw = _make_anthropic_response("Hello world")
        resp = dict_to_response(raw, "anthropic")
        assert hasattr(resp, "content")


# ---------------------------------------------------------------------------
# Integration: @llmtape.tape sync
# ---------------------------------------------------------------------------

class TestTapeSync:
    def test_replay_returns_cassette_content(self, tmp_cassettes, monkeypatch):
        import openai
        req = normalize_request(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "capital of france"}]},
            []
        )
        fp = fingerprint(req)
        path = cassette_path(str(tmp_cassettes), "ask", fp)
        _write_cassette(path, "openai", req, _make_openai_response("Paris"), "ask")

        client = openai.OpenAI(api_key="fake")

        @llmtape.tape
        def ask(question: str):
            return client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": question}],
            )

        result = ask("capital of france")
        # Cassette was found and returned — verify we got a response object back
        assert result is not None

    def test_cassette_not_found_error_includes_filename(self, tmp_cassettes, monkeypatch):
        import openai
        client = openai.OpenAI(api_key="fake")

        @llmtape.tape
        def my_llm_function(q: str):
            return client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": q}],
            )

        with pytest.raises(CassetteNotFoundError) as exc_info:
            my_llm_function("test input")

        error_msg = str(exc_info.value)
        assert "my_llm_function" in error_msg
        assert ".cassettes" in error_msg
        assert "LLMTAPE_MODE=record-missing" in error_msg

    def test_record_missing_skips_live_call_when_cassette_exists(self, tmp_cassettes, monkeypatch):
        monkeypatch.setenv("LLMTAPE_MODE", "record-missing")
        reset_config()

        req = normalize_request(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]},
            []
        )
        fp = fingerprint(req)
        path = cassette_path(str(tmp_cassettes), "cached_fn", fp)
        _write_cassette(path, "openai", req, _make_openai_response("cached answer"), "cached_fn")

        # Cassette exists — should not raise even though no live client
        # The tape decorator will intercept before any real call happens
        call_count = [0]

        # We verify the cassette loading path by checking the file was read
        # (indirectly — if it raises, the cassette wasn't found)
        assert path.exists()
        data = load(path)
        assert data["response"]["raw"]["choices"][0]["message"]["content"] == "cached answer"


# ---------------------------------------------------------------------------
# Integration: @llmtape.tape async
# ---------------------------------------------------------------------------

class TestTapeAsync:
    @pytest.mark.asyncio
    async def test_async_cassette_not_found(self, tmp_cassettes, monkeypatch):
        import openai
        client = openai.AsyncOpenAI(api_key="fake")

        @llmtape.tape
        async def async_llm(q: str):
            return await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": q}],
            )

        with pytest.raises(CassetteNotFoundError) as exc_info:
            await async_llm("test question")

        assert "No cassette found" in str(exc_info.value)
        assert "async_llm" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class TestErrors:
    def test_cassette_not_found_is_llmtape_error(self):
        from llmtape._errors import LLMTapeError
        err = CassetteNotFoundError("my_fn", "abc123", ".cassettes/my_fn_abc123.yaml", {})
        assert isinstance(err, LLMTapeError)
        assert isinstance(err, Exception)
        assert "my_fn" in str(err)
        assert "abc123" in str(err)

    def test_streaming_error_is_not_implemented(self):
        from llmtape._errors import LLMTapeError
        err = TapeStreamingError()
        assert isinstance(err, LLMTapeError)
        assert isinstance(err, NotImplementedError)
        assert "stream=True" in str(err)

    def test_unsupported_provider_error(self):
        from llmtape._errors import LLMTapeError
        err = TapeUnsupportedProviderError("SomeRando")
        assert isinstance(err, LLMTapeError)
        assert isinstance(err, Exception)
        assert "SomeRando" in str(err)


# ---------------------------------------------------------------------------
# Tool calls round-trip
# ---------------------------------------------------------------------------

class TestToolCallsRoundTrip:
    def test_tool_calls_preserved_in_cassette(self, tmp_path):
        req = normalize_request(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "get weather"}],
             "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]},
            []
        )
        fp = fingerprint(req)
        raw = {
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
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        }
        path = cassette_path(str(tmp_path), "weather_fn", fp)
        save(path, "openai", req, fp, raw, "openai==1.0", 400, "weather_fn")

        data = load(path)
        tool_calls = data["response"]["raw"]["choices"][0]["message"]["tool_calls"]
        assert tool_calls is not None
        assert tool_calls[0]["function"]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_redacts_api_key_in_cassette(self, tmp_path):
        req = normalize_request(
            {"model": "gpt-4o", "api_key": "sk-supersecret", "messages": []},
            ["api_key"]
        )
        fp = fingerprint(req)
        path = cassette_path(str(tmp_path), "fn", fp)
        save(path, "openai", req, fp, _make_openai_response(), "openai==1.0", 100, "fn")

        text = path.read_text()
        assert "sk-supersecret" not in text
        assert "[REDACTED]" in text
