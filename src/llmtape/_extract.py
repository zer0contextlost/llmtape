"""
Provider-specific: extract normalized request dict and raw response dict
from a captured API call, and detect provider from response object type.
"""
from __future__ import annotations
from typing import Any

from ._errors import TapeStreamingError, TapeUnsupportedProviderError


def detect_provider(response: Any) -> str:
    t = type(response)
    module = getattr(t, "__module__", "") or ""
    name = t.__name__

    if module.startswith("openai"):
        return "openai"
    if module.startswith("anthropic"):
        return "anthropic"
    # Fallback: check class name
    if "ChatCompletion" in name or "Completion" in name:
        return "openai"
    if "Message" in name and "anthropic" in str(t).lower():
        return "anthropic"
    raise TapeUnsupportedProviderError(f"{module}.{name}")


def sdk_version(provider: str) -> str:
    try:
        if provider == "openai":
            import openai
            return f"openai=={openai.__version__}"
        if provider == "anthropic":
            import anthropic
            return f"anthropic=={anthropic.__version__}"
    except Exception:
        pass
    return f"{provider}==unknown"


def response_to_dict(response: Any, provider: str) -> dict:
    """Serialize the full SDK response to a plain dict."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "__dict__"):
        import copy
        return copy.deepcopy(response.__dict__)
    raise TapeUnsupportedProviderError(type(response).__name__)


def dict_to_response(raw: dict, provider: str) -> Any:
    """Reconstruct a provider response object from a saved dict."""
    if provider == "openai":
        try:
            from openai.types.chat import ChatCompletion
            return ChatCompletion.model_validate(raw)
        except Exception:
            pass
        # Fallback: return a simple namespace that mimics the response
        return _SimpleResponse(raw, provider)
    if provider == "anthropic":
        try:
            from anthropic.types import Message
            return Message.model_validate(raw)
        except Exception:
            pass
        return _SimpleResponse(raw, provider)
    return _SimpleResponse(raw, provider)


class _SimpleResponse:
    """Minimal stand-in when SDK model reconstruction fails."""
    def __init__(self, raw: dict, provider: str):
        self._raw = raw
        self._provider = provider
        # Expose common attributes
        if provider == "openai":
            choices = raw.get("choices", [{}])
            msg = choices[0].get("message", {}) if choices else {}
            self.content = msg.get("content", "")
            self.tool_calls = msg.get("tool_calls")
            self.finish_reason = choices[0].get("finish_reason") if choices else None
            self.usage = raw.get("usage", {})
        elif provider == "anthropic":
            content_blocks = raw.get("content", [])
            text_blocks = [b for b in content_blocks if b.get("type") == "text"]
            self.content = text_blocks[0].get("text", "") if text_blocks else ""
            self.stop_reason = raw.get("stop_reason")
            self.usage = raw.get("usage", {})

    def model_dump(self) -> dict:
        return self._raw

    def to_dict(self) -> dict:
        return self._raw

    def __repr__(self) -> str:
        return f"<LLMTapeReplay provider={self._provider}>"


def extract_request_from_kwargs(provider: str, kwargs: dict) -> dict:
    """
    Extract a normalized request dict from the kwargs passed to the provider API.
    Handles streaming detection.
    """
    if kwargs.get("stream"):
        raise TapeStreamingError()

    if provider == "openai":
        keys = ["model", "messages", "temperature", "max_tokens", "top_p",
                "frequency_penalty", "presence_penalty", "tools", "tool_choice",
                "response_format", "seed", "stop"]
    elif provider == "anthropic":
        keys = ["model", "messages", "max_tokens", "temperature", "top_p",
                "tools", "tool_choice", "system", "stop_sequences"]
    else:
        keys = list(kwargs.keys())

    return {k: kwargs[k] for k in keys if k in kwargs}
