"""
LLMTape — record and replay LLM API calls in tests.

Usage:
    import llmtape

    @llmtape.tape
    def answer(question: str) -> str:
        return openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}],
        )

Modes (LLMTAPE_MODE env var):
    replay         - return saved cassette, no network (default)
    record         - call through and save cassette
    record-missing - replay if cassette exists, record if not
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Callable

log = logging.getLogger("llmtape")

from ._cassette import (
    cassette_path,
    fingerprint,
    load as load_cassette,
    normalize_request,
    save as save_cassette,
)
from ._config import get_config
from ._errors import CassetteNotFoundError, TapeStreamingError, TapeUnsupportedProviderError
from ._extract import (
    detect_provider,
    dict_to_response,
    extract_request_from_kwargs,
    response_to_dict,
    sdk_version,
)

__all__ = [
    "tape",
    "CassetteNotFoundError",
    "TapeStreamingError",
    "TapeUnsupportedProviderError",
]


def _get_provider_targets() -> list[tuple[str, str, str]]:
    """Return (provider, module_path, method_name) for each supported provider."""
    targets = []
    try:
        import openai  # noqa: F401
        targets.append(("openai", "openai.resources.chat.completions", "Completions.create"))
    except ImportError:
        pass
    try:
        import anthropic  # noqa: F401
        targets.append(("anthropic", "anthropic.resources.messages", "Messages.create"))
    except ImportError:
        pass
    return targets


def _qualified_name(fn: Callable) -> str:
    """Return module__name for cassette filenames.

    Uses __name__ (not __qualname__) to avoid '<locals>' segments that differ
    between the record and replay call sites. Uses module to distinguish same-named
    functions across different modules. Falls back to repr for partials/lambdas
    that lack a meaningful __name__.
    """
    module = getattr(fn, "__module__", None) or ""
    name = getattr(fn, "__name__", None)
    if not name or name == "<lambda>":
        name = getattr(fn, "__qualname__", None) or repr(fn)
    return f"{module}__{name}" if module else name


def _warn_if_stale(path: "Path", cfg, data: dict) -> None:
    """Warn if the cassette is older than max_age_days. Accepts pre-loaded data to avoid double read."""
    from datetime import datetime, timezone
    recorded_at_str = data.get("metadata", {}).get("recorded_at", "")
    if not recorded_at_str:
        return
    try:
        recorded = datetime.fromisoformat(recorded_at_str)
        days = (datetime.now(timezone.utc) - recorded).total_seconds() / 86400
    except Exception:
        return
    if days > cfg.max_age_days:
        log.warning(
            "llmtape: cassette '%s' is %.0f days old (max_age_days=%d). "
            "Re-record with LLMTAPE_MODE=record-missing.",
            path.name, days, cfg.max_age_days,
        )


def _intercept(qualified: str, cfg, original_fn: Callable, provider: str, *args, **kwargs) -> Any:
    """Shared sync interception logic."""
    raw_request = extract_request_from_kwargs(provider, kwargs)
    normalized = normalize_request(raw_request, cfg.redact)
    fp = fingerprint(normalized)
    path = cassette_path(cfg.cassette_dir, qualified, fp)

    mode = cfg.mode

    if mode in ("replay", "record-missing") and path.exists():
        data = load_cassette(path)
        _warn_if_stale(path, cfg, data)
        raw = data["response"]["raw"]
        prov = data.get("provider", provider)
        return dict_to_response(raw, prov)

    if mode == "replay":
        raise CassetteNotFoundError(qualified, fp, str(path), normalized)

    t0 = time.monotonic()
    response = original_fn(*args, **kwargs)
    latency_ms = int((time.monotonic() - t0) * 1000)

    prov = detect_provider(response)
    raw_response = response_to_dict(response, prov)
    save_cassette(
        path=path,
        provider=prov,
        normalized_request=normalized,
        fp=fp,
        raw_response=raw_response,
        sdk_version=sdk_version(prov),
        latency_ms=latency_ms,
        function_name=qualified,
    )
    return response


async def _intercept_async(qualified: str, cfg, original_fn: Callable, provider: str, *args, **kwargs) -> Any:
    """Shared async interception logic."""
    raw_request = extract_request_from_kwargs(provider, kwargs)
    normalized = normalize_request(raw_request, cfg.redact)
    fp = fingerprint(normalized)
    path = cassette_path(cfg.cassette_dir, qualified, fp)

    mode = cfg.mode

    if mode in ("replay", "record-missing") and path.exists():
        data = load_cassette(path)
        _warn_if_stale(path, cfg, data)
        raw = data["response"]["raw"]
        prov = data.get("provider", provider)
        return dict_to_response(raw, prov)

    if mode == "replay":
        raise CassetteNotFoundError(qualified, fp, str(path), normalized)

    t0 = time.monotonic()
    response = await original_fn(*args, **kwargs)
    latency_ms = int((time.monotonic() - t0) * 1000)

    prov = detect_provider(response)
    raw_response = response_to_dict(response, prov)
    save_cassette(
        path=path,
        provider=prov,
        normalized_request=normalized,
        fp=fp,
        raw_response=raw_response,
        sdk_version=sdk_version(prov),
        latency_ms=latency_ms,
        function_name=qualified,
    )
    return response


def tape(fn: Callable) -> Callable:
    """
    Decorator that records/replays LLM API calls made inside the wrapped function.

    Intercepts calls to openai.chat.completions.create and anthropic.messages.create
    for the duration of the wrapped function.
    """
    qualified = _qualified_name(fn)

    if asyncio.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            cfg = get_config()

            def make_sync_patch(provider: str, original):
                def patched(*a, **kw):
                    raise RuntimeError("Sync provider method called inside async function — use async client")
                return patched

            def make_async_patch(provider: str, original):
                async def patched(*a, **kw):
                    return await _intercept_async(qualified, cfg, original, provider, *a, **kw)
                return patched

            patches = _build_patches(make_sync_patch, make_async_patch)
            with _apply_patches(patches):
                return await fn(*args, **kwargs)

        return async_wrapper
    else:
        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            cfg = get_config()

            def make_sync_patch(provider: str, original):
                def patched(*a, **kw):
                    return _intercept(qualified, cfg, original, provider, *a, **kw)
                return patched

            def make_async_patch(provider: str, original):
                async def patched(*a, **kw):
                    return await _intercept_async(qualified, cfg, original, provider, *a, **kw)
                return patched

            patches = _build_patches(make_sync_patch, make_async_patch)
            with _apply_patches(patches):
                return fn(*args, **kwargs)

        return sync_wrapper


def _build_patches(make_sync: Callable, make_async: Callable) -> list[tuple[str, str, Any]]:
    """Build (module, attr, replacement) tuples for each provider."""
    patches = []

    # openai: chat completions (primary)
    try:
        import openai.resources.chat.completions as _oa_mod
        patches.append(("openai.resources.chat.completions", "Completions.create",
                        make_sync("openai", _oa_mod.Completions.create)))
        if hasattr(_oa_mod, "AsyncCompletions"):
            patches.append(("openai.resources.chat.completions", "AsyncCompletions.create",
                            make_async("openai", _oa_mod.AsyncCompletions.create)))
    except (ImportError, AttributeError):
        pass

    # openai: beta structured outputs (.parse) — SDK 1.40+
    try:
        import openai.resources.beta.chat.completions as _oa_beta
        if hasattr(_oa_beta, "Completions") and hasattr(_oa_beta.Completions, "parse"):
            patches.append(("openai.resources.beta.chat.completions", "Completions.parse",
                            make_sync("openai", _oa_beta.Completions.parse)))
        if hasattr(_oa_beta, "AsyncCompletions") and hasattr(_oa_beta.AsyncCompletions, "parse"):
            patches.append(("openai.resources.beta.chat.completions", "AsyncCompletions.parse",
                            make_async("openai", _oa_beta.AsyncCompletions.parse)))
    except (ImportError, AttributeError):
        pass

    # openai: responses API — SDK 1.40+
    try:
        import openai.resources.responses as _oa_resp
        if hasattr(_oa_resp, "Responses") and hasattr(_oa_resp.Responses, "create"):
            patches.append(("openai.resources.responses", "Responses.create",
                            make_sync("openai", _oa_resp.Responses.create)))
        if hasattr(_oa_resp, "AsyncResponses") and hasattr(_oa_resp.AsyncResponses, "create"):
            patches.append(("openai.resources.responses", "AsyncResponses.create",
                            make_async("openai", _oa_resp.AsyncResponses.create)))
    except (ImportError, AttributeError):
        pass

    # anthropic: messages
    try:
        import anthropic.resources.messages as _an_mod
        patches.append(("anthropic.resources.messages", "Messages.create",
                        make_sync("anthropic", _an_mod.Messages.create)))
        if hasattr(_an_mod, "AsyncMessages"):
            patches.append(("anthropic.resources.messages", "AsyncMessages.create",
                            make_async("anthropic", _an_mod.AsyncMessages.create)))
    except (ImportError, AttributeError):
        pass

    return patches


from contextlib import contextmanager

@contextmanager
def _apply_patches(patches: list[tuple[str, str, Any]]):
    """Apply all patches and restore on exit."""
    originals = []
    for module_path, attr_path, replacement in patches:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            parts = attr_path.split(".")
            obj = mod
            for part in parts[:-1]:
                obj = getattr(obj, part)
            attr = parts[-1]
            original = getattr(obj, attr)
            setattr(obj, attr, replacement)
            originals.append((obj, attr, original))
        except ImportError:
            pass  # provider not installed — expected
        except AttributeError as e:
            log.warning("llmtape: could not patch %s.%s: %s", module_path, attr_path, e)
    try:
        yield
    finally:
        for obj, attr, original in originals:
            setattr(obj, attr, original)
