from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Patterns that look like API keys in string values — redacted regardless of key name
_VALUE_REDACT_PATTERNS = re.compile(
    r'(sk-[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._\-]{20,})',
    re.IGNORECASE,
)

CASSETTE_VERSION = 1
_LONG_STRING_THRESHOLD = 80


def _literal_str(s: str) -> yaml.ScalarNode:
    """Force YAML literal block scalar (|) for long/multiline strings.
    Falls back to double-quoted style if any line has trailing whitespace —
    literal blocks strip trailing whitespace, which would corrupt the replay value.
    """
    tag = "tag:yaml.org,2002:str"
    wants_literal = "\n" in s or len(s) > _LONG_STRING_THRESHOLD
    has_trailing_ws = any(line != line.rstrip(" \t") for line in s.splitlines())
    if wants_literal and not has_trailing_ws:
        style = "|"
    elif wants_literal:
        style = '"'  # double-quoted preserves trailing whitespace exactly
    else:
        style = None
    return yaml.ScalarNode(tag=tag, value=s, style=style)


class _LiteralDumper(yaml.Dumper):
    pass


_LiteralDumper.add_representer(
    str,
    lambda dumper, data: _literal_str(data),
)


def _redact(obj: Any, keys: set[str]) -> Any:
    if isinstance(obj, dict):
        return {
            k: "[REDACTED]" if k.lower() in keys else _redact(v, keys)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(item, keys) for item in obj]
    if isinstance(obj, str):
        return _VALUE_REDACT_PATTERNS.sub("[REDACTED]", obj)
    return obj


def normalize_request(request: dict, redact_keys: list[str]) -> dict:
    """Sort keys, strip top-level None values, redact sensitive fields.

    None is only stripped at the top level — inside messages/content/tools,
    None values are meaningful (e.g. Anthropic tool-result content=None).
    """
    redact_set = {k.lower() for k in redact_keys}

    def _sort_keys(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _sort_keys(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return [_sort_keys(item) for item in obj]
        return obj

    # Strip None only at top level
    top_level = {k: v for k, v in request.items() if v is not None}
    sorted_req = _sort_keys(top_level)
    return _redact(sorted_req, redact_set)


def _json_default(obj: Any) -> Any:
    """Strict serializer — rejects non-primitive objects rather than calling str()
    so fingerprints don't silently become non-deterministic on Pydantic models etc."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON-serializable. "
        "Convert it to a primitive (dict/list/str/int/float) before passing to the LLM API."
    )


def fingerprint(normalized: dict) -> str:
    serialized = json.dumps(normalized, sort_keys=True, default=_json_default)
    return hashlib.sha256(serialized.encode()).hexdigest()


def cassette_path(cassette_dir: str, qualified_name: str, fp: str) -> Path:
    """Build cassette path from qualified function name (module.qualname) and fingerprint."""
    safe_name = re.sub(r"[^\w]", "_", qualified_name)
    if len(safe_name) > 60:
        # Append 6-char hash of the full name so truncated names don't collide
        name_hash = hashlib.sha256(safe_name.encode()).hexdigest()[:6]
        safe_name = safe_name[-54:] + "_" + name_hash
    return Path(cassette_dir) / f"{safe_name}_{fp[:12]}.yaml"


def save(
    path: Path,
    provider: str,
    normalized_request: dict,
    fp: str,
    raw_response: dict,
    sdk_version: str,
    latency_ms: int,
    function_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "cassette_version": CASSETTE_VERSION,
        "provider": provider,
        "request": {
            "fingerprint": f"sha256:{fp}",
            "normalized": normalized_request,
        },
        "response": {
            "raw": raw_response,
        },
        "metadata": {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "sdk_version": sdk_version,
            "latency_ms": latency_ms,
            "function_name": function_name,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=_LiteralDumper, allow_unicode=True,
                  default_flow_style=False, sort_keys=False)


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def age_days(path: Path) -> float:
    recorded_at_str = load(path).get("metadata", {}).get("recorded_at", "")
    if not recorded_at_str:
        return float("inf")
    try:
        recorded = datetime.fromisoformat(recorded_at_str)
        return (datetime.now(timezone.utc) - recorded).total_seconds() / 86400
    except Exception:
        return float("inf")
