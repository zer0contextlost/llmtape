from __future__ import annotations
import os
import tomllib
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class TapeConfig:
    mode: str = "replay"          # replay | record | record-missing
    cassette_dir: str = ".cassettes"
    max_age_days: int = 30
    redact: list[str] = field(default_factory=lambda: [
        "authorization", "api_key", "x-api-key",
        "openai-organization", "anthropic-auth-token",
    ])


def _load_file_config() -> dict:
    for name in [".llmtape.toml", "pyproject.toml"]:
        p = Path.cwd() / name
        if not p.exists():
            continue
        try:
            with open(p, "rb") as f:
                data = tomllib.load(f)
            if name == "pyproject.toml":
                data = data.get("tool", {}).get("llmtape", {})
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


_config: TapeConfig | None = None
_config_env_snapshot: tuple[str, str] | None = None

_ENV_KEYS = ("LLMTAPE_MODE", "LLMTAPE_CASSETTE_DIR")


def _env_snapshot() -> tuple[str, str]:
    return tuple(os.environ.get(k, "") for k in _ENV_KEYS)  # type: ignore[return-value]


def get_config() -> TapeConfig:
    global _config, _config_env_snapshot
    current_snapshot = _env_snapshot()
    if _config is not None and _config_env_snapshot == current_snapshot:
        return _config

    file_cfg = _load_file_config()
    cfg = TapeConfig(
        mode=os.environ.get("LLMTAPE_MODE", file_cfg.get("mode", "replay")),
        cassette_dir=os.environ.get("LLMTAPE_CASSETTE_DIR", file_cfg.get("cassette_dir", ".cassettes")),
        max_age_days=int(file_cfg.get("max_age_days", 30)),
        redact=file_cfg.get("redact", TapeConfig().redact),
    )
    _config = cfg
    _config_env_snapshot = current_snapshot
    return cfg


def reset_config() -> None:
    global _config, _config_env_snapshot
    _config = None
    _config_env_snapshot = None
