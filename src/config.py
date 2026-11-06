"""Typed project settings, loaded from ``config/settings.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


class BPESettings(BaseModel):
    vocab_size: int = 512
    demo_vocab_sizes: list[int] = Field(default_factory=lambda: [256, 512, 1024, 2048])


class CorporaSettings(BaseModel):
    training_chars: int = 200_000


class CostSettings(BaseModel):
    usd_per_1k_tokens: float = 0.01


class Settings(BaseModel):
    """All project settings, with sensible defaults if the YAML is absent."""

    seed: int = 7
    bpe: BPESettings = Field(default_factory=BPESettings)
    corpora: CorporaSettings = Field(default_factory=CorporaSettings)
    cost: CostSettings = Field(default_factory=CostSettings)


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from ``path`` (default ``config/settings.yaml``)."""
    resolved = path if path is not None else DEFAULT_CONFIG_PATH
    if resolved.exists():
        data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        return Settings.model_validate(data)
    return Settings()


settings = load_settings()
