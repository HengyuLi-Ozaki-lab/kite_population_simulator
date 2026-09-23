"""Project settings. Override any field with a KITE_-prefixed environment variable.

API keys are not stored here: the TypeSafe, OpenAI and Anthropic SDKs read TYPESAFE_API_KEY,
OPENAI_API_KEY and ANTHROPIC_API_KEY from the environment themselves. The CLI loads `.env` first.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KITE_", extra="ignore")

    cache_path: Path = Path("cache/responses.sqlite")
    prices_path: Path = Path("configs/prices.yaml")
    freeze_path: Path = Path("configs/eval/frozen.yaml")  # the phrasing chosen on dev; required by every test run
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    rpm: float = 960.0  # 80% of Jev's published 1,200 requests per minute
    tps: float = 200_000.0  # 80% of Jev's published 250,000 tokens per second
    max_concurrency: int = 16
