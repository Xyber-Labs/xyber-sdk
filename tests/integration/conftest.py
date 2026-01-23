"""
Pytest configuration for integration tests.

Loads API keys from tests/.env.test (copy from tests/.env.test.example).
"""
from pathlib import Path

import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict

_tests_dir = Path(__file__).parent.parent


class APIKeys(BaseSettings):
    """API keys for integration tests."""

    model_config = SettingsConfigDict(
        env_file=_tests_dir / ".env.test",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str | None = None
    together_api_key: str | None = None

    @property
    def has_google(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_together(self) -> bool:
        return bool(self.together_api_key)

    @property
    def has_all(self) -> bool:
        return self.has_google and self.has_together


@pytest.fixture(scope="session")
def api_keys() -> APIKeys:
    """Provide API keys from tests/.env.test file."""
    return APIKeys()


@pytest.fixture(scope="session")
def require_google_api_key(api_keys: APIKeys):
    """Skip test if Google API key is not configured."""
    if not api_keys.has_google:
        pytest.skip("GOOGLE_API_KEY not configured. Copy tests/.env.test.example to tests/.env.test")


@pytest.fixture(scope="session")
def require_together_api_key(api_keys: APIKeys):
    """Skip test if Together API key is not configured."""
    if not api_keys.has_together:
        pytest.skip("TOGETHER_API_KEY not configured. Copy tests/.env.test.example to tests/.env.test")


@pytest.fixture(scope="session")
def require_all_api_keys(api_keys: APIKeys):
    """Skip test if any API key is missing."""
    if not api_keys.has_all:
        missing = []
        if not api_keys.has_google:
            missing.append("GOOGLE_API_KEY")
        if not api_keys.has_together:
            missing.append("TOGETHER_API_KEY")
        pytest.skip(f"{', '.join(missing)} not configured. Copy tests/.env.test.example to tests/.env.test")
