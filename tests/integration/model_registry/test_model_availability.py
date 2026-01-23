"""
Integration tests to verify all SupportedModels are available via their APIs.

Setup:
    1. Copy tests/.env.test.example to tests/.env.test
    2. Fill in your API keys

Run:
    uv run pytest tests/integration/ -v
"""
import asyncio

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether

from xyber_sdk.model_registry import SupportedModels, get_model


@pytest.fixture(autouse=True)
async def rate_limit_delay():
    """Add small delay between tests to avoid rate limiting."""
    yield
    await asyncio.sleep(0.5)


def get_google_models() -> list[SupportedModels]:
    """Return all Google models from SupportedModels enum."""
    return [m for m in SupportedModels if m.model_provider == ChatGoogleGenerativeAI]


def get_together_models() -> list[SupportedModels]:
    """Return all Together AI models from SupportedModels enum."""
    return [m for m in SupportedModels if m.model_provider == ChatTogether]


@pytest.mark.usefixtures("require_google_api_key")
class TestGoogleModels:
    """Tests for Google (Gemini) models."""

    @pytest.mark.parametrize("model", get_google_models(), ids=lambda m: m.name)
    def test_model_initialization(self, api_keys, model: SupportedModels):
        """Verify each Google model can be initialized."""
        llm = get_model(model, google_api_key=api_keys.google_api_key)
        assert llm is not None
        assert isinstance(llm, ChatGoogleGenerativeAI)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", get_google_models(), ids=lambda m: m.name)
    async def test_model_responds(self, api_keys, model: SupportedModels):
        """Verify each Google model responds to a simple prompt."""
        llm = get_model(model, google_api_key=api_keys.google_api_key)
        response = await llm.ainvoke("Say 'hello' in one word")
        assert response.content, f"Model {model.name} returned empty response"


@pytest.mark.usefixtures("require_together_api_key")
class TestTogetherModels:
    """Tests for Together AI models."""

    @pytest.mark.parametrize("model", get_together_models(), ids=lambda m: m.name)
    def test_model_initialization(self, api_keys, model: SupportedModels):
        """Verify each Together AI model can be initialized."""
        llm = get_model(model, together_api_key=api_keys.together_api_key)
        assert llm is not None
        assert isinstance(llm, ChatTogether)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", get_together_models(), ids=lambda m: m.name)
    async def test_model_responds(self, api_keys, model: SupportedModels):
        """Verify each Together AI model responds to a simple prompt."""
        llm = get_model(model, together_api_key=api_keys.together_api_key)
        response = await llm.ainvoke("Say 'hello' in one word")
        assert response.content, f"Model {model.name} returned empty response"
