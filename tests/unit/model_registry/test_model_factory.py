import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether

from xyber_sdk.model_registry.chat_models import (
    SupportedGoogleModels,
    SupportedModels,
    SupportedTogetherModels,
)
from xyber_sdk.model_registry.model_factory import (
    get_model,
    get_multiple_model_instances,
)

# --- Fixtures ---


@pytest.fixture
def clean_env():
    """Clean environment without API keys."""
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def google_keys_env():
    """Environment with Google API keys (JSON array format)."""
    env_vars = {"GOOGLE_API_KEYS": '["env-gkey1","env-gkey2"]'}
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def together_keys_env():
    """Environment with Together API keys (JSON array format)."""
    env_vars = {"TOGETHER_API_KEYS": '["env-tkey1","env-tkey2"]'}
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def both_keys_env():
    """Environment with both Google and Together API keys (JSON array format)."""
    env_vars = {
        "GOOGLE_API_KEYS": '["env-gkey1","env-gkey2"]',
        "TOGETHER_API_KEYS": '["env-tkey1"]',
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


# --- get_model() Happy Path Tests ---


def test_get_model_creates_google_model():
    """get_model creates ChatGoogleGenerativeAI for Google models."""
    with patch.object(SupportedModels.GEMINI_2_0_FLASH, "model_provider") as mock_provider:
        mock_instance = MagicMock(spec=ChatGoogleGenerativeAI)
        mock_provider.return_value = mock_instance

        model = get_model(SupportedModels.GEMINI_2_0_FLASH, google_api_key="test-key")

        mock_provider.assert_called_once_with(
            model="gemini-2.0-flash",
            google_api_key="test-key",
        )
        assert model is mock_instance


def test_get_model_creates_together_model():
    """get_model creates ChatTogether for Together AI models."""
    with patch.object(SupportedModels.META_LLAMA_3_3_70B, "model_provider") as mock_provider:
        mock_instance = MagicMock(spec=ChatTogether)
        mock_provider.return_value = mock_instance

        model = get_model(SupportedModels.META_LLAMA_3_3_70B, together_api_key="test-key")

        mock_provider.assert_called_once_with(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            together_api_key="test-key",
        )
        assert model is mock_instance


def test_get_model_passes_additional_kwargs():
    """get_model forwards additional kwargs to model constructor."""
    with patch.object(SupportedModels.GEMINI_2_0_FLASH, "model_provider") as mock_provider:
        mock_instance = MagicMock(spec=ChatGoogleGenerativeAI)
        mock_provider.return_value = mock_instance

        model = get_model(
            SupportedModels.GEMINI_2_0_FLASH,
            google_api_key="test-key",
            temperature=0.7,
            max_tokens=1000,
        )

        mock_provider.assert_called_once_with(
            model="gemini-2.0-flash",
            google_api_key="test-key",
            temperature=0.7,
            max_tokens=1000,
        )
        assert model is mock_instance


def test_get_model_with_google_enum():
    """get_model works with SupportedGoogleModels enum."""
    with patch.object(SupportedGoogleModels.GEMINI_2_5_PRO, "model_provider") as mock_provider:
        mock_instance = MagicMock(spec=ChatGoogleGenerativeAI)
        mock_provider.return_value = mock_instance

        model = get_model(SupportedGoogleModels.GEMINI_2_5_PRO, google_api_key="test-key")

        mock_provider.assert_called_once_with(
            model="gemini-2.5-pro",
            google_api_key="test-key",
        )
        assert model is mock_instance


def test_get_model_with_together_enum():
    """get_model works with SupportedTogetherModels enum."""
    with patch.object(SupportedTogetherModels.QWEN_2_5_7B_TURBO, "model_provider") as mock_provider:
        mock_instance = MagicMock(spec=ChatTogether)
        mock_provider.return_value = mock_instance

        model = get_model(
            SupportedTogetherModels.QWEN_2_5_7B_TURBO,
            together_api_key="test-key",
        )

        mock_provider.assert_called_once_with(
            model="Qwen/Qwen2.5-7B-Instruct-Turbo",
            together_api_key="test-key",
        )
        assert model is mock_instance


# --- get_model() Error Path Tests ---


def test_get_model_raises_value_error_for_invalid_kwargs():
    """get_model raises ValueError when model constructor fails with TypeError."""
    with patch.object(SupportedModels.GEMINI_2_0_FLASH, "model_provider") as mock_provider:
        mock_provider.side_effect = TypeError("unexpected keyword argument")

        with pytest.raises(ValueError) as exc_info:
            get_model(
                SupportedModels.GEMINI_2_0_FLASH,
                google_api_key="test-key",
                invalid_param="bad",
            )

        assert "Error initializing model" in str(exc_info.value)
        assert "gemini-2.0-flash" in str(exc_info.value)


def test_get_model_propagates_type_error_as_value_error():
    """get_model wraps TypeError in ValueError with helpful message."""
    with patch.object(SupportedModels.DEEPSEEK_V3, "model_provider") as mock_provider:
        mock_provider.side_effect = TypeError("got unexpected kwarg")

        with pytest.raises(ValueError) as exc_info:
            get_model(SupportedModels.DEEPSEEK_V3, bad_arg=True)

        error_msg = str(exc_info.value)
        assert "Error initializing model" in error_msg
        assert "deepseek-ai/DeepSeek-V3" in error_msg
        assert "Ensure the arguments are valid" in error_msg


# --- get_multiple_model_instances() Happy Path Tests ---


def test_get_multiple_instances_creates_google_models(clean_env):
    """get_multiple_model_instances creates one instance per Google API key."""
    with patch("xyber_sdk.model_registry.model_factory.get_model") as mock_get_model:
        mock_get_model.return_value = MagicMock(spec=ChatGoogleGenerativeAI)

        models = get_multiple_model_instances(
            model_names=["GEMINI_2_0_FLASH"],
            google_api_keys=["key1", "key2", "key3"],
            together_api_keys=[],
        )

        assert len(models) == 3
        assert mock_get_model.call_count == 3

        # Verify each key was used
        call_args_list = [call.kwargs["google_api_key"] for call in mock_get_model.call_args_list]
        assert call_args_list == ["key1", "key2", "key3"]


def test_get_multiple_instances_creates_together_models(clean_env):
    """get_multiple_model_instances creates one instance per Together API key."""
    with patch("xyber_sdk.model_registry.model_factory.get_model") as mock_get_model:
        mock_get_model.return_value = MagicMock(spec=ChatTogether)

        models = get_multiple_model_instances(
            model_names=["META_LLAMA_3_3_70B"],
            google_api_keys=[],
            together_api_keys=["tkey1", "tkey2"],
        )

        assert len(models) == 2
        assert mock_get_model.call_count == 2

        # Verify each key was used
        call_args_list = [call.kwargs["together_api_key"] for call in mock_get_model.call_args_list]
        assert call_args_list == ["tkey1", "tkey2"]


def test_get_multiple_instances_mixed_providers(clean_env):
    """get_multiple_model_instances handles mixed Google and Together models."""
    with patch("xyber_sdk.model_registry.model_factory.get_model") as mock_get_model:
        mock_get_model.return_value = MagicMock()

        models = get_multiple_model_instances(
            model_names=["GEMINI_2_0_FLASH", "META_LLAMA_3_3_70B"],
            google_api_keys=["gkey1"],
            together_api_keys=["tkey1"],
        )

        assert len(models) == 2
        assert mock_get_model.call_count == 2


def test_get_multiple_instances_multiple_models_and_keys(clean_env):
    """get_multiple_model_instances creates correct total count (models x keys)."""
    with patch("xyber_sdk.model_registry.model_factory.get_model") as mock_get_model:
        mock_get_model.return_value = MagicMock()

        models = get_multiple_model_instances(
            model_names=["GEMINI_2_0_FLASH", "GEMINI_2_5_PRO"],
            google_api_keys=["gkey1", "gkey2"],
            together_api_keys=[],
        )

        # 2 models x 2 keys = 4 instances
        assert len(models) == 4
        assert mock_get_model.call_count == 4


def test_get_multiple_instances_uses_config_fallback(google_keys_env):
    """get_multiple_model_instances loads keys from ModelConfig when not provided."""
    with patch("xyber_sdk.model_registry.model_factory.get_model") as mock_get_model:
        mock_get_model.return_value = MagicMock()

        models = get_multiple_model_instances(
            model_names=["GEMINI_2_0_FLASH"],
            # google_api_keys not provided - should use config from env
        )

        assert len(models) == 2
        assert mock_get_model.call_count == 2


# --- get_multiple_model_instances() Edge Cases ---


def test_get_multiple_instances_empty_model_names_raises(clean_env):
    """get_multiple_model_instances raises ValueError for empty model_names list."""
    with pytest.raises(ValueError) as exc_info:
        get_multiple_model_instances(
            model_names=[],
            google_api_keys=["key1"],
            together_api_keys=[],
        )

    assert "No models were successfully initialized" in str(exc_info.value)


def test_get_multiple_instances_no_keys_and_no_env_raises(clean_env):
    """get_multiple_model_instances raises ValueError when no API keys available."""
    with pytest.raises(ValueError) as exc_info:
        get_multiple_model_instances(
            model_names=["GEMINI_2_0_FLASH"],
            # No keys provided, and clean_env has no env vars
        )

    assert "No models were successfully initialized" in str(exc_info.value)


def test_get_multiple_instances_skips_missing_provider_keys(clean_env):
    """get_multiple_model_instances creates 0 instances when no keys for provider."""
    with pytest.raises(ValueError) as exc_info:
        get_multiple_model_instances(
            model_names=["GEMINI_2_0_FLASH"],
            google_api_keys=[],  # No keys for Google
            together_api_keys=["tkey1"],  # Keys for Together, but no Together models
        )

    assert "No models were successfully initialized" in str(exc_info.value)


def test_get_multiple_instances_continues_on_single_failure(clean_env):
    """get_multiple_model_instances continues when one key fails."""
    with patch("xyber_sdk.model_registry.model_factory.get_model") as mock_get_model:
        # First call raises, second succeeds
        mock_get_model.side_effect = [
            ValueError("bad key"),
            MagicMock(spec=ChatGoogleGenerativeAI),
        ]

        models = get_multiple_model_instances(
            model_names=["GEMINI_2_0_FLASH"],
            google_api_keys=["bad-key", "good-key"],
            together_api_keys=[],
        )

        assert len(models) == 1


# --- get_multiple_model_instances() Bad Input Tests ---


def test_get_multiple_instances_invalid_model_name_skipped(clean_env):
    """get_multiple_model_instances logs error and skips invalid model names."""
    with patch("xyber_sdk.model_registry.model_factory.get_model") as mock_get_model:
        mock_get_model.return_value = MagicMock()

        models = get_multiple_model_instances(
            model_names=["GEMINI_2_0_FLASH", "NONEXISTENT_MODEL"],
            google_api_keys=["gkey1"],
            together_api_keys=[],
        )

        # Only the valid model should be created
        assert len(models) == 1
        mock_get_model.assert_called_once()


def test_get_multiple_instances_all_invalid_models_raises(clean_env):
    """get_multiple_model_instances raises ValueError when all model names invalid."""
    with pytest.raises(ValueError) as exc_info:
        get_multiple_model_instances(
            model_names=["INVALID_1", "INVALID_2"],
            google_api_keys=["key1"],
            together_api_keys=["key2"],
        )

    assert "No models were successfully initialized" in str(exc_info.value)


def test_get_multiple_instances_all_keys_fail_raises(clean_env):
    """get_multiple_model_instances raises ValueError when all key initializations fail."""
    with patch("xyber_sdk.model_registry.model_factory.get_model") as mock_get_model:
        mock_get_model.side_effect = ValueError("all fail")

        with pytest.raises(ValueError) as exc_info:
            get_multiple_model_instances(
                model_names=["GEMINI_2_0_FLASH"],
                google_api_keys=["bad-key1", "bad-key2"],
                together_api_keys=[],
            )

        assert "No models were successfully initialized" in str(exc_info.value)
