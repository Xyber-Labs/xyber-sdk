import os
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsError

from xyber_sdk.model_registry.config import ModelConfig

# --- Fixtures ---


@pytest.fixture
def clean_env():
    """Fixture that provides clean environment (no API key variables)."""
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def valid_google_keys_env():
    """Fixture with valid Google API keys (JSON array format)."""
    env_vars = {
        "GOOGLE_API_KEYS": '["key1","key2","key3"]',
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def single_google_key_env():
    """Fixture with single Google API key (JSON array format)."""
    env_vars = {
        "GOOGLE_API_KEYS": '["single_key"]',
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def valid_together_keys_env():
    """Fixture with valid Together API keys (JSON array format)."""
    env_vars = {
        "TOGETHER_API_KEYS": '["tkey1","tkey2"]',
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def single_together_key_env():
    """Fixture with single Together API key (JSON array format)."""
    env_vars = {
        "TOGETHER_API_KEYS": '["tkey1"]',
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def both_providers_env():
    """Fixture with both Google and Together API keys (JSON array format)."""
    env_vars = {
        "GOOGLE_API_KEYS": '["gkey1","gkey2"]',
        "TOGETHER_API_KEYS": '["tkey1","tkey2","tkey3"]',
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def google_only_env():
    """Fixture with only Google API keys (Together defaults to empty)."""
    env_vars = {
        "GOOGLE_API_KEYS": '["gkey1","gkey2"]',
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def special_chars_keys_env():
    """Fixture with keys containing special characters."""
    env_vars = {
        "GOOGLE_API_KEYS": '["key-with-dashes","key_with_underscores"]',
        "TOGETHER_API_KEYS": '["AIza123_abc-XYZ"]',
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def empty_json_array_env():
    """Fixture with empty JSON array."""
    env_vars = {
        "GOOGLE_API_KEYS": "[]",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def non_json_comma_separated_env():
    """Fixture with non-JSON comma-separated string (invalid for env vars)."""
    env_vars = {
        "GOOGLE_API_KEYS": "key1,key2",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def malformed_json_env():
    """Fixture with malformed JSON (missing closing bracket)."""
    env_vars = {
        "GOOGLE_API_KEYS": '["key1","key2"',
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def json_object_env():
    """Fixture with JSON object instead of array."""
    env_vars = {
        "GOOGLE_API_KEYS": '{"key": "value"}',
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def plain_string_env():
    """Fixture with plain string without JSON format."""
    env_vars = {
        "GOOGLE_API_KEYS": "single-key",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def extra_fields_env():
    """Fixture with extra unrelated environment variables."""
    env_vars = {
        "GOOGLE_API_KEYS": '["gkey"]',
        "UNRELATED_VAR": "should-be-ignored",
        "ANOTHER_RANDOM_VAR": "also-ignored",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


# --- Happy Path Tests (Environment Variables - JSON Format) ---


def test_json_array_parsed_correctly(valid_google_keys_env):
    """ModelConfig parses JSON array from environment variable."""
    config = ModelConfig(_env_file=None)

    assert config.google_api_keys == ["key1", "key2", "key3"]


def test_single_key_json_array(single_google_key_env):
    """ModelConfig handles single key as JSON array."""
    config = ModelConfig(_env_file=None)

    assert config.google_api_keys == ["single_key"]


def test_both_google_and_together_keys_set(both_providers_env):
    """ModelConfig parses both google_api_keys and together_api_keys from JSON arrays."""
    config = ModelConfig(_env_file=None)

    assert config.google_api_keys == ["gkey1", "gkey2"]
    assert config.together_api_keys == ["tkey1", "tkey2", "tkey3"]


def test_only_google_keys_uses_default_for_together(google_only_env):
    """ModelConfig defaults together_api_keys to empty when not set."""
    config = ModelConfig(_env_file=None)

    assert config.google_api_keys == ["gkey1", "gkey2"]
    assert config.together_api_keys == []


def test_only_together_keys_uses_default_for_google(single_together_key_env):
    """ModelConfig defaults google_api_keys to empty when not set."""
    config = ModelConfig(_env_file=None)

    assert config.google_api_keys == []
    assert config.together_api_keys == ["tkey1"]


def test_keys_with_special_characters(special_chars_keys_env):
    """ModelConfig handles keys with special characters in JSON format."""
    config = ModelConfig(_env_file=None)

    assert config.google_api_keys == ["key-with-dashes", "key_with_underscores"]
    assert config.together_api_keys == ["AIza123_abc-XYZ"]


# --- Edge Case Tests ---


def test_no_env_vars_returns_empty_lists(clean_env):
    """ModelConfig defaults to empty lists when no env vars set."""
    config = ModelConfig(_env_file=None)

    assert config.google_api_keys == []
    assert config.together_api_keys == []


def test_empty_json_array_returns_empty_list(empty_json_array_env):
    """ModelConfig returns empty list for empty JSON array."""
    config = ModelConfig(_env_file=None)

    assert config.google_api_keys == []


# --- Direct Constructor Input Tests (Comma-Separated Validator) ---


def test_comma_separated_string_parsed_in_constructor(clean_env):
    """ModelConfig validator parses comma-separated strings from constructor."""
    config = ModelConfig(
        google_api_keys="key1,key2,key3",
        _env_file=None,
    )

    assert config.google_api_keys == ["key1", "key2", "key3"]


def test_single_key_string_in_constructor(clean_env):
    """ModelConfig validator handles single key string from constructor."""
    config = ModelConfig(
        google_api_keys="single_key",
        _env_file=None,
    )

    assert config.google_api_keys == ["single_key"]


def test_whitespace_stripped_in_constructor(clean_env):
    """ModelConfig validator strips whitespace from comma-separated keys."""
    config = ModelConfig(
        google_api_keys="key1, key2 , key3",
        _env_file=None,
    )

    assert config.google_api_keys == ["key1", "key2", "key3"]


def test_empty_string_in_constructor_returns_empty_list(clean_env):
    """ModelConfig validator returns empty list for empty string."""
    config = ModelConfig(
        google_api_keys="",
        _env_file=None,
    )

    assert config.google_api_keys == []


def test_only_commas_in_constructor_returns_empty_list(clean_env):
    """ModelConfig validator returns empty list for string with only commas."""
    config = ModelConfig(
        google_api_keys=",,,",
        _env_file=None,
    )

    assert config.google_api_keys == []


def test_only_whitespace_in_constructor_returns_empty_list(clean_env):
    """ModelConfig validator returns empty list for string with only whitespace."""
    config = ModelConfig(
        google_api_keys="   ",
        _env_file=None,
    )

    assert config.google_api_keys == []


def test_commas_and_whitespace_in_constructor_returns_empty_list(clean_env):
    """ModelConfig validator returns empty list for commas and whitespace."""
    config = ModelConfig(
        google_api_keys=" , , , ",
        _env_file=None,
    )

    assert config.google_api_keys == []


def test_list_input_passed_through(clean_env):
    """ModelConfig validator passes list input through unchanged."""
    config = ModelConfig(
        google_api_keys=["key1", "key2"],
        together_api_keys=["tkey1"],
        _env_file=None,
    )

    assert config.google_api_keys == ["key1", "key2"]
    assert config.together_api_keys == ["tkey1"]


def test_none_input_returns_empty_list(clean_env):
    """ModelConfig validator converts None to empty list."""
    config = ModelConfig(
        google_api_keys=None,
        together_api_keys=None,
        _env_file=None,
    )

    assert config.google_api_keys == []
    assert config.together_api_keys == []


# --- Bad Input Tests (Lenient Validator Behavior) ---


def test_non_string_non_list_returns_empty_list(clean_env):
    """ModelConfig validator returns empty list for unsupported types.

    The validator is lenient and returns [] for invalid input types
    instead of raising errors.
    """
    config = ModelConfig(
        google_api_keys=12345,  # type: ignore
        _env_file=None,
    )

    assert config.google_api_keys == []


# --- Environment Variable Error Cases ---


def test_non_json_env_string_raises_settings_error(non_json_comma_separated_env):
    """ModelConfig raises SettingsError for non-JSON string in env var.

    Pydantic-settings requires JSON format for list[str] fields from environment.
    Comma-separated format only works via direct constructor calls.
    """
    with pytest.raises(SettingsError) as exc_info:
        ModelConfig(_env_file=None)

    assert "google_api_keys" in str(exc_info.value)


def test_malformed_json_raises_settings_error(malformed_json_env):
    """ModelConfig raises SettingsError for malformed JSON in env var."""
    with pytest.raises(SettingsError) as exc_info:
        ModelConfig(_env_file=None)

    assert "google_api_keys" in str(exc_info.value)


def test_non_array_json_returns_empty_list(json_object_env):
    """ModelConfig returns empty list for JSON object (lenient validator).

    Pydantic-settings parses JSON object as dict, which is then passed to
    the validator. The validator returns [] for unsupported types.
    """
    config = ModelConfig(_env_file=None)

    # Validator returns [] for dict (unsupported type)
    assert config.google_api_keys == []


def test_plain_string_without_quotes_raises_settings_error(plain_string_env):
    """ModelConfig raises SettingsError for plain string without JSON format."""
    with pytest.raises(SettingsError) as exc_info:
        ModelConfig(_env_file=None)

    assert "google_api_keys" in str(exc_info.value)


def test_extra_fields_ignored(extra_fields_env):
    """ModelConfig ignores extra environment variables (extra='ignore')."""
    config = ModelConfig(_env_file=None)

    assert config.google_api_keys == ["gkey"]
    assert not hasattr(config, "unrelated_var")
    assert not hasattr(config, "another_random_var")
