# Model Registry

Initialize and manage LLMs from Google and Together AI.

## Prerequisites

- [langchain-google-genai](https://pypi.org/project/langchain-google-genai/) - Google Gemini models
- [langchain-together](https://pypi.org/project/langchain-together/) - Together AI models

## Public API

```python
from xyber_sdk.model_registry import (
    get_model,
    get_multiple_model_instances,
    ModelConfig,
    SupportedModels,
    SupportedGoogleModels,
    SupportedTogetherModels,
)
```
## Configuration

### Environment Variables

```bash
# Option: Multiple keys (instantiate ModelConfig)
GOOGLE_API_KEYS='["key1","key2","key3"]'
TOGETHER_API_KEYS='["key1","key2"]'

# Option: Single keys (Used inexplicitly)
GOOGLE_API_KEYS=key1
TOGETHER_API_KEYS=key1
```

### Loading Config Programmatically

```python
from xyber_sdk.model_registry import ModelConfig

config = ModelConfig()
# Reads GOOGLE_API_KEYS and TOGETHER_API_KEYS from environment

print(config.google_api_keys)    # ['key1', 'key2', 'key3']
print(config.together_api_keys)  # ['key1', 'key2']
```

## Usage Scenarios

### Scenario 1: Single Model

```python
from xyber_sdk.model_registry import get_model, SupportedModels

# Option A: With explicit API key
llm = get_model(SupportedModels.GEMINI_2_0_FLASH, google_api_key="your-key")

# Option B: With env var (GOOGLE_API_KEY or TOGETHER_API_KEY inexplicitly)
llm = get_model(SupportedModels.META_LLAMA_3_3_70B)

# Option C: Using ModelConfig
from xyber_sdk.model_registry import ModelConfig
config = ModelConfig()
llm = get_model(SupportedModels.GEMINI_2_0_FLASH, google_api_key=config.google_api_keys[0])

# Use it
response = await llm.ainvoke([HumanMessage(content="Hello")])
```

### Scenario 2: Multiple Models

Use this when you need parallel processing or API key rotation:

```python
from xyber_sdk.model_registry import get_multiple_model_instances, ModelConfig

# Option A: Using ModelConfig (recommended)
config = ModelConfig()
models = get_multiple_model_instances(
    model_names=["GEMINI_2_0_FLASH", "META_LLAMA_3_3_70B"],
    google_api_keys=config.google_api_keys,
    together_api_keys=config.together_api_keys,
)

# Option B: With explicit keys
models = get_multiple_model_instances(
    model_names=["GEMINI_2_0_FLASH", "DEEPSEEK_V3"],
    google_api_keys=["key1", "key2"],
    together_api_keys=["key3"],
)

# Result: list of LLM instances (one per model per API key)
```
