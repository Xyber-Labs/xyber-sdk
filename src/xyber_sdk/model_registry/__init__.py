"""
Model Registry - Initialize LLMs from Google and Together AI.

Environment Variables
---------------------
For single API key (recommended for simple usage):
    GOOGLE_API_KEY : str
        Google AI API key for Gemini models
    TOGETHER_API_KEY : str
        Together AI API key for Llama, DeepSeek, Qwen models

For multiple API keys (key rotation / parallel processing):
    GOOGLE_API_KEYS : str
        JSON array of keys, e.g. '["key1","key2"]'
    TOGETHER_API_KEYS : str
        JSON array of keys

Quick Start
-----------
>>> from xyber_sdk.model_registry import get_model, SupportedModels
>>> llm = get_model(SupportedModels.GEMINI_2_0_FLASH)
>>> response = await llm.ainvoke("Hello!")

With explicit key:
>>> llm = get_model(SupportedModels.GEMINI_2_0_FLASH, google_api_key="your-key")
"""
from xyber_sdk.model_registry.chat_models import (
    SupportedGoogleModels,
    SupportedModels,
    SupportedTogetherModels,
)
from xyber_sdk.model_registry.config import ModelConfig
from xyber_sdk.model_registry.model_factory import (
    get_model,
    get_multiple_model_instances,
)

__all__ = [
    "ModelConfig",
    "get_model",
    "get_multiple_model_instances",
    "SupportedModels",
    "SupportedGoogleModels",
    "SupportedTogetherModels",
]
