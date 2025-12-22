"""
Structured LLM - A unified interface for structured LLM outputs.

This package provides a simple way to get structured outputs from various LLM providers
(OpenAI, Anthropic, Gemini) with automatic Pydantic validation and retry logic.

Example usage::

    from structured_llm import StructuredLLMClient
    from pydantic import BaseModel

    class RecipeTags(BaseModel):
        tags: list[str]
        confidence: float

    # Simple initialization
    client = StructuredLLMClient(provider="openai")

    # Get structured output
    result = client.complete(
        response_model=RecipeTags,
        user_prompt="Tag this recipe: Pasta Carbonara",
        system_prompt="You are a recipe tagger"
    )

    # With fallback providers
    client = StructuredLLMClient(
        provider="openai",
        fallback_providers=["anthropic", "gemini"],
    )
"""

from structured_llm.client import StructuredLLMClient
from structured_llm.config import (
    ClientConfig,
    ProviderConfig,
    ProviderType,
    RetryConfig,
)
from structured_llm.exceptions import (
    StructuredLLMError,
    ProviderError,
    ValidationError,
    ConfigurationError,
    AllProvidersFailedError,
    RateLimitError,
)
from structured_llm.providers import (
    BaseProvider,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
)
from structured_llm.fallback import FallbackManager, ProviderFactory

__version__ = "0.1.0"

__all__ = [
    # Main client
    "StructuredLLMClient",
    # Configuration
    "ClientConfig",
    "ProviderConfig",
    "ProviderType",
    "RetryConfig",
    # Exceptions
    "StructuredLLMError",
    "ProviderError",
    "ValidationError",
    "ConfigurationError",
    "AllProvidersFailedError",
    "RateLimitError",
    # Providers
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    # Fallback
    "FallbackManager",
    "ProviderFactory",
]
