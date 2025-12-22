"""Configuration models for the structured-llm package."""

from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator


class ProviderType(str, Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class RetryConfig(BaseModel):
    """
    Configuration for retry behavior.

    :param max_retries: Maximum number of retry attempts for validation failures
    :param min_wait_seconds: Minimum wait time between retries in seconds
    :param max_wait_seconds: Maximum wait time between retries in seconds
    :param exponential_base: Base for exponential backoff calculation
    """

    max_retries: int = Field(default=3, ge=0, le=10)
    min_wait_seconds: float = Field(default=1.0, ge=0.1)
    max_wait_seconds: float = Field(default=60.0, ge=1.0)
    exponential_base: float = Field(default=2.0, ge=1.0)

    @field_validator("max_wait_seconds")
    @classmethod
    def max_wait_must_exceed_min(cls, v: float, info) -> float:
        """Ensure max_wait_seconds is greater than min_wait_seconds."""
        min_wait = info.data.get("min_wait_seconds", 1.0)
        if v < min_wait:
            raise ValueError("max_wait_seconds must be >= min_wait_seconds")
        return v


class ProviderConfig(BaseModel):
    """
    Configuration for a single LLM provider.

    :param provider_type: The type of LLM provider
    :param api_key: API key for the provider (can also be set via environment variables)
    :param model: Model name to use
    :param temperature: Sampling temperature (0.0 to 2.0)
    :param max_tokens: Maximum tokens in the response
    :param timeout_seconds: Request timeout in seconds
    """

    provider_type: ProviderType
    api_key: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    timeout_seconds: float = Field(default=120.0, ge=1.0)

    def get_default_model(self) -> str:
        """
        Get the default model for this provider type.

        :return: Default model name for the provider
        :rtype: str
        """
        defaults = {
            ProviderType.OPENAI: "gpt-4o",
            ProviderType.ANTHROPIC: "claude-sonnet-4-20250514",
            ProviderType.GEMINI: "gemini-2.0-flash",
        }
        return self.model or defaults[self.provider_type]


class ClientConfig(BaseModel):
    """
    Main configuration for the StructuredLLMClient.

    :param primary_provider: Configuration for the primary provider
    :param fallback_providers: List of fallback provider configurations
    :param retry_config: Configuration for retry behavior
    :param enable_fallback: Whether to enable fallback to other providers on failure
    """

    primary_provider: ProviderConfig
    fallback_providers: List[ProviderConfig] = Field(default_factory=list)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    enable_fallback: bool = Field(default=True)

    def get_all_providers(self) -> List[ProviderConfig]:
        """
        Get all providers in order of preference.

        :return: List of provider configurations starting with primary
        :rtype: List[ProviderConfig]
        """
        if self.enable_fallback:
            return [self.primary_provider] + self.fallback_providers
        return [self.primary_provider]

