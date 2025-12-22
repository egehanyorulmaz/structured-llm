"""Main client for structured LLM completions."""

import logging
from typing import Type, TypeVar, Optional, List, Union

from pydantic import BaseModel

from structured_llm.config import (
    ClientConfig,
    ProviderConfig,
    ProviderType,
    RetryConfig,
)
from structured_llm.exceptions import ConfigurationError
from structured_llm.fallback import FallbackManager, ProviderFactory

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class StructuredLLMClient:
    """
    Main client for generating structured LLM completions.

    This client provides a unified interface for interacting with various LLM providers
    (OpenAI, Anthropic, Gemini) and getting structured outputs validated against
    Pydantic models.

    Features:
    - Automatic Pydantic validation with retries
    - Provider fallback on failures
    - Both sync and async support

    Example usage::

        from structured_llm import StructuredLLMClient
        from pydantic import BaseModel

        class RecipeTags(BaseModel):
            tags: list[str]
            confidence: float

        # Simple initialization with string provider
        client = StructuredLLMClient(provider="openai")

        # Sync usage
        result = client.complete(
            response_model=RecipeTags,
            user_prompt="Tag this recipe: Pasta Carbonara",
            system_prompt="You are a recipe tagger"
        )

        # Async usage
        result = await client.acomplete(
            response_model=RecipeTags,
            user_prompt="Tag this recipe: Pasta Carbonara"
        )
    """

    def __init__(
        self,
        provider: Optional[Union[str, ProviderType]] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        fallback_providers: Optional[List[Union[str, ProviderType]]] = None,
        max_retries: int = 3,
        config: Optional[ClientConfig] = None,
    ):
        """
        Initialize the structured LLM client.

        You can either pass a full ClientConfig object or use the simplified parameters.

        :param provider: Primary provider to use (e.g., "openai", "anthropic", "gemini")
        :type provider: Optional[Union[str, ProviderType]]
        :param api_key: API key for the provider (uses env vars if not provided)
        :type api_key: Optional[str]
        :param model: Model name to use (uses provider defaults if not provided)
        :type model: Optional[str]
        :param temperature: Sampling temperature (0.0 to 2.0)
        :type temperature: float
        :param max_tokens: Maximum tokens in the response
        :type max_tokens: Optional[int]
        :param fallback_providers: List of fallback providers to try on failure
        :type fallback_providers: Optional[List[Union[str, ProviderType]]]
        :param max_retries: Maximum retries for validation failures per provider
        :type max_retries: int
        :param config: Full client configuration (overrides other parameters)
        :type config: Optional[ClientConfig]
        :raises ConfigurationError: If configuration is invalid
        """
        if config is not None:
            self._config = config
        elif provider is not None:
            self._config = self._build_config_from_params(
                provider=provider,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                fallback_providers=fallback_providers,
                max_retries=max_retries,
            )
        else:
            raise ConfigurationError(
                "Either 'provider' or 'config' must be provided"
            )

        self._fallback_manager = FallbackManager(
            provider_configs=self._config.get_all_providers(),
            retry_config=self._config.retry_config,
        )

    def _normalize_provider_type(
        self,
        provider: Union[str, ProviderType],
    ) -> ProviderType:
        """
        Normalize a provider string or enum to ProviderType.

        :param provider: Provider as string or enum
        :type provider: Union[str, ProviderType]
        :return: Normalized ProviderType
        :rtype: ProviderType
        :raises ConfigurationError: If provider is not recognized
        """
        if isinstance(provider, ProviderType):
            return provider

        provider_lower = provider.lower()
        try:
            return ProviderType(provider_lower)
        except ValueError:
            valid_providers = [p.value for p in ProviderType]
            raise ConfigurationError(
                f"Unknown provider: '{provider}'. "
                f"Valid providers are: {valid_providers}"
            )

    def _build_config_from_params(
        self,
        provider: Union[str, ProviderType],
        api_key: Optional[str],
        model: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        fallback_providers: Optional[List[Union[str, ProviderType]]],
        max_retries: int,
    ) -> ClientConfig:
        """
        Build ClientConfig from individual parameters.

        :param provider: Primary provider
        :param api_key: API key
        :param model: Model name
        :param temperature: Temperature
        :param max_tokens: Max tokens
        :param fallback_providers: Fallback providers
        :param max_retries: Max retries
        :return: ClientConfig instance
        """
        primary_config = ProviderConfig(
            provider_type=self._normalize_provider_type(provider),
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        fallback_configs = []
        if fallback_providers:
            for fb_provider in fallback_providers:
                fb_config = ProviderConfig(
                    provider_type=self._normalize_provider_type(fb_provider),
                    api_key=None,
                    model=None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                fallback_configs.append(fb_config)

        retry_config = RetryConfig(max_retries=max_retries)

        return ClientConfig(
            primary_provider=primary_config,
            fallback_providers=fallback_configs,
            retry_config=retry_config,
            enable_fallback=bool(fallback_providers),
        )

    def complete(
        self,
        response_model: Type[T],
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> T:
        """
        Generate a structured completion synchronously.

        :param response_model: Pydantic model class for the expected response
        :type response_model: Type[T]
        :param user_prompt: The user's prompt/query
        :type user_prompt: str
        :param system_prompt: Optional system prompt to set context
        :type system_prompt: Optional[str]
        :param max_retries: Override max retries for this call
        :type max_retries: Optional[int]
        :return: Instance of the response model
        :rtype: T
        :raises AllProvidersFailedError: If all providers fail
        :raises ValidationError: If response cannot be validated
        """
        retries = max_retries or self._config.retry_config.max_retries

        return self._fallback_manager.complete_with_fallback(
            response_model=response_model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_retries=retries,
        )

    async def acomplete(
        self,
        response_model: Type[T],
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> T:
        """
        Generate a structured completion asynchronously.

        :param response_model: Pydantic model class for the expected response
        :type response_model: Type[T]
        :param user_prompt: The user's prompt/query
        :type user_prompt: str
        :param system_prompt: Optional system prompt to set context
        :type system_prompt: Optional[str]
        :param max_retries: Override max retries for this call
        :type max_retries: Optional[int]
        :return: Instance of the response model
        :rtype: T
        :raises AllProvidersFailedError: If all providers fail
        :raises ValidationError: If response cannot be validated
        """
        retries = max_retries or self._config.retry_config.max_retries

        return await self._fallback_manager.acomplete_with_fallback(
            response_model=response_model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_retries=retries,
        )

    @property
    def config(self) -> ClientConfig:
        """
        Get the current client configuration.

        :return: Client configuration
        :rtype: ClientConfig
        """
        return self._config

