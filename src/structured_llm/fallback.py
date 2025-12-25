"""Fallback mechanism for provider switching on failures."""

import logging
from typing import Type, TypeVar, List, Optional, Callable, Awaitable

from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from structured_llm.config import ProviderConfig, RetryConfig, ProviderType
from structured_llm.exceptions import (
    ProviderError,
    AllProvidersFailedError,
    RateLimitError,
)
from structured_llm.providers.base import BaseProvider
from structured_llm.providers.openai import OpenAIProvider
from structured_llm.providers.anthropic import AnthropicProvider
from structured_llm.providers.gemini import GeminiProvider
from structured_llm.providers.vertexai import VertexAIProvider

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating provider instances."""

    _provider_map = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.GEMINI: GeminiProvider,
        ProviderType.VERTEXAI: VertexAIProvider,
    }

    @classmethod
    def create(cls, config: ProviderConfig) -> BaseProvider:
        """
        Create a provider instance from configuration.

        :param config: Provider configuration
        :type config: ProviderConfig
        :return: Provider instance
        :rtype: BaseProvider
        :raises ValueError: If provider type is not supported
        """
        provider_class = cls._provider_map.get(config.provider_type)
        if provider_class is None:
            raise ValueError(f"Unsupported provider type: {config.provider_type}")
        return provider_class(config)


class FallbackManager:
    """
    Manages provider fallback chain with retry logic.

    This class handles attempting completions across multiple providers,
    falling back to the next provider when one fails.
    """

    def __init__(
        self,
        provider_configs: List[ProviderConfig],
        retry_config: RetryConfig,
    ):
        """
        Initialize the fallback manager.

        :param provider_configs: List of provider configurations in order of preference
        :type provider_configs: List[ProviderConfig]
        :param retry_config: Configuration for retry behavior
        :type retry_config: RetryConfig
        """
        self.provider_configs = provider_configs
        self.retry_config = retry_config
        self._providers: List[BaseProvider] = []
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """
        Initialize provider instances from configurations.
        
        Providers that fail to initialize are skipped and logged.
        
        :raises AllProvidersFailedError: If no providers successfully initialize
        """
        for config in self.provider_configs:
            try:
                provider = ProviderFactory.create(config)
                self._providers.append(provider)
                logger.info(f"Successfully initialized provider: {config.provider_type.value}")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize provider {config.provider_type.value}: {e}. "
                    f"Skipping this provider."
                )
                continue
        
        if not self._providers:
            raise AllProvidersFailedError(
                "No providers could be initialized. Check your API keys and configuration."
            )

    def _create_retry_decorator(self):
        """
        Create a tenacity retry decorator based on configuration.

        :return: Configured retry decorator
        """
        return retry(
            stop=stop_after_attempt(self.retry_config.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self.retry_config.min_wait_seconds,
                max=self.retry_config.max_wait_seconds,
                exp_base=self.retry_config.exponential_base,
            ),
            retry=retry_if_exception_type(RateLimitError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    def complete_with_fallback(
        self,
        response_model: Type[T],
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> T:
        """
        Attempt completion with fallback across providers.

        :param response_model: Pydantic model class for the expected response
        :type response_model: Type[T]
        :param user_prompt: The user's prompt/query
        :type user_prompt: str
        :param system_prompt: Optional system prompt to set context
        :type system_prompt: Optional[str]
        :param max_retries: Maximum number of retries for validation failures
        :type max_retries: int
        :return: Instance of the response model
        :rtype: T
        :raises AllProvidersFailedError: If all providers fail
        """
        provider_errors: List[ProviderError] = []

        for provider in self._providers:
            try:
                logger.info(f"Using provider: {provider.provider_name}")
                result = provider.complete(
                    response_model=response_model,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_retries=max_retries,
                )
                return result
            except ProviderError as e:
                logger.warning(
                    f"Provider {provider.provider_name} failed: {e}. "
                    f"Attempting next provider..."
                )
                provider_errors.append(e)
                continue

        raise AllProvidersFailedError(
            message="All providers failed to generate a valid response",
            provider_errors=provider_errors,
        )

    async def acomplete_with_fallback(
        self,
        response_model: Type[T],
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> T:
        """
        Attempt async completion with fallback across providers.

        :param response_model: Pydantic model class for the expected response
        :type response_model: Type[T]
        :param user_prompt: The user's prompt/query
        :type user_prompt: str
        :param system_prompt: Optional system prompt to set context
        :type system_prompt: Optional[str]
        :param max_retries: Maximum number of retries for validation failures
        :type max_retries: int
        :return: Instance of the response model
        :rtype: T
        :raises AllProvidersFailedError: If all providers fail
        """
        provider_errors: List[ProviderError] = []

        for provider in self._providers:
            try:
                logger.info(f"Using provider: {provider.provider_name}")
                result = await provider.acomplete(
                    response_model=response_model,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_retries=max_retries,
                )
                return result
            except ProviderError as e:
                logger.warning(
                    f"Provider {provider.provider_name} failed: {e}. "
                    f"Attempting next provider..."
                )
                provider_errors.append(e)
                continue

        raise AllProvidersFailedError(
            message="All providers failed to generate a valid response",
            provider_errors=provider_errors,
        )

