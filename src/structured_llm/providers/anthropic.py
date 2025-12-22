"""Anthropic provider implementation using instructor."""

import os
from typing import Type, TypeVar, Optional

import instructor
from anthropic import Anthropic, AsyncAnthropic
from pydantic import BaseModel

from structured_llm.config import ProviderConfig
from structured_llm.exceptions import ProviderError, ConfigurationError
from structured_llm.providers.base import BaseProvider

T = TypeVar("T", bound=BaseModel)


class AnthropicProvider(BaseProvider):
    """
    Anthropic provider implementation using the instructor library.

    This provider supports both synchronous and asynchronous operations
    with automatic Pydantic validation and retry logic.
    """

    @property
    def provider_name(self) -> str:
        """
        Get the name of this provider.

        :return: Provider name
        :rtype: str
        """
        return "anthropic"

    def _get_api_key(self) -> str:
        """
        Get the API key from config or environment.

        :return: API key
        :rtype: str
        :raises ConfigurationError: If no API key is found
        """
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key in the provider configuration."
            )
        return api_key

    def _initialize_sync_client(self) -> None:
        """Initialize the synchronous client with instructor patching."""
        if self._sync_client is None:
            api_key = self._get_api_key()
            base_client = Anthropic(
                api_key=api_key,
                timeout=self.config.timeout_seconds,
            )
            self._sync_client = instructor.from_anthropic(base_client)

    def _initialize_async_client(self) -> None:
        """Initialize the asynchronous client with instructor patching."""
        if self._async_client is None:
            api_key = self._get_api_key()
            base_client = AsyncAnthropic(
                api_key=api_key,
                timeout=self.config.timeout_seconds,
            )
            self._async_client = instructor.from_anthropic(base_client)

    def _build_messages_anthropic(
        self,
        user_prompt: str,
    ) -> list[dict]:
        """
        Build the messages list for Anthropic API (system is separate).

        :param user_prompt: The user's prompt/query
        :type user_prompt: str
        :return: List of message dictionaries (user messages only)
        :rtype: list[dict]
        """
        return [{"role": "user", "content": user_prompt}]

    def complete(
        self,
        response_model: Type[T],
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> T:
        """
        Generate a structured completion synchronously.

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
        :raises ProviderError: If the API call fails
        """
        self._initialize_sync_client()
        messages = self._build_messages_anthropic(user_prompt)

        try:
            kwargs = {
                "model": self.config.get_default_model(),
                "messages": messages,
                "response_model": response_model,
                "max_retries": max_retries,
                "max_tokens": self.config.max_tokens or 4096,
                "temperature": self.config.temperature,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = self._sync_client.messages.create(**kwargs)
            return response
        except Exception as e:
            raise ProviderError(
                message=f"Failed to generate completion: {str(e)}",
                provider_name=self.provider_name,
                cause=e,
            )

    async def acomplete(
        self,
        response_model: Type[T],
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> T:
        """
        Generate a structured completion asynchronously.

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
        :raises ProviderError: If the API call fails
        """
        self._initialize_async_client()
        messages = self._build_messages_anthropic(user_prompt)

        try:
            kwargs = {
                "model": self.config.get_default_model(),
                "messages": messages,
                "response_model": response_model,
                "max_retries": max_retries,
                "max_tokens": self.config.max_tokens or 4096,
                "temperature": self.config.temperature,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = await self._async_client.messages.create(**kwargs)
            return response
        except Exception as e:
            raise ProviderError(
                message=f"Failed to generate completion: {str(e)}",
                provider_name=self.provider_name,
                cause=e,
            )

