"""Base provider protocol and abstract class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Type, TypeVar, List, Optional

from pydantic import BaseModel

from structured_llm.config import ProviderConfig

T = TypeVar("T", bound=BaseModel)


class Message(BaseModel):
    """
    Represents a message in the conversation.

    :param role: The role of the message sender (system, user, assistant)
    :param content: The content of the message
    """

    role: str
    content: str


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.

    All provider implementations must inherit from this class and implement
    the required methods for both sync and async operations.
    """

    def __init__(self, config: ProviderConfig):
        """
        Initialize the provider with configuration.

        :param config: Provider configuration containing API key, model, etc.
        :type config: ProviderConfig
        """
        self.config = config
        self._sync_client = None
        self._async_client = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the name of this provider.

        :return: Provider name
        :rtype: str
        """
        pass

    @abstractmethod
    def _initialize_sync_client(self) -> None:
        """Initialize the synchronous client with instructor patching."""
        pass

    @abstractmethod
    def _initialize_async_client(self) -> None:
        """Initialize the asynchronous client with instructor patching."""
        pass

    def _build_messages(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> List[dict]:
        """
        Build the messages list for the API call.

        :param user_prompt: The user's prompt/query
        :type user_prompt: str
        :param system_prompt: Optional system prompt to set context
        :type system_prompt: Optional[str]
        :return: List of message dictionaries
        :rtype: List[dict]
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    @abstractmethod
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
        :raises ValidationError: If the response cannot be validated
        """
        pass

    @abstractmethod
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
        :raises ValidationError: If the response cannot be validated
        """
        pass

