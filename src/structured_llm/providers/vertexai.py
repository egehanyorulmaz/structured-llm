"""Vertex AI provider implementation using direct API calls."""

import os
import json
from typing import Type, TypeVar, Optional, Dict, Any

import httpx
from pydantic import BaseModel

from structured_llm.config import ProviderConfig
from structured_llm.exceptions import ProviderError, ConfigurationError
from structured_llm.providers.base import BaseProvider

T = TypeVar("T", bound=BaseModel)


class VertexAIProvider(BaseProvider):
    """
    Google Vertex AI provider implementation using direct API calls.

    This provider supports both synchronous and asynchronous operations
    with automatic Pydantic validation and retry logic.
    It uses the Vertex AI API with API key authentication.
    """

    BASE_URL = "https://aiplatform.googleapis.com/v1"

    @property
    def provider_name(self) -> str:
        """
        Get the name of this provider.

        :return: Provider name
        :rtype: str
        """
        return "vertexai"

    def _get_api_key(self) -> str:
        """
        Get the API key from config or environment.

        :return: API key
        :rtype: str
        :raises ConfigurationError: If no API key is found
        """
        api_key = self.config.api_key or os.getenv("VERTEX_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "Vertex AI API key not found. Set VERTEX_AI_API_KEY or GOOGLE_API_KEY environment "
                "variable or pass api_key in the provider configuration."
            )
        return api_key

    def _get_endpoint_url(self, model: str) -> str:
        """
        Build the endpoint URL for the Vertex AI API.

        :param model: Model name
        :type model: str
        :return: Full endpoint URL with API key
        :rtype: str
        """
        api_key = self._get_api_key()
        return f"{self.BASE_URL}/publishers/google/models/{model}:generateContent?key={api_key}"

    def _build_request_body(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the request body for Vertex AI API.

        :param user_prompt: The user's prompt/query
        :type user_prompt: str
        :param system_prompt: Optional system prompt to set context
        :type system_prompt: Optional[str]
        :param response_schema: Optional JSON schema for structured output
        :type response_schema: Optional[Dict[str, Any]]
        :return: Request body dictionary
        :rtype: Dict[str, Any]
        """
        contents = []
        
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System: {system_prompt}"}]
            })
        
        contents.append({
            "role": "user",
            "parts": [{"text": user_prompt}]
        })

        request_body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
            }
        }

        if self.config.max_tokens:
            request_body["generationConfig"]["maxOutputTokens"] = self.config.max_tokens

        if response_schema:
            request_body["generationConfig"]["responseMimeType"] = "application/json"
            request_body["generationConfig"]["responseSchema"] = response_schema

        return request_body

    def _extract_text_from_response(self, response_data: Dict[str, Any]) -> str:
        """
        Extract text content from Vertex AI API response.

        :param response_data: Response JSON from Vertex AI
        :type response_data: Dict[str, Any]
        :return: Extracted text content
        :rtype: str
        :raises ProviderError: If response format is unexpected
        """
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                raise ProviderError(
                    message="No candidates in response",
                    provider_name=self.provider_name,
                )
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                raise ProviderError(
                    message="No parts in response content",
                    provider_name=self.provider_name,
                )
            
            return parts[0].get("text", "")
        except (KeyError, IndexError) as e:
            raise ProviderError(
                message=f"Unexpected response format: {str(e)}",
                provider_name=self.provider_name,
                cause=e,
            )

    def _pydantic_to_json_schema(self, model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Convert Pydantic model to JSON schema for Vertex AI.

        :param model: Pydantic model class
        :type model: Type[BaseModel]
        :return: JSON schema dictionary
        :rtype: Dict[str, Any]
        """
        schema = model.model_json_schema()
        
        vertex_schema = {
            "type": "object",
            "properties": schema.get("properties", {}),
        }
        
        if "required" in schema:
            vertex_schema["required"] = schema["required"]
        
        return vertex_schema

    def _initialize_sync_client(self) -> None:
        """Initialize the synchronous HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                timeout=self.config.timeout_seconds,
            )

    def _initialize_async_client(self) -> None:
        """Initialize the asynchronous HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
            )

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
        
        model = self.config.get_default_model()
        endpoint_url = self._get_endpoint_url(model)
        response_schema = self._pydantic_to_json_schema(response_model)
        request_body = self._build_request_body(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
        )

        for attempt in range(max_retries):
            try:
                response = self._sync_client.post(
                    endpoint_url,
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    raise ProviderError(
                        message=f"Vertex AI API error (status {response.status_code}): {error_detail}",
                        provider_name=self.provider_name,
                    )
                
                response_data = response.json()
                text_content = self._extract_text_from_response(response_data)
                
                json_data = json.loads(text_content)
                return response_model.model_validate(json_data)
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    continue
                raise ProviderError(
                    message=f"Failed to parse JSON response: {str(e)}",
                    provider_name=self.provider_name,
                    cause=e,
                )
            except Exception as e:
                if attempt < max_retries - 1 and not isinstance(e, ProviderError):
                    continue
                if isinstance(e, ProviderError):
                    raise
                raise ProviderError(
                    message=f"Failed to generate completion: {str(e)}",
                    provider_name=self.provider_name,
                    cause=e,
                )
        
        raise ProviderError(
            message=f"Failed after {max_retries} attempts",
            provider_name=self.provider_name,
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
        
        model = self.config.get_default_model()
        endpoint_url = self._get_endpoint_url(model)
        response_schema = self._pydantic_to_json_schema(response_model)
        request_body = self._build_request_body(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
        )

        for attempt in range(max_retries):
            try:
                response = await self._async_client.post(
                    endpoint_url,
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    raise ProviderError(
                        message=f"Vertex AI API error (status {response.status_code}): {error_detail}",
                        provider_name=self.provider_name,
                    )
                
                response_data = response.json()
                text_content = self._extract_text_from_response(response_data)
                
                json_data = json.loads(text_content)
                return response_model.model_validate(json_data)
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    continue
                raise ProviderError(
                    message=f"Failed to parse JSON response: {str(e)}",
                    provider_name=self.provider_name,
                    cause=e,
                )
            except Exception as e:
                if attempt < max_retries - 1 and not isinstance(e, ProviderError):
                    continue
                if isinstance(e, ProviderError):
                    raise
                raise ProviderError(
                    message=f"Failed to generate completion: {str(e)}",
                    provider_name=self.provider_name,
                    cause=e,
                )
        
        raise ProviderError(
            message=f"Failed after {max_retries} attempts",
            provider_name=self.provider_name,
        )

    def __del__(self):
        """Clean up HTTP clients on deletion."""
        if self._sync_client is not None:
            self._sync_client.close()
        if self._async_client is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._async_client.aclose())
                else:
                    loop.run_until_complete(self._async_client.aclose())
            except Exception:
                pass

