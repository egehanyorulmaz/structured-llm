"""Tests for VertexAI provider implementation."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pydantic import BaseModel

from structured_llm.config import ProviderConfig, ProviderType
from structured_llm.exceptions import ProviderError, ConfigurationError
from structured_llm.providers.vertexai import VertexAIProvider


class SampleResponse(BaseModel):
    """Sample response model for testing."""
    
    tags: list[str]
    confidence: float


class TestVertexAIProviderInitialization:
    """Tests for VertexAI provider initialization."""

    def test_provider_name(self):
        """Test provider name is correct."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        assert provider.provider_name == "vertexai"

    def test_get_api_key_from_config(self):
        """Test API key is retrieved from config."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key-from-config",
        )
        provider = VertexAIProvider(config)
        assert provider._get_api_key() == "test-key-from-config"

    def test_get_api_key_from_vertex_ai_env(self):
        """Test API key is retrieved from VERTEX_AI_API_KEY environment variable."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
        )
        provider = VertexAIProvider(config)
        
        with patch.dict("os.environ", {"VERTEX_AI_API_KEY": "test-key-from-env"}):
            assert provider._get_api_key() == "test-key-from-env"

    def test_get_api_key_from_google_env(self):
        """Test API key is retrieved from GOOGLE_API_KEY environment variable."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
        )
        provider = VertexAIProvider(config)
        
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key-from-google-env"}, clear=True):
            assert provider._get_api_key() == "test-key-from-google-env"

    def test_get_api_key_missing_raises_error(self):
        """Test missing API key raises ConfigurationError."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
        )
        provider = VertexAIProvider(config)
        
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                provider._get_api_key()
            assert "API key not found" in str(exc_info.value)


class TestVertexAIProviderURLBuilding:
    """Tests for URL building."""

    def test_get_endpoint_url(self):
        """Test endpoint URL is correctly built."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-api-key",
        )
        provider = VertexAIProvider(config)
        
        url = provider._get_endpoint_url("gemini-2.5-flash-lite")
        
        assert url == (
            "https://aiplatform.googleapis.com/v1/publishers/google/"
            "models/gemini-2.5-flash-lite:generateContent?key=test-api-key"
        )

    def test_get_endpoint_url_with_different_model(self):
        """Test endpoint URL with different model."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="my-key",
        )
        provider = VertexAIProvider(config)
        
        url = provider._get_endpoint_url("gemini-pro")
        
        assert url == (
            "https://aiplatform.googleapis.com/v1/publishers/google/"
            "models/gemini-pro:generateContent?key=my-key"
        )


class TestVertexAIProviderRequestBuilding:
    """Tests for request body building."""

    def test_build_request_body_basic(self):
        """Test basic request body building."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        body = provider._build_request_body(
            user_prompt="What is AI?",
        )
        
        assert "contents" in body
        assert len(body["contents"]) == 1
        assert body["contents"][0]["role"] == "user"
        assert body["contents"][0]["parts"][0]["text"] == "What is AI?"
        assert "generationConfig" in body
        assert body["generationConfig"]["temperature"] == 0.1

    def test_build_request_body_with_system_prompt(self):
        """Test request body with system prompt."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        body = provider._build_request_body(
            user_prompt="What is AI?",
            system_prompt="You are a helpful assistant",
        )
        
        assert len(body["contents"]) == 2
        assert "System: You are a helpful assistant" in body["contents"][0]["parts"][0]["text"]
        assert body["contents"][1]["parts"][0]["text"] == "What is AI?"

    def test_build_request_body_with_max_tokens(self):
        """Test request body with max tokens."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
            max_tokens=1000,
        )
        provider = VertexAIProvider(config)
        
        body = provider._build_request_body(
            user_prompt="What is AI?",
        )
        
        assert body["generationConfig"]["maxOutputTokens"] == 1000

    def test_build_request_body_with_response_schema(self):
        """Test request body with response schema."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
            },
        }
        
        body = provider._build_request_body(
            user_prompt="What is AI?",
            response_schema=schema,
        )
        
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert body["generationConfig"]["responseSchema"] == schema

    def test_build_request_body_with_custom_temperature(self):
        """Test request body with custom temperature."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
            temperature=0.7,
        )
        provider = VertexAIProvider(config)
        
        body = provider._build_request_body(
            user_prompt="What is AI?",
        )
        
        assert body["generationConfig"]["temperature"] == 0.7


class TestVertexAIProviderResponseParsing:
    """Tests for response parsing."""

    def test_extract_text_from_response(self):
        """Test text extraction from response."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        response_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"tags": ["ai", "ml"], "confidence": 0.95}'}
                        ]
                    }
                }
            ]
        }
        
        text = provider._extract_text_from_response(response_data)
        assert text == '{"tags": ["ai", "ml"], "confidence": 0.95}'

    def test_extract_text_from_response_no_candidates(self):
        """Test text extraction with no candidates raises error."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        response_data = {"candidates": []}
        
        with pytest.raises(ProviderError) as exc_info:
            provider._extract_text_from_response(response_data)
        assert "No candidates" in str(exc_info.value)

    def test_extract_text_from_response_no_parts(self):
        """Test text extraction with no parts raises error."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        response_data = {
            "candidates": [
                {"content": {"parts": []}}
            ]
        }
        
        with pytest.raises(ProviderError) as exc_info:
            provider._extract_text_from_response(response_data)
        assert "No parts" in str(exc_info.value)


class TestVertexAIProviderSchemaConversion:
    """Tests for Pydantic to JSON schema conversion."""

    def test_pydantic_to_json_schema(self):
        """Test conversion of Pydantic model to JSON schema."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        schema = provider._pydantic_to_json_schema(SampleResponse)
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "tags" in schema["properties"]
        assert "confidence" in schema["properties"]

    def test_pydantic_to_json_schema_with_required(self):
        """Test schema conversion includes required fields."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        schema = provider._pydantic_to_json_schema(SampleResponse)
        
        if "required" in SampleResponse.model_json_schema():
            assert "required" in schema


class TestVertexAIProviderComplete:
    """Tests for synchronous completion."""

    def test_complete_success(self):
        """Test successful completion."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
            model="gemini-2.5-flash-lite",
        )
        provider = VertexAIProvider(config)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"tags": ["ai", "ml"], "confidence": 0.95}'}
                        ]
                    }
                }
            ]
        }
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        provider._sync_client = mock_client
        
        result = provider.complete(
            response_model=SampleResponse,
            user_prompt="What is AI?",
        )
        
        assert isinstance(result, SampleResponse)
        assert result.tags == ["ai", "ml"]
        assert result.confidence == 0.95

    def test_complete_with_system_prompt(self):
        """Test completion with system prompt."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"tags": ["test"], "confidence": 0.9}'}
                        ]
                    }
                }
            ]
        }
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        provider._sync_client = mock_client
        
        result = provider.complete(
            response_model=SampleResponse,
            user_prompt="Analyze this",
            system_prompt="You are an expert",
        )
        
        assert isinstance(result, SampleResponse)
        call_kwargs = mock_client.post.call_args[1]
        request_body = call_kwargs["json"]
        assert len(request_body["contents"]) == 2

    def test_complete_api_error(self):
        """Test completion with API error."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        provider._sync_client = mock_client
        
        with pytest.raises(ProviderError) as exc_info:
            provider.complete(
                response_model=SampleResponse,
                user_prompt="What is AI?",
            )
        assert "400" in str(exc_info.value)

    def test_complete_json_decode_error_with_retry(self):
        """Test completion with JSON decode error and retry."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        mock_response_bad = Mock()
        mock_response_bad.status_code = 200
        mock_response_bad.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "invalid json"}
                        ]
                    }
                }
            ]
        }
        
        mock_response_good = Mock()
        mock_response_good.status_code = 200
        mock_response_good.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"tags": ["test"], "confidence": 0.8}'}
                        ]
                    }
                }
            ]
        }
        
        mock_client = Mock()
        mock_client.post.side_effect = [mock_response_bad, mock_response_good]
        provider._sync_client = mock_client
        
        result = provider.complete(
            response_model=SampleResponse,
            user_prompt="What is AI?",
            max_retries=2,
        )
        
        assert isinstance(result, SampleResponse)
        assert result.tags == ["test"]
        assert mock_client.post.call_count == 2

    def test_complete_max_retries_exceeded(self):
        """Test completion with max retries exceeded."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "invalid json"}
                        ]
                    }
                }
            ]
        }
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        provider._sync_client = mock_client
        
        with pytest.raises(ProviderError) as exc_info:
            provider.complete(
                response_model=SampleResponse,
                user_prompt="What is AI?",
                max_retries=2,
            )
        assert "Failed to parse JSON" in str(exc_info.value)


class TestVertexAIProviderAsyncComplete:
    """Tests for asynchronous completion."""

    @pytest.mark.asyncio
    async def test_acomplete_success(self):
        """Test successful async completion."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"tags": ["async", "test"], "confidence": 0.88}'}
                        ]
                    }
                }
            ]
        }
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._async_client = mock_client
        
        result = await provider.acomplete(
            response_model=SampleResponse,
            user_prompt="What is async?",
        )
        
        assert isinstance(result, SampleResponse)
        assert result.tags == ["async", "test"]
        assert result.confidence == 0.88

    @pytest.mark.asyncio
    async def test_acomplete_with_retry(self):
        """Test async completion with retry."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        mock_response_bad = Mock()
        mock_response_bad.status_code = 200
        mock_response_bad.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "not json"}
                        ]
                    }
                }
            ]
        }
        
        mock_response_good = Mock()
        mock_response_good.status_code = 200
        mock_response_good.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"tags": ["retry"], "confidence": 0.7}'}
                        ]
                    }
                }
            ]
        }
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_response_bad, mock_response_good])
        provider._async_client = mock_client
        
        result = await provider.acomplete(
            response_model=SampleResponse,
            user_prompt="Test retry",
            max_retries=2,
        )
        
        assert isinstance(result, SampleResponse)
        assert result.tags == ["retry"]
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_acomplete_api_error(self):
        """Test async completion with API error."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        provider = VertexAIProvider(config)
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._async_client = mock_client
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.acomplete(
                response_model=SampleResponse,
                user_prompt="What is AI?",
            )
        assert "500" in str(exc_info.value)


class TestVertexAIProviderIntegration:
    """Integration tests for VertexAI provider."""

    def test_default_model(self):
        """Test default model is set correctly."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
        )
        
        assert config.get_default_model() == "gemini-2.5-flash-lite"

    def test_custom_model(self):
        """Test custom model is used."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
            model="gemini-pro",
        )
        
        assert config.get_default_model() == "gemini-pro"

    def test_custom_temperature(self):
        """Test custom temperature is used."""
        config = ProviderConfig(
            provider_type=ProviderType.VERTEXAI,
            api_key="test-key",
            temperature=0.8,
        )
        provider = VertexAIProvider(config)
        
        body = provider._build_request_body("test")
        assert body["generationConfig"]["temperature"] == 0.8

