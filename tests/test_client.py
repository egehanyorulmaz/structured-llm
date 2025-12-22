"""Tests for the StructuredLLMClient."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from structured_llm import (
    StructuredLLMClient,
    ClientConfig,
    ProviderConfig,
    ProviderType,
    RetryConfig,
    ConfigurationError,
)
from structured_llm.providers import OpenAIProvider


class TestStructuredLLMClientInitialization:
    """Tests for client initialization."""

    def test_init_with_string_provider(self):
        """Test initialization with string provider name."""
        with patch.object(OpenAIProvider, '_get_api_key', return_value='test-key'):
            client = StructuredLLMClient(provider="openai")
            assert client.config.primary_provider.provider_type == ProviderType.OPENAI

    def test_init_with_enum_provider(self):
        """Test initialization with ProviderType enum."""
        with patch.object(OpenAIProvider, '_get_api_key', return_value='test-key'):
            client = StructuredLLMClient(provider=ProviderType.OPENAI)
            assert client.config.primary_provider.provider_type == ProviderType.OPENAI

    def test_init_with_invalid_provider(self):
        """Test initialization with invalid provider raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            StructuredLLMClient(provider="invalid_provider")
        assert "Unknown provider" in str(exc_info.value)

    def test_init_without_provider_or_config(self):
        """Test initialization without provider or config raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            StructuredLLMClient()
        assert "Either 'provider' or 'config' must be provided" in str(exc_info.value)

    def test_init_with_full_config(self):
        """Test initialization with full ClientConfig."""
        config = ClientConfig(
            primary_provider=ProviderConfig(
                provider_type=ProviderType.OPENAI,
                api_key="test-key",
                model="gpt-4o",
            ),
            retry_config=RetryConfig(max_retries=5),
        )
        
        with patch.object(OpenAIProvider, '_get_api_key', return_value='test-key'):
            client = StructuredLLMClient(config=config)
            assert client.config.primary_provider.model == "gpt-4o"
            assert client.config.retry_config.max_retries == 5

    def test_init_with_fallback_providers(self):
        """Test initialization with fallback providers."""
        with patch.object(OpenAIProvider, '_get_api_key', return_value='test-key'):
            with patch('structured_llm.providers.anthropic.AnthropicProvider._get_api_key', return_value='test-key'):
                client = StructuredLLMClient(
                    provider="openai",
                    fallback_providers=["anthropic"],
                )
                assert len(client.config.fallback_providers) == 1
                assert client.config.fallback_providers[0].provider_type == ProviderType.ANTHROPIC


class TestStructuredLLMClientConfig:
    """Tests for configuration handling."""

    def test_default_temperature(self):
        """Test default temperature is set correctly."""
        with patch.object(OpenAIProvider, '_get_api_key', return_value='test-key'):
            client = StructuredLLMClient(provider="openai")
            assert client.config.primary_provider.temperature == 0.1

    def test_custom_temperature(self):
        """Test custom temperature is respected."""
        with patch.object(OpenAIProvider, '_get_api_key', return_value='test-key'):
            client = StructuredLLMClient(provider="openai", temperature=0.7)
            assert client.config.primary_provider.temperature == 0.7

    def test_custom_model(self):
        """Test custom model is respected."""
        with patch.object(OpenAIProvider, '_get_api_key', return_value='test-key'):
            client = StructuredLLMClient(provider="openai", model="gpt-4-turbo")
            assert client.config.primary_provider.model == "gpt-4-turbo"


class TestProviderTypeNormalization:
    """Tests for provider type normalization."""

    def test_lowercase_string(self):
        """Test lowercase string normalization."""
        with patch.object(OpenAIProvider, '_get_api_key', return_value='test-key'):
            client = StructuredLLMClient(provider="openai")
            assert client.config.primary_provider.provider_type == ProviderType.OPENAI

    def test_mixed_case_string_works(self):
        """Test mixed case string normalization works."""
        with patch.object(OpenAIProvider, '_get_api_key', return_value='test-key'):
            # ProviderType enum handles case conversion internally
            client = StructuredLLMClient(provider="OpenAI")
            assert client.config.primary_provider.provider_type == ProviderType.OPENAI

