"""Tests for configuration models."""

import pytest
from pydantic import ValidationError

from structured_llm.config import (
    ProviderType,
    RetryConfig,
    ProviderConfig,
    ClientConfig,
)


class TestRetryConfig:
    """Tests for RetryConfig model."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.min_wait_seconds == 1.0
        assert config.max_wait_seconds == 60.0
        assert config.exponential_base == 2.0

    def test_custom_values(self):
        """Test custom values are accepted."""
        config = RetryConfig(
            max_retries=5,
            min_wait_seconds=0.5,
            max_wait_seconds=120.0,
            exponential_base=3.0,
        )
        assert config.max_retries == 5
        assert config.min_wait_seconds == 0.5
        assert config.max_wait_seconds == 120.0
        assert config.exponential_base == 3.0

    def test_max_retries_bounds(self):
        """Test max_retries bounds validation."""
        with pytest.raises(ValidationError):
            RetryConfig(max_retries=-1)
        with pytest.raises(ValidationError):
            RetryConfig(max_retries=11)

    def test_max_wait_must_exceed_min(self):
        """Test max_wait_seconds must be >= min_wait_seconds."""
        with pytest.raises(ValidationError):
            RetryConfig(min_wait_seconds=10.0, max_wait_seconds=5.0)


class TestProviderConfig:
    """Tests for ProviderConfig model."""

    def test_default_model_openai(self):
        """Test default model for OpenAI."""
        config = ProviderConfig(provider_type=ProviderType.OPENAI)
        assert config.get_default_model() == "gpt-4o"

    def test_default_model_anthropic(self):
        """Test default model for Anthropic."""
        config = ProviderConfig(provider_type=ProviderType.ANTHROPIC)
        assert config.get_default_model() == "claude-sonnet-4-20250514"

    def test_default_model_gemini(self):
        """Test default model for Gemini."""
        config = ProviderConfig(provider_type=ProviderType.GEMINI)
        assert config.get_default_model() == "gemini-2.0-flash"

    def test_custom_model_override(self):
        """Test custom model overrides default."""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            model="gpt-4-turbo",
        )
        assert config.get_default_model() == "gpt-4-turbo"

    def test_temperature_bounds(self):
        """Test temperature bounds validation."""
        with pytest.raises(ValidationError):
            ProviderConfig(provider_type=ProviderType.OPENAI, temperature=-0.1)
        with pytest.raises(ValidationError):
            ProviderConfig(provider_type=ProviderType.OPENAI, temperature=2.1)


class TestClientConfig:
    """Tests for ClientConfig model."""

    def test_get_all_providers_with_fallback_enabled(self):
        """Test get_all_providers returns all when fallback is enabled."""
        config = ClientConfig(
            primary_provider=ProviderConfig(provider_type=ProviderType.OPENAI),
            fallback_providers=[
                ProviderConfig(provider_type=ProviderType.ANTHROPIC),
            ],
            enable_fallback=True,
        )
        providers = config.get_all_providers()
        assert len(providers) == 2
        assert providers[0].provider_type == ProviderType.OPENAI
        assert providers[1].provider_type == ProviderType.ANTHROPIC

    def test_get_all_providers_with_fallback_disabled(self):
        """Test get_all_providers returns only primary when fallback is disabled."""
        config = ClientConfig(
            primary_provider=ProviderConfig(provider_type=ProviderType.OPENAI),
            fallback_providers=[
                ProviderConfig(provider_type=ProviderType.ANTHROPIC),
            ],
            enable_fallback=False,
        )
        providers = config.get_all_providers()
        assert len(providers) == 1
        assert providers[0].provider_type == ProviderType.OPENAI

