# Structured LLM

A unified interface for structured LLM outputs with Pydantic validation, automatic retries, and multi-provider support.

## Features

- **Structured Outputs**: Get Pydantic-validated responses from LLMs
- **Multi-Provider Support**: OpenAI, Anthropic, and Google Gemini
- **Automatic Retries**: Built-in retry logic for validation failures
- **Provider Fallback**: Automatically switch to backup providers on failures
- **Sync & Async**: Both synchronous and asynchronous APIs

## Installation

```bash
# Using uv
uv add structured-llm

# Or from git
uv add git+https://github.com/egehanyorulmaz/structured-llm
```

## Quick Start

```python
from structured_llm import StructuredLLMClient
from pydantic import BaseModel

# Define your response schema
class RecipeTags(BaseModel):
    tags: list[str]
    confidence: float

# Initialize the client
client = StructuredLLMClient(provider="openai")

# Get structured output
result = client.complete(
    response_model=RecipeTags,
    user_prompt="Tag this recipe: Pasta Carbonara with eggs, pecorino, guanciale",
    system_prompt="You are a recipe tagger. Return relevant tags and your confidence."
)

print(result.tags)  # ['italian', 'pasta', 'quick-meal', 'comfort-food']
print(result.confidence)  # 0.95
```

## Provider Fallback

Configure fallback providers to ensure reliability:

```python
client = StructuredLLMClient(
    provider="openai",
    fallback_providers=["anthropic", "gemini"],
    max_retries=3
)

# If OpenAI fails, automatically tries Anthropic, then Gemini
result = client.complete(
    response_model=RecipeTags,
    user_prompt="Tag this recipe..."
)
```

## Async Support

```python
import asyncio
from structured_llm import StructuredLLMClient

async def main():
    client = StructuredLLMClient(provider="openai")
    
    result = await client.acomplete(
        response_model=RecipeTags,
        user_prompt="Tag this recipe..."
    )
    return result

asyncio.run(main())
```

## Configuration

### Environment Variables

Set API keys via environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"  # or GEMINI_API_KEY
```

### Advanced Configuration

```python
from structured_llm import (
    StructuredLLMClient,
    ClientConfig,
    ProviderConfig,
    ProviderType,
    RetryConfig,
)

# Full configuration
config = ClientConfig(
    primary_provider=ProviderConfig(
        provider_type=ProviderType.OPENAI,
        model="gpt-4o",
        temperature=0.1,
        max_tokens=1000,
    ),
    fallback_providers=[
        ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            model="claude-sonnet-4-20250514",
        ),
    ],
    retry_config=RetryConfig(
        max_retries=3,
        min_wait_seconds=1.0,
        max_wait_seconds=60.0,
    ),
    enable_fallback=True,
)

client = StructuredLLMClient(config=config)
```

## Supported Providers

| Provider | Default Model | Environment Variable |
|----------|---------------|---------------------|
| OpenAI | gpt-4o | `OPENAI_API_KEY` |
| Anthropic | claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY` |
| Gemini | gemini-2.0-flash | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |

## Error Handling

```python
from structured_llm import (
    StructuredLLMClient,
    AllProvidersFailedError,
    ProviderError,
    ConfigurationError,
)

client = StructuredLLMClient(
    provider="openai",
    fallback_providers=["anthropic"],
)

try:
    result = client.complete(
        response_model=RecipeTags,
        user_prompt="Tag this recipe..."
    )
except AllProvidersFailedError as e:
    print(f"All providers failed: {e}")
    for error in e.provider_errors:
        print(f"  - {error.provider_name}: {error.message}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## Development

```bash
# Clone the repository
git clone https://github.com/egehanyorulmaz/structured-llm
cd structured-llm

# Install with dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Run linting
uv run ruff check src/
```

## License

MIT License

