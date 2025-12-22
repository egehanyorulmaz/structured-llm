"""Pytest configuration and fixtures for structured-llm tests."""

import pytest
from pydantic import BaseModel
from typing import List


class SampleResponse(BaseModel):
    """Sample response model for testing."""
    
    tags: List[str]
    confidence: float


class ComplexResponse(BaseModel):
    """Complex response model for testing nested structures."""
    
    title: str
    items: List[str]
    metadata: dict


@pytest.fixture
def sample_response_model():
    """Provide a sample response model for tests."""
    return SampleResponse


@pytest.fixture
def complex_response_model():
    """Provide a complex response model for tests."""
    return ComplexResponse


@pytest.fixture
def sample_user_prompt():
    """Provide a sample user prompt for tests."""
    return "Analyze this text and provide tags"


@pytest.fixture
def sample_system_prompt():
    """Provide a sample system prompt for tests."""
    return "You are a helpful assistant that provides structured outputs."

