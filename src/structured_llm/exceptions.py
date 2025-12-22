"""Custom exceptions for the structured-llm package."""

from typing import Optional, List


class StructuredLLMError(Exception):
    """Base exception for all structured-llm errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        """
        Initialize the exception.

        :param message: Error message describing what went wrong
        :type message: str
        :param cause: Optional underlying exception that caused this error
        :type cause: Optional[Exception]
        """
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


class ConfigurationError(StructuredLLMError):
    """Raised when there is a configuration issue."""

    pass


class ProviderError(StructuredLLMError):
    """Raised when a provider-specific error occurs."""

    def __init__(
        self,
        message: str,
        provider_name: str,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize the provider error.

        :param message: Error message describing what went wrong
        :type message: str
        :param provider_name: Name of the provider that caused the error
        :type provider_name: str
        :param cause: Optional underlying exception that caused this error
        :type cause: Optional[Exception]
        """
        super().__init__(message, cause)
        self.provider_name = provider_name

    def __str__(self) -> str:
        base_msg = f"[{self.provider_name}] {self.message}"
        if self.cause:
            return f"{base_msg} (caused by: {self.cause})"
        return base_msg


class ValidationError(StructuredLLMError):
    """Raised when the LLM output fails Pydantic validation."""

    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[dict]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize the validation error.

        :param message: Error message describing the validation failure
        :type message: str
        :param validation_errors: List of Pydantic validation error details
        :type validation_errors: Optional[List[dict]]
        :param cause: Optional underlying exception that caused this error
        :type cause: Optional[Exception]
        """
        super().__init__(message, cause)
        self.validation_errors = validation_errors or []


class RateLimitError(ProviderError):
    """Raised when a rate limit is hit."""

    def __init__(
        self,
        message: str,
        provider_name: str,
        retry_after: Optional[float] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize the rate limit error.

        :param message: Error message describing the rate limit issue
        :type message: str
        :param provider_name: Name of the provider that hit the rate limit
        :type provider_name: str
        :param retry_after: Optional number of seconds to wait before retrying
        :type retry_after: Optional[float]
        :param cause: Optional underlying exception that caused this error
        :type cause: Optional[Exception]
        """
        super().__init__(message, provider_name, cause)
        self.retry_after = retry_after


class AllProvidersFailedError(StructuredLLMError):
    """Raised when all providers in the fallback chain have failed."""

    def __init__(
        self,
        message: str,
        provider_errors: Optional[List[ProviderError]] = None,
    ):
        """
        Initialize the all providers failed error.

        :param message: Error message describing the overall failure
        :type message: str
        :param provider_errors: List of errors from each provider that was tried
        :type provider_errors: Optional[List[ProviderError]]
        """
        super().__init__(message)
        self.provider_errors = provider_errors or []

    def __str__(self) -> str:
        if self.provider_errors:
            error_details = "; ".join(str(e) for e in self.provider_errors)
            return f"{self.message}: [{error_details}]"
        return self.message

