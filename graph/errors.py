class ElicitationError(RuntimeError):
    """Raised when elicitation LLM fails to return valid JSON after retries."""
