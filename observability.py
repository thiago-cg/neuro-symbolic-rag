"""Structured logging and observability setup."""

import logging
import time
from functools import wraps
from typing import Callable

import structlog

from config import settings


def configure_logging() -> None:
    """Configure structlog for JSON structured logging."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structlog bound logger."""
    return structlog.get_logger(name)


def log_node_execution(node_name: str) -> Callable:
    """Decorator that logs node execution with input/output sizes and duration."""
    def decorator(func: Callable) -> Callable:
        logger = get_logger(node_name)

        @wraps(func)
        def wrapper(state: dict, *args, **kwargs) -> dict:
            input_size = len(str(state))
            start = time.perf_counter()

            result = func(state, *args, **kwargs)

            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            output_size = len(str(result)) if result else 0

            logger.info(
                "node_executed",
                node=node_name,
                input_size=input_size,
                output_size=output_size,
                duration_ms=duration_ms,
            )
            return result

        return wrapper
    return decorator
