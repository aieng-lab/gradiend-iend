"""
Logging configuration for GRADIEND.

This module sets up logging with appropriate levels and formatting.

Only the ``gradiend`` logger namespace is configured — the root logger and
third-party loggers (httpx, huggingface_hub, transformers, …) are left
untouched so library users keep full control over their own logging.
"""

import logging
import sys
from contextlib import contextmanager
from typing import Generator

_LIB_LOGGER_NAME = "gradiend"


def setup_logging(level=logging.INFO):
    """
    Set up logging configuration for the ``gradiend`` namespace.

    Attaches a :class:`~logging.StreamHandler` to the ``gradiend`` logger
    (not the root logger) so only GRADIEND messages are affected.
    Third-party loggers (httpx, huggingface_hub, transformers, …) are
    never touched.

    Args:
        level: Logging level (default: INFO)
    """
    if not isinstance(level, int):
        raise TypeError(f"level must be int (e.g. logging.INFO), got {type(level).__name__}")

    lib_logger = logging.getLogger(_LIB_LOGGER_NAME)
    lib_logger.setLevel(level)

    if not lib_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        lib_logger.addHandler(handler)
    else:
        for handler in lib_logger.handlers:
            handler.setLevel(level)

    # Let applications and pytest's caplog observe GRADIEND records through the
    # normal root logging path. Applications that want isolated GRADIEND output
    # can still set logging.getLogger("gradiend").propagate = False.
    lib_logger.propagate = True


def get_logger(name):
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be str, got {type(name).__name__}")
    return logging.getLogger(name)


# Loggers used by Hugging Face tokenizers when emitting the "Token indices sequence
# length is longer than the specified maximum sequence length" warning (truncation is
# applied when max_length/truncation are set, but the warning still appears).
_TOKENIZER_LOGGER_NAMES = (
    "transformers.tokenization_utils",
    "transformers.tokenization_utils_base",
    "transformers.tokenization_utils_fast",
)


@contextmanager
def suppress_tokenizer_length_warning() -> Generator[None, None, None]:
    """
    Temporarily suppress the Hugging Face tokenizer warning about sequence length
    exceeding the model maximum when truncation is explicitly requested (max_length +
    truncation=True). Use only around tokenizer calls that pass truncation and
    max_length.
    """
    loggers = [logging.getLogger(n) for n in _TOKENIZER_LOGGER_NAMES]
    old_levels = [log.level for log in loggers]
    try:
        for log in loggers:
            log.setLevel(logging.WARNING)
        yield
    finally:
        for log, level in zip(loggers, old_levels):
            log.setLevel(level)


# Configure the gradiend logger once on first import.
setup_logging(logging.INFO)
