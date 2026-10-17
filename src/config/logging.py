"""
Logging Configuration for TechFiling Copilot.

Provides a centralized logging setup that can be used across all modules.

Usage:
    from config.logging import get_logger
    
    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.error("Something went wrong", exc_info=True)
"""

import logging
import sys
from functools import lru_cache
from typing import Literal

from .settings import settings


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(
    level: LogLevel | None = None,
    format_string: str | None = None,
) -> None:
    """
    Configure the root logger for the application.
    
    Args:
        level: Log level (defaults to settings.LOG_LEVEL)
        format_string: Custom format string (optional)
    """
    level = level or settings.LOG_LEVEL
    
    if format_string is None:
        if settings.is_dev:
            # More verbose format for development
            format_string = (
                "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
            )
        else:
            # Cleaner format for production
            format_string = "%(asctime)s | %(levelname)-8s | %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,  # Override any existing configuration
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.
    
    Uses lru_cache to ensure we return the same logger instance
    for the same name, avoiding duplicate handlers.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Starting process")
    """
    return logging.getLogger(name)


# Initialize logging on module import
setup_logging()
