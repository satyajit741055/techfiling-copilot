"""
Centralized Configuration Module.

This module provides a single source of truth for all application settings.
Configuration is loaded from environment variables with support for .env files.

Usage:
    from config import settings
    
    # Access any setting
    print(settings.ENVIRONMENT)  # "dev" or "prod"
    print(settings.LLM_PROVIDER)  # "ollama" or "huggingface"
    
    # Check environment
    if settings.is_dev:
        print("Running in development mode")
"""

from .settings import settings, Settings, Environment

__all__ = ["settings", "Settings", "Environment"]
