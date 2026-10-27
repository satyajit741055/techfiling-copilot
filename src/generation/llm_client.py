"""
LLM Client Abstraction.

Unified interface for multiple LLM providers:
- OpenAI (gpt-4o-mini, gpt-4o)
- Anthropic (claude-3-5-sonnet)
- Google (gemini-1.5-pro)
- Ollama (local models with CUDA)
- Hugging Face Inference API (cloud-hosted models)

Configuration is loaded from centralized settings (config.settings).

Usage:
    # Using defaults from .env / settings
    from generation.llm_client import LLMClient
    client = LLMClient()  # Uses LLM_PROVIDER and LLM_MODEL from settings
    response = client.generate(system_prompt, user_prompt)

    # Override specific settings
    client = LLMClient(provider="huggingface", model="Qwen/Qwen2.5-7B-Instruct")
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .schemas import LLMResponse

logger = logging.getLogger(__name__)


def _get_settings():
    """Lazy import to avoid circular imports."""
    from config.settings import settings
    return settings


class LLMClient:
    """
    Unified interface for different LLM providers.

    Supports: OpenAI, Anthropic, Google, Ollama, Hugging Face

    All defaults are loaded from centralized settings (config.settings).

    Attributes:
        provider: Provider name (openai, anthropic, google, ollama, huggingface)
        model: Model name
        temperature: Generation temperature
        max_tokens: Maximum response length
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """
        Initialize the LLM client.

        Args:
            provider: Provider name (openai, anthropic, google, ollama, huggingface).
                     If None, uses LLM_PROVIDER from settings.
            model: Model name/identifier for the provider.
                   If None, uses the default model for the provider from settings.
            temperature: Sampling temperature (0 for deterministic).
                        If None, uses LLM_TEMPERATURE from settings.
            max_tokens: Maximum response token length.
                       If None, uses LLM_MAX_TOKENS from settings.
        """
        settings = _get_settings()

        # Use settings defaults if not provided
        self.provider = (provider or settings.effective_llm_provider).lower()
        self.model = model or settings.model_for_provider(self.provider)
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS

        # Store settings reference for API keys
        self._settings = settings
        self._client: Any = None

        self._init_client()

    def _init_client(self) -> None:
        """Initialize the appropriate client based on provider."""
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "google":
            self._init_google()
        elif self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "huggingface":
            self._init_huggingface()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            # Use API key from settings if available, otherwise fall back to env var
            api_key = self._settings.OPENAI_API_KEY
            self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
            logger.info(f"OpenAI client initialized (model: {self.model})")
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def _init_anthropic(self) -> None:
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic
            api_key = self._settings.ANTHROPIC_API_KEY
            self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
            logger.info(f"Anthropic client initialized (model: {self.model})")
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    def _init_google(self) -> None:
        """Initialize Google AI client."""
        try:
            import google.generativeai as genai
            api_key = self._settings.GOOGLE_API_KEY
            if api_key:
                genai.configure(api_key=api_key)
            else:
                genai.configure()  # Uses GOOGLE_API_KEY env var
            self._client = genai.GenerativeModel(self.model)
            logger.info(f"Google client initialized (model: {self.model})")
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

    def _init_ollama(self) -> None:
        """Initialize Ollama client for local LLM with CUDA."""
        try:
            import ollama
            ollama_host = self._settings.OLLAMA_HOST
            self._client = ollama.Client(host=ollama_host) if ollama_host else ollama
            logger.info(f"Ollama client initialized (model: {self.model}, host: {ollama_host})")
        except ImportError:
            raise ImportError("Install ollama: pip install ollama")

    def _init_huggingface(self) -> None:
        """Initialize Hugging Face Inference Providers chat client."""
        from huggingface_hub import InferenceClient
        self._client = InferenceClient(
            provider=self._settings.HUGGINGFACE_PROVIDER,
            api_key=self._settings.HUGGINGFACE_API_KEY,
            timeout=self._settings.HUGGINGFACE_TIMEOUT,
        )
        logger.info(
            f"Hugging Face client initialized (model: {self.model}, "
            f"provider: {self._settings.HUGGINGFACE_PROVIDER})"
        )

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System message defining behavior
            user_prompt: User message with question and context

        Returns:
            LLMResponse with content and metadata
        """
        start_time = time.time()

        if self.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt, start_time)
        elif self.provider == "anthropic":
            return self._generate_anthropic(system_prompt, user_prompt, start_time)
        elif self.provider == "google":
            return self._generate_google(system_prompt, user_prompt, start_time)
        elif self.provider == "ollama":
            return self._generate_ollama(system_prompt, user_prompt, start_time)
        elif self.provider == "huggingface":
            return self._generate_huggingface(system_prompt, user_prompt, start_time)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _generate_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        start_time: float,
    ) -> LLMResponse:
        """Generate a response using OpenAI's chat completions API with JSON output."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},  # Enforce JSON output
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_seconds=time.time() - start_time,
            raw_response=response,
        )

    def _generate_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        start_time: float,
    ) -> LLMResponse:
        """Generate a response using the Anthropic Messages API."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature,
        )

        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            latency_seconds=time.time() - start_time,
            raw_response=response,
        )

    def _generate_google(
        self,
        system_prompt: str,
        user_prompt: str,
        start_time: float,
    ) -> LLMResponse:
        """Generate a response using Google's Generative AI API."""
        # Google combines system + user in the prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self._client.generate_content(
            full_prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )

        # Token counts may not be available for all models
        prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
        completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

        return LLMResponse(
            content=response.text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=time.time() - start_time,
            raw_response=response,
        )

    def _generate_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        start_time: float,
    ) -> LLMResponse:
        """Generate a response using the local Ollama server."""
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )

        return LLMResponse(
            content=response['message']['content'],
            model=self.model,
            prompt_tokens=response.get('prompt_eval_count', 0),
            completion_tokens=response.get('eval_count', 0),
            latency_seconds=time.time() - start_time,
            raw_response=response,
        )

    def _generate_huggingface(
        self,
        system_prompt: str,
        user_prompt: str,
        start_time: float,
    ) -> LLMResponse:
        """Generate a response with the Hugging Face chat-completion API.

        The provider applies the model's own chat template, so no manual prompt
        formatting is needed.
        """
        response = self._client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        usage = getattr(response, "usage", None)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_seconds=time.time() - start_time,
            raw_response=response,
        )
