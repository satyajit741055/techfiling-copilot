"""
LLM Client Abstraction.

Unified interface for multiple LLM providers:
- OpenAI (gpt-4o-mini, gpt-4o)
- Anthropic (claude-3-5-sonnet)
- Google (gemini-1.5-pro)
- Ollama (local models)

Usage:
    client = LLMClient(provider="ollama", model="qwen2.5-7b-instruct:latest")
    response = client.generate(system_prompt, user_prompt)
"""

from __future__ import annotations

import time
from typing import Any

from .schemas import LLMResponse


class LLMClient:
    """
    Unified interface for different LLM providers.
    
    Supports: OpenAI, Anthropic, Google, Ollama
    
    Attributes:
        provider: Provider name (openai, anthropic, google, ollama)
        model: Model name
        temperature: Generation temperature
        max_tokens: Maximum response length
    """
    
    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5-7b-instruct:latest",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        """
        Initialize the LLM client.

        Args:
            provider: Provider name (openai, anthropic, google, ollama)
            model: Model name/identifier for the provider
            temperature: Sampling temperature (0 for deterministic)
            max_tokens: Maximum response token length
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None
        
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize the appropriate client based on provider."""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI()  # Uses OPENAI_API_KEY env var
                print(f"OpenAI client initialized (model: {self.model})")
            except ImportError:
                raise ImportError("Install openai: pip install openai")
                
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self._client = Anthropic()  # Uses ANTHROPIC_API_KEY env var
                print(f"Anthropic client initialized (model: {self.model})")
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
                
        elif self.provider == "google":
            try:
                import google.generativeai as genai
                genai.configure()  # Uses GOOGLE_API_KEY env var
                self._client = genai.GenerativeModel(self.model)
                print(f"Google client initialized (model: {self.model})")
            except ImportError:
                raise ImportError("Install google-generativeai: pip install google-generativeai")
                
        elif self.provider == "ollama":
            try:
                import ollama
                self._client = ollama
                print(f"Ollama client initialized (model: {self.model})")
            except ImportError:
                raise ImportError("Install ollama: pip install ollama")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
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
