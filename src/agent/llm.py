"""Agent chat-model construction and bounded structured output."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config.settings import Settings, settings
from generation.validation import _extract_json_object

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def get_chat_model(role: str = "agent", config: Settings | None = None) -> BaseChatModel:
    config = config or settings
    provider = config.effective_agent_provider
    model = config.effective_agent_model
    if provider == "ollama":
        return ChatOllama(
            model=model, base_url=config.OLLAMA_HOST, temperature=0, num_ctx=config.OLLAMA_NUM_CTX
        )
    if provider == "openai":
        return ChatOpenAI(model=model, api_key=config.OPENAI_API_KEY, temperature=0)
    if provider == "huggingface":
        return ChatOpenAI(
            model=model,
            api_key=config.HUGGINGFACE_API_KEY,
            base_url="https://router.huggingface.co/v1",
            temperature=0,
        )
    raise ValueError(f"Agent provider {provider!r} is not supported by the LangGraph layer")


def _content_text(response: Any) -> str:
    content = response.content if hasattr(response, "content") else response
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
        )
    return json.dumps(content, ensure_ascii=False)


class AgentLLM:
    def __init__(self, model: BaseChatModel, concurrency: int = 2) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.model = model
        self._semaphore = asyncio.Semaphore(concurrency)

    async def invoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> Any:
        return await self.invoke_model(self.model, messages, **kwargs)

    async def invoke_model(self, model: Any, messages: Sequence[BaseMessage], **kwargs: Any) -> Any:
        async with self._semaphore:
            return await model.ainvoke(list(messages), **kwargs)

    async def structured(
        self,
        schema: type[SchemaT],
        messages: Sequence[BaseMessage],
    ) -> SchemaT:
        try:
            async with self._semaphore:
                result = await self.model.with_structured_output(schema).ainvoke(list(messages))
            return result if isinstance(result, schema) else schema.model_validate(result)
        except Exception as first_error:
            response = await self.invoke(messages)
            data = _extract_json_object(_content_text(response))
            if data is not None:
                try:
                    return schema.model_validate(data)
                except Exception:
                    pass
            repair = HumanMessage(
                content=(
                    f"Return only valid JSON matching this schema: "
                    f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
                )
            )
            response = await self.invoke([*messages, repair])
            data = _extract_json_object(_content_text(response))
            if data is None:
                raise ValueError("Model did not return a JSON object") from first_error
            return schema.model_validate(data)
