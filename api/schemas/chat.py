"""
Chat Schemas.

Request model for POST /chat. The response is a text/event-stream; its events are
documented on the endpoint.
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_MESSAGE_CHARS = 2000
MAX_TICKERS = 6
MAX_YEARS = 10


class ChatFilters(BaseModel):
    """Optional company/year chips. When set they define the scope and skip clarification."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(default_factory=list, max_length=MAX_TICKERS)
    fiscal_years: list[int] = Field(default_factory=list, max_length=MAX_YEARS)

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip().upper() for value in values if value.strip()))

    @field_validator("fiscal_years")
    @classmethod
    def _unique_years(cls, values: list[int]) -> list[int]:
        return sorted(set(values))


class ChatResume(BaseModel):
    """Answer to a clarification: one of its option ids, or free text."""

    model_config = ConfigDict(extra="forbid")

    option_id: str | None = Field(default=None, pattern=r"^opt_\d{1,2}$")
    text: str | None = Field(default=None, min_length=1, max_length=500)

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None

    @model_validator(mode="after")
    def _exactly_one(self) -> "ChatResume":
        if (self.option_id is None) == (self.text is None):
            raise ValueError("Pass exactly one of option_id or text")
        return self


class ChatRequest(BaseModel):
    """A new message, or the answer to a pending clarification on the same thread."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "message": "Compare R&D spend for Apple and Microsoft over the last 3 years",
                    "filters": {"tickers": ["AAPL", "MSFT"]},
                },
                {
                    "thread_id": "6f1c2a52-6a3b-4c51-9b7e-3d1f0f5c2e11",
                    "resume": {"option_id": "opt_1"},
                },
            ]
        },
    )

    message: str | None = Field(default=None, min_length=1, max_length=MAX_MESSAGE_CHARS)
    thread_id: UUID | None = Field(
        default=None, description="Conversation id; the server creates one when absent"
    )
    filters: ChatFilters | None = None
    resume: ChatResume | None = None

    @field_validator("message")
    @classmethod
    def _strip_message(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("Message must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def _message_or_resume(self) -> "ChatRequest":
        if (self.message is None) == (self.resume is None):
            raise ValueError("Pass exactly one of message or resume")
        if self.resume is not None and self.thread_id is None:
            raise ValueError("resume requires the thread_id of the clarification")
        return self
