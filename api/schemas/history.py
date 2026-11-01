"""Response models for the conversation-history endpoints."""

import sys
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, Field

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from persistence import StoredMessage, ThreadSummary  # noqa: E402


class ThreadsResponse(BaseModel):
    """The current user's threads, most recently updated first."""

    threads: list[ThreadSummary] = Field(default_factory=list)


class ThreadMessagesResponse(BaseModel):
    """One thread's saved turns, oldest first."""

    thread_id: UUID
    messages: list[StoredMessage] = Field(default_factory=list)
