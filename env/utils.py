from __future__ import annotations

from typing import Any

from env.models import Email, ProcessedEmail


def recent_history(history: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    """Return a bounded tail of recent actions for observations."""

    return history[-limit:]


def make_processed_emails(emails: list[Email]) -> list[ProcessedEmail]:
    """Initialize the per-email state for an episode."""

    return [ProcessedEmail(email=email) for email in emails]


def serialize_processed_emails(processed_emails: list[ProcessedEmail]) -> list[dict[str, Any]]:
    """Serialize processed email state for the state() method."""

    return [item.model_dump() for item in processed_emails]
