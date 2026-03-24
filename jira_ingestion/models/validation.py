"""
Runtime schema validation for pipeline inputs.

Uses Pydantic v2 when available, falls back to a lightweight dataclass-based
validator so the pipeline can run without Pydantic installed.

Usage:
    from jira_ingestion.models.validation import validate_ticket_input

    validated = validate_ticket_input(raw_ticket_data)   # raises SchemaValidationError on failure
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try Pydantic v2, then v1, then fall back to lightweight validation
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel, Field, field_validator  # type: ignore

    _PYDANTIC_AVAILABLE = True
    _PYDANTIC_V2 = True

    class AttachmentModel(BaseModel):
        id: Optional[str] = None
        filename: str
        mimeType: Optional[str] = None
        size: Optional[int] = None
        content: Optional[str] = None   # download URL

        @field_validator("filename")
        @classmethod
        def filename_not_empty(cls, v: str) -> str:
            if not v.strip():
                raise ValueError("attachment filename must not be empty")
            return v

    class TicketDataModel(BaseModel):
        key: str
        fields: Dict[str, Any]
        attachments: List[AttachmentModel] = Field(default_factory=list)
        themes: List[Dict[str, Any]] = Field(default_factory=list)

        @field_validator("key")
        @classmethod
        def key_not_empty(cls, v: str) -> str:
            if not v.strip():
                raise ValueError("ticket key must not be empty")
            return v

    def _validate_with_pydantic(data: dict) -> dict:
        model = TicketDataModel(**data)
        return model.model_dump()

except ImportError:
    try:
        from pydantic import BaseModel, Field, validator  # type: ignore

        _PYDANTIC_AVAILABLE = True
        _PYDANTIC_V2 = False

        class AttachmentModel(BaseModel):  # type: ignore[no-redef]
            id: Optional[str] = None
            filename: str
            mimeType: Optional[str] = None
            size: Optional[int] = None
            content: Optional[str] = None

            @validator("filename")
            @classmethod
            def filename_not_empty(cls, v: str) -> str:
                if not v.strip():
                    raise ValueError("attachment filename must not be empty")
                return v

        class TicketDataModel(BaseModel):  # type: ignore[no-redef]
            key: str
            fields: Dict[str, Any]
            attachments: List[AttachmentModel] = Field(default_factory=list)
            themes: List[Dict[str, Any]] = Field(default_factory=list)

        def _validate_with_pydantic(data: dict) -> dict:
            model = TicketDataModel(**data)
            return model.dict()

    except ImportError:
        _PYDANTIC_AVAILABLE = False
        _PYDANTIC_V2 = False

        def _validate_with_pydantic(data: dict) -> dict:  # type: ignore[misc]
            raise RuntimeError("Pydantic not installed — using lightweight validation only")


# ---------------------------------------------------------------------------
# Lightweight fallback validator (no external deps)
# ---------------------------------------------------------------------------

def _validate_lightweight(data: Any) -> dict:
    """
    Minimal structural validation of a ticket input dict.

    Checks only the fields the pipeline strictly requires.
    Raises SchemaValidationError on failure.
    """
    from jira_ingestion.errors import SchemaValidationError

    if not isinstance(data, dict):
        raise SchemaValidationError(f"Expected dict, got {type(data).__name__}")

    key = data.get("key")
    if not key or not isinstance(key, str) or not key.strip():
        raise SchemaValidationError(
            f"'key' must be a non-empty string, got: {key!r}"
        )

    fields = data.get("fields")
    if fields is None or not isinstance(fields, dict):
        raise SchemaValidationError(
            f"'fields' must be a dict, got: {type(fields).__name__}"
        )

    attachments = data.get("attachments", [])
    if not isinstance(attachments, list):
        raise SchemaValidationError(
            f"'attachments' must be a list, got: {type(attachments).__name__}"
        )
    for i, att in enumerate(attachments):
        if not isinstance(att, dict):
            raise SchemaValidationError(f"attachment[{i}] must be a dict, got {type(att).__name__}")
        if not att.get("filename"):
            raise SchemaValidationError(f"attachment[{i}] is missing 'filename'")

    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_ticket_input(data: Any, strict: bool = False) -> dict:
    """
    Validate a raw ticket data payload.

    Args:
        data:   Dict output of JiraValueStreamClient.get_ticket_data().
        strict: If True and Pydantic is installed, run full Pydantic validation
                (coerces and normalises values).  If False (default), run only
                the lightweight structural check.

    Returns:
        The validated (and possibly coerced) dict.

    Raises:
        SchemaValidationError: if validation fails.
    """
    from jira_ingestion.errors import SchemaValidationError

    if strict and _PYDANTIC_AVAILABLE:
        try:
            return _validate_with_pydantic(data)
        except Exception as exc:
            raise SchemaValidationError(str(exc), cause=exc) from exc

    try:
        return _validate_lightweight(data)
    except SchemaValidationError:
        raise
    except Exception as exc:
        raise SchemaValidationError(str(exc), cause=exc) from exc


def is_pydantic_available() -> bool:
    """Return True if Pydantic is installed and validation models are available."""
    return _PYDANTIC_AVAILABLE
