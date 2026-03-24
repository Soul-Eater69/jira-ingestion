"""
Tests for schema validation.

Covers:
  - validate_ticket_input: valid input passes
  - validate_ticket_input: missing key raises SchemaValidationError
  - validate_ticket_input: missing fields raises SchemaValidationError
  - validate_ticket_input: attachment without filename raises
  - validate_ticket_input: non-dict input raises
"""

from __future__ import annotations

import pytest

from jira_ingestion.models.validation import validate_ticket_input
from jira_ingestion.errors import SchemaValidationError


def _valid_ticket(**overrides) -> dict:
    base = {
        "key": "IDEA-1234",
        "fields": {"summary": "Test ticket", "description": ""},
        "attachments": [],
        "themes": [],
    }
    base.update(overrides)
    return base


class TestValidateTicketInput:
    def test_valid_passes(self):
        result = validate_ticket_input(_valid_ticket())
        assert result["key"] == "IDEA-1234"

    def test_missing_key_raises(self):
        data = _valid_ticket()
        del data["key"]
        with pytest.raises(SchemaValidationError):
            validate_ticket_input(data)

    def test_empty_key_raises(self):
        with pytest.raises(SchemaValidationError):
            validate_ticket_input(_valid_ticket(key=""))

    def test_whitespace_key_raises(self):
        with pytest.raises(SchemaValidationError):
            validate_ticket_input(_valid_ticket(key="   "))

    def test_missing_fields_raises(self):
        data = _valid_ticket()
        del data["fields"]
        with pytest.raises(SchemaValidationError):
            validate_ticket_input(data)

    def test_fields_not_dict_raises(self):
        with pytest.raises(SchemaValidationError):
            validate_ticket_input(_valid_ticket(fields="not-a-dict"))

    def test_attachment_without_filename_raises(self):
        with pytest.raises(SchemaValidationError):
            validate_ticket_input(_valid_ticket(attachments=[{"size": 1000}]))

    def test_attachment_with_empty_filename_raises(self):
        with pytest.raises(SchemaValidationError):
            validate_ticket_input(_valid_ticket(attachments=[{"filename": ""}]))

    def test_non_dict_input_raises(self):
        with pytest.raises(SchemaValidationError):
            validate_ticket_input("not-a-dict")

    def test_valid_with_attachments(self):
        data = _valid_ticket(attachments=[
            {"filename": "deck.pptx", "mimeType": "application/vnd.ms-powerpoint",
             "size": 500000, "content": "https://jira.example.com/secure/att/deck.pptx"},
        ])
        result = validate_ticket_input(data)
        assert len(result["attachments"]) == 1
