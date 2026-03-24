"""
Tests for attachment extraction handling.

Covers:
  - fetch_attachment_content: markitdown absent → clean disabled result
  - fetch_attachment_content: missing URL → no_content_url error
  - fetch_attachment_content: unauthenticated client → RuntimeError
  - download_attachment: missing URL → ValueError
  - download_attachment: missing auth → RuntimeError
  - _build_ticket_fields: returns base + custom fields from config
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from jira_ingestion.clients.jira.value_stream_client import JiraValueStreamClient
from jira_ingestion.config import JiraIngestionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> JiraValueStreamClient:
    return JiraValueStreamClient(
        base_url="https://jira.example.com",
        token="test-token",
        verify_ssl=False,
    )


# ---------------------------------------------------------------------------
# _build_ticket_fields
# ---------------------------------------------------------------------------

class TestBuildTicketFields:
    def test_base_fields_always_present(self):
        client = _make_client()
        fields = client._build_ticket_fields()
        for f in ("summary", "description", "reporter", "created", "attachment", "issuelinks"):
            assert f in fields

    def test_config_custom_fields_added(self):
        client = _make_client()
        config = JiraIngestionConfig(jira_field_map={
            "business_unit": "customfield_99999",
            "product_area": "customfield_88888",
        })
        fields = client._build_ticket_fields(config)
        assert "customfield_99999" in fields
        assert "customfield_88888" in fields

    def test_non_customfield_values_excluded(self):
        client = _make_client()
        config = JiraIngestionConfig(jira_field_map={
            "epic_link": "customfield_10014",
            "ignored": "not_a_customfield",
        })
        fields = client._build_ticket_fields(config)
        assert "customfield_10014" in fields
        assert "not_a_customfield" not in fields

    def test_no_duplicates(self):
        client = _make_client()
        config = JiraIngestionConfig(jira_field_map={
            "f1": "customfield_10001",
            "f2": "customfield_10001",  # same ID twice
        })
        fields = client._build_ticket_fields(config)
        assert fields.count("customfield_10001") == 1

    def test_no_config_returns_base_only(self):
        client = _make_client()
        fields = client._build_ticket_fields(None)
        # No customfield_* IDs beyond base
        custom = [f for f in fields if f.startswith("customfield_")]
        assert custom == []


# ---------------------------------------------------------------------------
# fetch_attachment_content — markitdown absent
# ---------------------------------------------------------------------------

class TestFetchAttachmentContentMarkitdownAbsent:
    @pytest.mark.asyncio
    async def test_returns_disabled_error_when_markitdown_missing(self):
        client = _make_client()
        # Simulate markitdown not installed
        client.client._client = MagicMock()  # authenticated

        with patch.dict("sys.modules", {"markitdown": None}):
            results = await client.fetch_attachment_content([
                {"filename": "deck.pptx", "mimeType": "application/vnd.ms-powerpoint",
                 "content": "https://jira.example.com/secure/att/deck.pptx"},
            ])

        assert len(results) == 1
        assert results[0]["text_content"] == ""
        assert results[0]["error"] == "markitdown_not_installed"

    @pytest.mark.asyncio
    async def test_no_content_url_returns_no_url_error(self):
        client = _make_client()
        client.client._client = MagicMock()

        with patch.dict("sys.modules", {"markitdown": None}):
            results = await client.fetch_attachment_content([
                {"filename": "mystery.pdf", "mimeType": "application/pdf", "content": ""},
            ])

        assert results[0]["error"] == "no_content_url"

    @pytest.mark.asyncio
    async def test_raises_if_not_authenticated(self):
        client = _make_client()
        # client._client is None — not authenticated
        with pytest.raises(RuntimeError, match="authenticate"):
            await client.fetch_attachment_content([{"filename": "f.pptx", "content": "http://x"}])


# ---------------------------------------------------------------------------
# download_attachment — contract
# ---------------------------------------------------------------------------

class TestDownloadAttachment:
    @pytest.mark.asyncio
    async def test_raises_if_not_authenticated(self):
        client = _make_client()
        with pytest.raises(RuntimeError, match="authenticate"):
            await client.download_attachment({"content": "http://x", "filename": "f.pptx"})

    @pytest.mark.asyncio
    async def test_raises_if_no_url(self):
        client = _make_client()
        client.client._client = MagicMock()
        with pytest.raises(ValueError, match="No download URL"):
            await client.download_attachment({"filename": "f.pptx"})

    @pytest.mark.asyncio
    async def test_returns_bytes_on_success(self):
        client = _make_client()
        mock_response = MagicMock()
        mock_response.content = b"fake-bytes"
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        client.client._client = mock_http

        result = await client.download_attachment(
            {"content": "https://jira.example.com/att/file.pptx", "filename": "file.pptx"}
        )
        assert result == b"fake-bytes"
