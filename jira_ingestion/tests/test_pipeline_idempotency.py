"""
Tests for pipeline idempotency logic and timestamp parsing.

Covers:
  - _parse_ts: valid ISO timestamps
  - _parse_ts: Z suffix normalization
  - _parse_ts: empty / malformed input returns None
  - Idempotency: skip when not modified
  - Idempotency: reprocess when modified
  - Idempotency: force_reprocess bypasses check
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from jira_ingestion.ingestion.pipeline import _parse_ts


# ---------------------------------------------------------------------------
# _parse_ts
# ---------------------------------------------------------------------------

class TestParseTs:
    def test_iso_with_offset(self):
        dt = _parse_ts("2024-03-15T10:00:00.000+0000")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 15

    def test_z_suffix_normalized(self):
        dt = _parse_ts("2024-03-15T10:00:00Z")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_iso_date_only(self):
        dt = _parse_ts("2024-03-15")
        assert dt is not None

    def test_empty_string_returns_none(self):
        assert _parse_ts("") is None

    def test_none_equiv_empty(self):
        # The function only accepts str, but defensive check
        assert _parse_ts("") is None

    def test_malformed_returns_none(self):
        assert _parse_ts("not-a-date") is None

    def test_ordering_works(self):
        older = _parse_ts("2024-01-01T00:00:00Z")
        newer = _parse_ts("2024-06-01T00:00:00Z")
        assert older is not None and newer is not None
        assert older < newer

    def test_equal_timestamps(self):
        t1 = _parse_ts("2024-03-15T10:00:00Z")
        t2 = _parse_ts("2024-03-15T10:00:00Z")
        assert t1 == t2
        assert t1 <= t2


# ---------------------------------------------------------------------------
# Idempotency integration (minimal mock-based)
# ---------------------------------------------------------------------------

class TestIdempotency:
    """
    These tests verify the idempotency decision logic by mocking the
    coarse index and Jira client responses.
    """

    @pytest.mark.asyncio
    async def test_skip_when_not_modified(self):
        """If Jira updated <= stored updated_at, ticket should be skipped."""
        from unittest.mock import AsyncMock, MagicMock

        existing_record = {
            "ticket_key": "IDEA-1",
            "metadata": {"updated_at": "2024-06-01T10:00:00Z"},
        }
        coarse = MagicMock()
        coarse.get = MagicMock(return_value=existing_record)

        jira = AsyncMock()
        jira.get_ticket_data = AsyncMock(return_value={
            "key": "IDEA-1",
            "fields": {"updated": "2024-06-01T10:00:00Z"},  # same time
            "attachments": [],
            "themes": [],
        })

        from jira_ingestion.ingestion.pipeline import ingest_ticket
        from jira_ingestion.ingestion.indexing import InMemoryVectorIndex, InMemoryMetadataIndex, InMemorySupervisionStore

        result = await ingest_ticket(
            ticket_key="IDEA-1",
            jira_client=jira,
            coarse_index=coarse,
            fine_index=InMemoryVectorIndex(),
            metadata_index=InMemoryMetadataIndex(),
            supervision_store=InMemorySupervisionStore(),
        )
        assert result is existing_record

    @pytest.mark.asyncio
    async def test_reprocess_when_modified(self):
        """If Jira updated > stored updated_at, ticket should be reprocessed."""
        from unittest.mock import AsyncMock, MagicMock, patch

        existing_record = {
            "ticket_key": "IDEA-2",
            "metadata": {"updated_at": "2024-01-01T00:00:00Z"},
        }
        coarse = MagicMock()
        coarse.get = MagicMock(return_value=existing_record)
        coarse.upsert = MagicMock()

        jira = AsyncMock()
        jira.get_ticket_data = AsyncMock(return_value={
            "key": "IDEA-2",
            "fields": {"updated": "2024-12-01T00:00:00Z"},  # newer
            "attachments": [],
            "themes": [],
        })
        jira.download_attachment = AsyncMock(return_value=b"")

        with patch("jira_ingestion.ingestion.pipeline.assemble_document") as mock_assemble, \
             patch("jira_ingestion.ingestion.indexing.index_retrieval_view"), \
             patch("jira_ingestion.ingestion.indexing.index_supervision_view"):

            mock_assemble.return_value = {
                "ticket_key": "IDEA-2",
                "schema_version": "2.0",
                "ingested_at": "2024-12-01T00:00:00Z",
                "observed": {
                    "quality_tier": "D",
                    "stats": {"chunk_count": 0, "section_count": 0},
                },
                "supervision": {"trainability": {"is_trainable_for_vs": False}},
            }

            from jira_ingestion.ingestion.pipeline import ingest_ticket
            from jira_ingestion.ingestion.indexing import InMemoryVectorIndex, InMemoryMetadataIndex, InMemorySupervisionStore

            result = await ingest_ticket(
                ticket_key="IDEA-2",
                jira_client=jira,
                coarse_index=coarse,
                fine_index=InMemoryVectorIndex(),
                metadata_index=InMemoryMetadataIndex(),
                supervision_store=InMemorySupervisionStore(),
            )
            assert mock_assemble.called
            assert result["ticket_key"] == "IDEA-2"

    @pytest.mark.asyncio
    async def test_force_reprocess_bypasses_check(self):
        """force_reprocess=True should skip idempotency check entirely."""
        from unittest.mock import AsyncMock, MagicMock, patch

        coarse = MagicMock()
        coarse.get = MagicMock()  # should NOT be called
        coarse.upsert = MagicMock()

        jira = AsyncMock()
        jira.get_ticket_data = AsyncMock(return_value={
            "key": "IDEA-3",
            "fields": {"updated": "2024-06-01T00:00:00Z"},
            "attachments": [],
            "themes": [],
        })
        jira.download_attachment = AsyncMock(return_value=b"")

        with patch("jira_ingestion.ingestion.pipeline.assemble_document") as mock_assemble, \
             patch("jira_ingestion.ingestion.indexing.index_retrieval_view"), \
             patch("jira_ingestion.ingestion.indexing.index_supervision_view"):

            mock_assemble.return_value = {
                "ticket_key": "IDEA-3",
                "schema_version": "2.0",
                "ingested_at": "now",
                "observed": {
                    "quality_tier": "D",
                    "stats": {"chunk_count": 0, "section_count": 0},
                },
                "supervision": {"trainability": {"is_trainable_for_vs": False}},
            }

            from jira_ingestion.ingestion.pipeline import ingest_ticket
            from jira_ingestion.ingestion.indexing import InMemoryVectorIndex, InMemoryMetadataIndex, InMemorySupervisionStore

            await ingest_ticket(
                ticket_key="IDEA-3",
                jira_client=jira,
                coarse_index=coarse,
                fine_index=InMemoryVectorIndex(),
                metadata_index=InMemoryMetadataIndex(),
                supervision_store=InMemorySupervisionStore(),
                force_reprocess=True,
            )
            coarse.get.assert_not_called()
            assert mock_assemble.called
