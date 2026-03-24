"""
Unit tests for metadata extraction.

Covers:
  - Basic field extraction
  - Config-driven custom field mapping
  - _as_text coercion
  - ADF comment extraction
  - Link classification
  - Epic key resolution
"""

from __future__ import annotations

import pytest

from jira_ingestion.ingestion.metadata import (
    extract_metadata,
    classify_links,
    _as_text,
    _resolve_field,
    _get_epic_key,
    _extract_adf_text,
)
from jira_ingestion.config import JiraIngestionConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fields(**overrides) -> dict:
    base = {
        "summary": "My great idea",
        "reporter": {"displayName": "Alice"},
        "created": "2024-01-15T10:00:00.000+0000",
        "labels": ["label-a", "label-b"],
        "components": [{"name": "Backend"}, {"name": "API"}],
        "priority": {"name": "High"},
        "issuelinks": [],
        "comment": {"comments": []},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _as_text
# ---------------------------------------------------------------------------

class TestAsText:
    def test_none(self):
        assert _as_text(None) == ""

    def test_string(self):
        assert _as_text("hello") == "hello"

    def test_dict_name(self):
        assert _as_text({"name": "Payments"}) == "Payments"

    def test_dict_value(self):
        assert _as_text({"value": "Enterprise"}) == "Enterprise"

    def test_dict_key(self):
        assert _as_text({"key": "IDEA-999"}) == "IDEA-999"

    def test_empty_dict(self):
        assert _as_text({}) == ""

    def test_list(self):
        result = _as_text([{"name": "A"}, {"name": "B"}])
        assert "A" in result and "B" in result

    def test_int(self):
        assert _as_text(42) == "42"


# ---------------------------------------------------------------------------
# extract_metadata — basic extraction
# ---------------------------------------------------------------------------

class TestExtractMetadata:
    def test_summary(self):
        meta = extract_metadata(_make_fields(), "IDEA-1")
        assert meta["summary"] == "My great idea"
        assert meta["title"] == "IDEA-1: My great idea"
        assert meta["ticket_key"] == "IDEA-1"

    def test_reporter(self):
        meta = extract_metadata(_make_fields(), "IDEA-1")
        assert meta["reporter"] == "Alice"

    def test_reporter_fallback_name(self):
        fields = _make_fields(reporter={"name": "bob"})
        meta = extract_metadata(fields, "IDEA-1")
        assert meta["reporter"] == "bob"

    def test_labels(self):
        meta = extract_metadata(_make_fields(), "IDEA-1")
        assert meta["labels"] == ["label-a", "label-b"]

    def test_components(self):
        meta = extract_metadata(_make_fields(), "IDEA-1")
        assert "Backend" in meta["components"]
        assert "API" in meta["components"]

    def test_priority(self):
        meta = extract_metadata(_make_fields(), "IDEA-1")
        assert meta["priority"] == "High"

    def test_created(self):
        meta = extract_metadata(_make_fields(), "IDEA-1")
        assert "2024-01-15" in meta["created"]

    def test_missing_fields_does_not_raise(self):
        meta = extract_metadata({}, "IDEA-99")
        assert meta["summary"] == ""
        assert meta["labels"] == []
        assert meta["components"] == []

    def test_metadata_text_contains_summary(self):
        meta = extract_metadata(_make_fields(), "IDEA-1")
        assert "IDEA-1" in meta["metadata_text"]
        assert "My great idea" in meta["metadata_text"]


# ---------------------------------------------------------------------------
# extract_metadata — config-driven custom field mapping
# ---------------------------------------------------------------------------

class TestConfigDrivenFieldMapping:
    def test_business_unit_from_config_field_map(self):
        fields = _make_fields(customfield_99999="FinTech Division")
        config = JiraIngestionConfig(jira_field_map={"business_unit": "customfield_99999"})
        meta = extract_metadata(fields, "IDEA-1", config=config)
        assert meta["business_unit"] == "FinTech Division"

    def test_product_area_from_config_field_map(self):
        fields = _make_fields(customfield_88888={"name": "Payments Platform"})
        config = JiraIngestionConfig(jira_field_map={"product_area": "customfield_88888"})
        meta = extract_metadata(fields, "IDEA-1", config=config)
        assert meta["product_area"] == "Payments Platform"

    def test_default_field_map_still_works(self):
        fields = _make_fields(customfield_10002="Enterprise")
        meta = extract_metadata(fields, "IDEA-1")  # no config — uses defaults
        assert meta["business_unit"] == "Enterprise"

    def test_wrong_field_id_returns_empty(self):
        fields = _make_fields(customfield_10002="Enterprise")
        config = JiraIngestionConfig(jira_field_map={"business_unit": "customfield_99999"})
        meta = extract_metadata(fields, "IDEA-1", config=config)
        assert meta["business_unit"] == ""

    def test_metadata_text_includes_business_unit(self):
        fields = _make_fields(customfield_10002="Retail Banking")
        meta = extract_metadata(fields, "IDEA-1")
        assert "Retail Banking" in meta["metadata_text"]


# ---------------------------------------------------------------------------
# Epic key resolution
# ---------------------------------------------------------------------------

class TestEpicKeyResolution:
    def test_epic_from_configured_field(self):
        fields = _make_fields(customfield_10014="EPIC-42")
        meta = extract_metadata(fields, "IDEA-1")
        assert meta["epic_key"] == "EPIC-42"

    def test_epic_from_parent(self):
        fields = _make_fields(parent={"key": "EPIC-99"})
        meta = extract_metadata(fields, "IDEA-1")
        assert meta["epic_key"] == "EPIC-99"

    def test_no_epic_returns_none(self):
        meta = extract_metadata(_make_fields(), "IDEA-1")
        assert meta["epic_key"] is None

    def test_custom_epic_field_overrides_default(self):
        fields = _make_fields(customfield_55555="EPIC-CUSTOM")
        config = JiraIngestionConfig(jira_field_map={"epic_link": "customfield_55555"})
        meta = extract_metadata(fields, "IDEA-1", config=config)
        assert meta["epic_key"] == "EPIC-CUSTOM"


# ---------------------------------------------------------------------------
# classify_links
# ---------------------------------------------------------------------------

class TestClassifyLinks:
    def _link(self, type_name: str, direction: str = "outward", key: str = "VS-1") -> dict:
        issue_key = "outwardIssue" if direction == "outward" else "inwardIssue"
        return {
            "type": {"name": type_name},
            issue_key: {"key": key, "fields": {"summary": "Test", "status": {"name": "Open"}}},
        }

    def test_vs_link(self):
        links = classify_links([self._link("Value Stream")])
        assert len(links["vs"]) == 1
        assert links["vs"][0]["key"] == "VS-1"

    def test_dependency_link(self):
        links = classify_links([self._link("Blocks")])
        assert len(links["dependency"]) == 1

    def test_parent_link(self):
        links = classify_links([self._link("Epic Link")])
        assert len(links["parent"]) == 1

    def test_unknown_link_type(self):
        links = classify_links([self._link("Mysterious Custom Link")])
        assert len(links["unknown"]) == 1

    def test_empty_issuelinks(self):
        links = classify_links([])
        assert all(len(v) == 0 for v in links.values())

    def test_inward_direction(self):
        links = classify_links([self._link("Relates To", direction="inward", key="OTHER-5")])
        assert links["related"][0]["direction"] == "inward"
        assert links["related"][0]["key"] == "OTHER-5"


# ---------------------------------------------------------------------------
# ADF comment extraction
# ---------------------------------------------------------------------------

class TestADFExtraction:
    def test_plain_text_comment(self):
        # Build a body that is clearly >50 words
        long_body = (
            "This is a substantive comment explaining the business rationale for "
            "this idea in detail. We have considered multiple approaches and believe "
            "this solution addresses the core problem. The key benefit is reduced "
            "operational overhead, faster time to market, and improved customer "
            "satisfaction scores across all business segments. Further analysis "
            "shows alignment with the strategic objectives for the year."
        )
        comment_container = {
            "comments": [
                {
                    "author": {"displayName": "Bob"},
                    "body": long_body,
                }
            ]
        }
        from jira_ingestion.ingestion.metadata import _extract_comments
        result = _extract_comments(comment_container)
        assert len(result) == 1
        assert "substantive comment" in result[0]

    def test_short_comment_excluded(self):
        from jira_ingestion.ingestion.metadata import _extract_comments
        result = _extract_comments({
            "comments": [{"author": {"displayName": "Bob"}, "body": "Short"}]
        })
        assert result == []

    def test_bot_comment_excluded(self):
        from jira_ingestion.ingestion.metadata import _extract_comments
        long_body = "word " * 60
        result = _extract_comments({
            "comments": [{
                "author": {"displayName": "Jira-Bot"},
                "body": long_body,
            }]
        })
        assert result == []
