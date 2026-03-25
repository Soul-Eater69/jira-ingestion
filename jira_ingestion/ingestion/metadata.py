"""
Metadata extraction and linked-issue classification.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Link type classification map
# ---------------------------------------------------------------------------

LINK_TYPE_MAP: dict[str, str] = {
    # Value stream links (ground truth — goes to supervision view)
    "Value Stream": "vs",
    "value-stream": "vs",
    "VS Link": "vs",
    "Implements Value Stream": "vs",
    # Product / system links
    "Product": "product",
    "Affects Product": "product",
    "Impacts System": "product",
    # Dependency links
    "Depends On": "dependency",
    "Blocks": "dependency",
    "Is Blocked By": "dependency",
    "blocks": "dependency",
    "is blocked by": "dependency",
    "depends on": "dependency",
    # Hierarchy links
    "Epic Link": "parent",
    "Parent": "parent",
    "Sub-task": "parent",
    "is child of": "parent",
    "is parent of": "parent",
    # Related ideas
    "Relates To": "related",
    "relates to": "related",
    "Duplicates": "related",
    "duplicates": "related",
    "Clones": "related",
    "clones": "related",
    # Implementation links
    "Implements": "implementation",
    "Story Link": "implementation",
    "is implemented by": "implementation",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_metadata(
    ticket_fields: dict,
    ticket_key: str,
    config: Optional[Any] = None,
) -> dict:
    """
    Extract and structure all useful metadata from raw Jira fields.

    Custom field IDs are resolved from config.jira_field_map when provided,
    making extraction portable across Jira instances.

    Args:
        ticket_fields: The 'fields' dict from Jira API response.
        ticket_key:    The issue key (e.g. "IDEA-1234").
        config:        JiraIngestionConfig instance (uses defaults if None).

    Returns:
        Flat metadata dict ready for pipeline use.
    """
    fields = ticket_fields or {}
    jira_field_map: dict[str, str] = (
        getattr(config, "jira_field_map", {}) if config else {}
    )

    # Tier 1 — always extract
    summary = fields.get("summary", "")
    reporter_obj = fields.get("reporter") or {}
    reporter = reporter_obj.get("displayName", reporter_obj.get("name", ""))
    created = fields.get("created", "")
    labels: list[str] = [lbl for lbl in (fields.get("labels") or []) if isinstance(lbl, str)]
    components: list[str] = [
        c.get("name", "") for c in (fields.get("components") or []) if isinstance(c, dict)
    ]

    # Tier 2 — config-driven custom fields (deterministic, not heuristic)
    business_unit = _as_text(
        _resolve_field(fields, jira_field_map.get("business_unit", "customfield_10002"))
    )
    product_area = _as_text(
        _resolve_field(fields, jira_field_map.get("product_area", "customfield_10003"))
    )
    priority = (fields.get("priority") or {}).get("name", "")
    epic_key = _get_epic_key(fields, jira_field_map)

    # Tier 2b — standard Jira fields (issue type, status, resolution)
    issue_type = (fields.get("issuetype") or {}).get("name", "")
    status = (fields.get("status") or {}).get("name", "")
    resolution = (fields.get("resolution") or {}).get("name", "")

    # Tier 2c — organisational fields
    requesting_org = _as_text(
        _resolve_field(fields, jira_field_map.get("requesting_org", "customfield_10102"))
    )
    delivery_org = _as_text(
        _resolve_field(fields, jira_field_map.get("delivery_org", "customfield_10103"))
    )

    # Tier 3 — comments (first 2-3 substantive ones for chunk use)
    substantive_comments = _extract_substantive_comments(fields.get("comment") or {})

    # Build metadata text for BM25 + embedding.
    # NOTE: VS labels / impacted products are NOT included here —
    # they go to the supervision view only (no label leakage into retrieval text).
    meta_parts: list[str] = [f"{ticket_key}: {summary}"]
    if components:
        meta_parts.append(f"Components: {', '.join(components)}")
    if labels:
        meta_parts.append(f"Labels: {', '.join(labels)}")
    if business_unit:
        meta_parts.append(f"Business Unit: {business_unit}")
    if product_area:
        meta_parts.append(f"Product Area: {product_area}")
    if reporter:
        meta_parts.append(f"Reporter: {reporter}")
    if issue_type:
        meta_parts.append(f"Type: {issue_type}")
    if status:
        meta_parts.append(f"Status: {status}")
    if requesting_org:
        meta_parts.append(f"Requesting Org: {requesting_org}")

    return {
        "ticket_key": ticket_key,
        "title": f"{ticket_key}: {summary}",
        "summary": summary,
        "reporter": reporter,
        "created": created,
        "labels": labels,
        "components": components,
        "business_unit": business_unit,
        "product_area": product_area,
        "priority": priority,
        "issue_type": issue_type,
        "status": status,
        "resolution": resolution,
        "requesting_org": requesting_org,
        "delivery_org": delivery_org,
        "epic_key": epic_key,
        "substantive_comments": substantive_comments,
        "metadata_text": ". ".join(meta_parts),
        # classified_links is set separately by classify_links()
    }


def extract_all_comments(ticket_fields: dict) -> list[dict]:
    """
    Extract all comments as structured records (raw layer).

    Returns a list of dicts with keys: comment_id, author, created_at,
    body_raw, body_cleaned, word_count, is_substantive.
    """
    comment_container = ticket_fields.get("comment") or {}
    comments_list = comment_container.get("comments", [])
    bot_patterns = ["atlassian-bot", "jira-bot", "automation", "webhook"]
    records: list[dict] = []

    for comment in comments_list:
        author_obj = comment.get("author") or {}
        author = author_obj.get("displayName", author_obj.get("name", ""))
        author_lower = author.lower()
        is_bot = any(p in author_lower for p in bot_patterns)

        body = comment.get("body", "")
        if isinstance(body, dict):
            body = _extract_adf_text(body)
        body_raw = body.strip()
        body_cleaned = _clean_comment_text(body_raw)
        word_count = len(body_cleaned.split())

        records.append({
            "comment_id": str(comment.get("id", "")),
            "author": author,
            "created_at": comment.get("created", ""),
            "body_raw": body_raw,
            "body_cleaned": body_cleaned[:2000],
            "word_count": word_count,
            "is_substantive": word_count >= 50 and not is_bot,
        })

    return records


def extract_impacted_products(
    ticket_fields: dict,
    jira_field_map: dict[str, str],
) -> tuple[list[dict], list[str], list[str]]:
    """
    Extract impacted products from custom fields.

    Returns:
        (raw_list, ids, names)
    """
    raw_value = _resolve_field(
        ticket_fields,
        jira_field_map.get("impacted_products", "customfield_10100"),
    )
    return _extract_product_list(raw_value)


def extract_impacted_it_products(
    ticket_fields: dict,
    jira_field_map: dict[str, str],
) -> tuple[list[dict], list[str], list[str]]:
    """
    Extract impacted IT products from custom fields.

    Returns:
        (raw_list, ids, names)
    """
    raw_value = _resolve_field(
        ticket_fields,
        jira_field_map.get("impacted_it_products", "customfield_10101"),
    )
    return _extract_product_list(raw_value)


def classify_links(issuelinks: list[dict]) -> dict[str, list[dict]]:
    """
    Classify Jira issue links into semantic categories.

    Returns a dict keyed by category (vs, product, dependency, parent,
    related, implementation, unknown).
    """
    classified: dict[str, list[dict]] = {
        "vs": [],
        "product": [],
        "dependency": [],
        "parent": [],
        "related": [],
        "implementation": [],
        "unknown": [],
    }

    for link in issuelinks or []:
        linked = link.get("outwardIssue") or link.get("inwardIssue")
        if not linked:
            continue

        link_type_name = (link.get("type") or {}).get("name", "")
        category = LINK_TYPE_MAP.get(link_type_name, "unknown")

        entry: dict = {
            "type": link_type_name,
            "category": category,
            "direction": "outward" if link.get("outwardIssue") else "inward",
            "key": linked.get("key", ""),
            "summary": (linked.get("fields") or {}).get("summary", ""),
            "project": linked.get("key", "").split("-")[0],
            "status": ((linked.get("fields") or {}).get("status") or {}).get("name", ""),
        }
        classified[category].append(entry)

    return classified


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_field(fields: dict, field_id: Optional[str]) -> Optional[Any]:
    """Return the raw value of a field by its Jira field ID."""
    if not field_id:
        return None
    return fields.get(field_id)


def _as_text(value: Any) -> str:
    """
    Coerce a Jira field value to a plain string.

    Handles strings, dicts (name/value/key), lists, and None.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(
            value.get("name")
            or value.get("value")
            or value.get("key")
            or ""
        )
    if isinstance(value, list):
        parts = [_as_text(item) for item in value]
        return ", ".join(p for p in parts if p)
    return str(value)


def _get_epic_key(fields: dict, jira_field_map: dict[str, str]) -> Optional[str]:
    """
    Resolve the epic link using configured field ID first, then fall back
    to the parent field used by next-gen Jira projects.
    """
    epic_field_id = jira_field_map.get("epic_link", "customfield_10014")
    epic = _resolve_field(fields, epic_field_id)
    if epic and isinstance(epic, str):
        return epic

    # Next-gen projects store the parent directly
    parent = fields.get("parent")
    if isinstance(parent, dict):
        return parent.get("key")

    return None


def _extract_substantive_comments(comment_container: dict) -> list[str]:
    """Extract first 2-3 substantive (>50 words) comments, skipping bots."""
    comments_list = comment_container.get("comments", [])
    substantive: list[str] = []
    bot_patterns = ["atlassian-bot", "jira-bot", "automation", "webhook"]

    for comment in comments_list:
        if len(substantive) >= 3:
            break
        author = (comment.get("author") or {}).get("displayName", "").lower()
        if any(p in author for p in bot_patterns):
            continue
        body = comment.get("body", "")
        if isinstance(body, dict):
            body = _extract_adf_text(body)
        body = body.strip()
        word_count = len(body.split())
        if word_count >= 50:
            substantive.append(body[:2000])

    return substantive


def _clean_comment_text(text: str) -> str:
    """
    Light cleaning for comment text: strip excessive whitespace and
    remove obvious boilerplate markers.
    """
    import re
    # Collapse runs of whitespace / newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _extract_adf_text(adf: dict) -> str:
    """Recursively extract plain text from Atlassian Document Format."""
    parts: list[str] = []
    if isinstance(adf, dict):
        if adf.get("type") == "text":
            parts.append(adf.get("text", ""))
        for child in adf.get("content", []):
            parts.append(_extract_adf_text(child))
    elif isinstance(adf, list):
        for item in adf:
            parts.append(_extract_adf_text(item))
    return " ".join(p for p in parts if p)


def _extract_product_list(raw_value: Any) -> tuple[list[dict], list[str], list[str]]:
    """
    Normalise a Jira custom field value into (raw_list, ids, names).

    Handles: None, str, dict (single item with id/name), list of dicts.
    """
    if not raw_value:
        return [], [], []

    if isinstance(raw_value, str):
        return [{"name": raw_value}], [], [raw_value]

    if isinstance(raw_value, dict):
        item_id = str(raw_value.get("id", raw_value.get("key", "")))
        item_name = str(raw_value.get("name", raw_value.get("value", "")))
        return [raw_value], [item_id] if item_id else [], [item_name] if item_name else []

    if isinstance(raw_value, list):
        raw_list: list[dict] = []
        ids: list[str] = []
        names: list[str] = []
        for item in raw_value:
            if isinstance(item, dict):
                raw_list.append(item)
                item_id = str(item.get("id", item.get("key", "")))
                item_name = str(item.get("name", item.get("value", "")))
                if item_id:
                    ids.append(item_id)
                if item_name:
                    names.append(item_name)
            elif isinstance(item, str):
                raw_list.append({"name": item})
                names.append(item)
        return raw_list, ids, names

    return [], [], []
