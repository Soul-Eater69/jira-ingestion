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

def extract_metadata(ticket_fields: dict, ticket_key: str) -> dict:
    """
    Extract and structure all useful metadata from raw Jira fields.

    Args:
        ticket_fields: The 'fields' dict from Jira API response.
        ticket_key:    The issue key (e.g. "IDEA-1234").

    Returns:
        Flat metadata dict ready for pipeline use.
    """
    fields = ticket_fields or {}

    # Tier 1 — always extract
    summary = fields.get("summary", "")
    reporter_obj = fields.get("reporter") or {}
    reporter = reporter_obj.get("displayName", reporter_obj.get("name", ""))
    created = fields.get("created", "")
    labels: list[str] = [lbl for lbl in (fields.get("labels") or []) if isinstance(lbl, str)]
    components: list[str] = [
        c.get("name", "") for c in (fields.get("components") or []) if isinstance(c, dict)
    ]

    # Tier 2 — extract if available
    business_unit = _get_custom_field(fields, "Business Unit") or ""
    product_area = _get_custom_field(fields, "Product Area") or ""
    priority = (fields.get("priority") or {}).get("name", "")
    epic_key = _get_epic_key(fields)

    # Tier 3 — comments (first 2-3 substantive ones)
    substantive_comments = _extract_comments(fields.get("comment") or {})

    # Build metadata text for BM25 + embedding
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
        "epic_key": epic_key,
        "substantive_comments": substantive_comments,
        "metadata_text": ". ".join(meta_parts),
        # classified_links is set separately by classify_links()
    }


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

def _get_custom_field(fields: dict, field_name: str) -> Optional[str]:
    """
    Try to find a custom field by common name patterns.
    Jira custom fields use IDs like 'customfield_10001' but some may be
    accessible under their display name through the API.
    """
    # Direct name match in top-level fields
    for key, value in fields.items():
        if not key.startswith("customfield_"):
            continue
        if isinstance(value, str) and field_name.lower() in key.lower():
            return value
        if isinstance(value, dict):
            name = value.get("name") or value.get("value") or ""
            if name and field_name.lower() in name.lower():
                return name
            # Check if the field itself contains the display name
            if field_name.lower() in str(value).lower():
                return str(value.get("name") or value.get("value") or "")

    # Fallback: try the camelCase version as a direct field
    key_guess = "customfield_" + field_name.lower().replace(" ", "_")
    val = fields.get(key_guess)
    if val:
        return str(val) if not isinstance(val, dict) else (val.get("name") or val.get("value") or "")

    return None


def _get_epic_key(fields: dict) -> Optional[str]:
    """Try several known ways to get the epic key."""
    # Direct epic link field
    epic = fields.get("customfield_10014")  # common Jira epic link field ID
    if epic and isinstance(epic, str):
        return epic

    # Parent (next-gen projects)
    parent = fields.get("parent")
    if isinstance(parent, dict):
        return parent.get("key")

    return None


def _extract_comments(comment_container: dict) -> list[str]:
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
            # Atlassian Document Format (ADF)
            body = _extract_adf_text(body)
        body = body.strip()
        word_count = len(body.split())
        if word_count >= 50:
            substantive.append(body[:2000])

    return substantive


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
