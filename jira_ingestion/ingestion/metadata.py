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

def extract_product_fields(
    ticket_fields: dict,
    jira_field_map: dict,
) -> dict:
    """
    Extract impacted product and IT product labels from Jira custom fields.

    Returns a dict safe for the supervision layer — these are labels,
    not retrieval text.

    Args:
        ticket_fields:  The 'fields' dict from the Jira API response.
        jira_field_map: Mapping of logical names to customfield_* IDs.

    Returns:
        {
            "impacted_products":    ProductLabels,
            "impacted_it_products": ProductLabels,
            "requesting_org":       str,
            "delivery_org":         str,
        }
    """
    fields = ticket_fields or {}

    def _extract_label_list(field_id: Optional[str]) -> dict:
        """Extract a list-of-objects field into raw/ids/names."""
        raw = _resolve_field(fields, field_id) if field_id else None
        if not raw:
            return {"raw": [], "ids": [], "names": []}
        if not isinstance(raw, list):
            raw = [raw]
        ids: list[str] = []
        names: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                item_id = str(item.get("id") or item.get("key") or "")
                item_name = str(item.get("name") or item.get("value") or item.get("displayName") or "")
            elif isinstance(item, str):
                item_id = ""
                item_name = item
            else:
                item_id = ""
                item_name = str(item)
            if item_id:
                ids.append(item_id)
            if item_name:
                names.append(item_name)
        return {"raw": raw, "ids": ids, "names": names}

    impacted_products_field = jira_field_map.get("impacted_products")
    impacted_it_products_field = jira_field_map.get("impacted_it_products")
    requesting_org_field = jira_field_map.get("requesting_org")
    delivery_org_field = jira_field_map.get("delivery_org")

    return {
        "impacted_products": _extract_label_list(impacted_products_field),
        "impacted_it_products": _extract_label_list(impacted_it_products_field),
        "requesting_org": _as_text(_resolve_field(fields, requesting_org_field)),
        "delivery_org": _as_text(_resolve_field(fields, delivery_org_field)),
    }


_COMMENT_CHATTER_RE = [
    __import__("re").compile(p, __import__("re").IGNORECASE)
    for p in [
        r"^(moved|transitioned|changed status|updated|assigned|resolved|closed|reopened)\b",
        r"^(done|ok|ack|acknowledged|noted|thanks|cheers|\+1|-1|approved|lgtm)\W*$",
        r"^\s*(see attached|see comments|see above|as per|per above|as discussed)\b",
        r"^(sent|forwarded|cc:|fyi\b)",
    ]
]


def extract_comments_enriched(comment_container: dict) -> dict:
    """
    Extract all comments with raw + cleaned representations.

    Cleaning steps:
      - Strip Jira wiki markup (reuses description cleaner)
      - Normalize whitespace
      - Detect and exclude trivial operational chatter

    Returns:
        CommentsEnriched-compatible dict with:
            - comments_raw:       list of CommentRecord dicts (all non-bot)
            - comments_cleaned:   list of substantive comment strings (>= 50 words)
            - comment_count:      total non-bot comment count
            - substantive_count:  comments with >= 50 words
            - important_spans:    key sentences from substantive comments
    """
    from .description import clean_jira_markup

    comments_list = (comment_container or {}).get("comments", [])
    bot_patterns = ["atlassian-bot", "jira-bot", "automation", "webhook"]

    comments_raw: list[dict] = []
    comments_cleaned: list[str] = []
    important_spans: list[str] = []

    for comment in comments_list:
        author_obj = comment.get("author") or {}
        author = author_obj.get("displayName", author_obj.get("name", ""))
        if any(p in author.lower() for p in bot_patterns):
            continue

        body = comment.get("body", "")
        if isinstance(body, dict):
            body = _extract_adf_text(body)
        body_raw = body.strip()

        # Clean: strip Jira markup, collapse whitespace
        body_cleaned = clean_jira_markup(body_raw) if body_raw else ""

        # Exclude trivial operational chatter before counting
        word_count_cleaned = len(body_cleaned.split())
        is_chatter = word_count_cleaned < 10 or any(
            p.search(body_cleaned) for p in _COMMENT_CHATTER_RE
        )

        word_count = len(body_raw.split())
        is_substantive = word_count >= 50 and not is_chatter

        record: dict = {
            "comment_id": comment.get("id", ""),
            "author": author,
            "created": comment.get("created", ""),
            "body_raw": body_raw,
            "body_cleaned": body_cleaned,
            "word_count": word_count,
            "is_substantive": is_substantive,
        }
        comments_raw.append(record)

        if is_substantive:
            comments_cleaned.append(body_cleaned[:2000])
            # Key sentences: split on period, keep those >= 8 words
            sentences = [s.strip() for s in body_cleaned.split(".") if len(s.strip().split()) >= 8]
            important_spans.extend(sentences[:2])

    return {
        "comments_raw": comments_raw,
        "comments_cleaned": comments_cleaned[:5],
        "comment_count": len(comments_raw),
        "substantive_count": sum(1 for c in comments_raw if c["is_substantive"]),
        "important_spans": important_spans[:10],
    }


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
    issue_type = (fields.get("issuetype") or {}).get("name", "")
    status = (fields.get("status") or {}).get("name", "")
    resolution = (fields.get("resolution") or {}).get("name", "")

    # Tier 2 — config-driven custom fields (deterministic, not heuristic)
    business_unit = _as_text(
        _resolve_field(fields, jira_field_map.get("business_unit", "customfield_10002"))
    )
    product_area = _as_text(
        _resolve_field(fields, jira_field_map.get("product_area", "customfield_10003"))
    )
    requesting_org = _as_text(
        _resolve_field(fields, jira_field_map.get("requesting_org"))
    )
    delivery_org = _as_text(
        _resolve_field(fields, jira_field_map.get("delivery_org"))
    )
    priority = (fields.get("priority") or {}).get("name", "")
    epic_key = _get_epic_key(fields, jira_field_map)

    # Tier 3 — comments (backward-compat: top-3 substantive strings)
    substantive_comments = _extract_comments(fields.get("comment") or {})

    # Build metadata text for BM25 + embedding (retrieval-safe — no label leakage)
    meta_parts: list[str] = [f"{ticket_key}: {summary}"]
    if issue_type:
        meta_parts.append(f"Type: {issue_type}")
    if status:
        meta_parts.append(f"Status: {status}")
    if components:
        meta_parts.append(f"Components: {', '.join(components)}")
    if labels:
        meta_parts.append(f"Labels: {', '.join(labels)}")
    if business_unit:
        meta_parts.append(f"Business Unit: {business_unit}")
    if product_area:
        meta_parts.append(f"Product Area: {product_area}")
    if requesting_org:
        meta_parts.append(f"Requesting Org: {requesting_org}")
    if delivery_org:
        meta_parts.append(f"Delivery Org: {delivery_org}")
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
        "issue_type": issue_type,
        "status": status,
        "resolution": resolution,
        "business_unit": business_unit,
        "product_area": product_area,
        "requesting_org": requesting_org,
        "delivery_org": delivery_org,
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
