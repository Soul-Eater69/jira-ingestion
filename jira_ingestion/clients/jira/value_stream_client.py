"""
Async Jira client with MarkItDown-powered attachment extraction.

Usage:
    client = JiraValueStreamClient(base_url, token, verify_ssl=False)
    await client.authenticate()
    data = await client.get_ticket_data("IDEA-1234")
    contents = await client.fetch_attachment_content(data["attachments"])
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------


class ValueStreamFetcher(ABC):
    """Interface that any value-stream data source must implement."""

    @abstractmethod
    async def authenticate(self) -> None: ...

    @abstractmethod
    async def get_ticket_data(self, ticket_id: str, config: Optional[Any] = None) -> Dict[str, Any]: ...

    @abstractmethod
    async def fetch_attachment_content(
        self, attachments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def download_attachment(self, url_or_att: Any, dest_path: str = "") -> Any: ...


# ---------------------------------------------------------------------------
# Low-level REST client
# ---------------------------------------------------------------------------


class JIRARestClient:
    """Thin async wrapper around the Jira REST API v2."""

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        api_token: str,
        verify_ssl: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.api_token = api_token
        self.verify_ssl = verify_ssl
        self._client: Optional[httpx.AsyncClient] = None

    async def _authenticate(self) -> None:
        """Create an authenticated httpx client and verify credentials."""
        self._client = httpx.AsyncClient(
            verify=self.verify_ssl,
            headers={
                "Authorization": f"Bearer {self.auth_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        response = await self._client.get(f"{self.base_url}/rest/api/2/myself")
        response.raise_for_status()
        me = response.json()
        logger.info("Authenticated as %s", me.get("displayName", me.get("name")))

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def get_issue_by_key(
        self,
        key: str,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch an issue by key, optionally restricting to *fields*."""
        if self._client is None:
            raise RuntimeError("Call _authenticate() first.")
        url = f"{self.base_url}/rest/api/2/issue/{key}"
        params: Dict[str, str] = {}
        if fields:
            params["fields"] = ",".join(fields)
        response = await self._client.get(url, params=params)
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# High-level value-stream client
# ---------------------------------------------------------------------------


class JiraValueStreamClient(ValueStreamFetcher):
    """Async Jira client that exposes ticket data and attachment text extraction."""

    def __init__(self, base_url: str, token: str, verify_ssl: bool = False) -> None:
        self.base_url = base_url
        self.token = token
        self.verify_ssl = verify_ssl
        self.client = JIRARestClient(
            base_url=base_url,
            auth_token=token,
            api_token=token,
            verify_ssl=verify_ssl,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def authenticate(self) -> None:
        await self.client._authenticate()

    async def close(self) -> None:
        await self.client.close()

    async def __aenter__(self) -> "JiraValueStreamClient":
        await self.authenticate()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Core API calls
    # ------------------------------------------------------------------

    # Non-custom base fields — always requested regardless of tenant config.
    _BASE_FIELDS: List[str] = [
        "summary", "description", "reporter", "assignee",
        "created", "updated", "status", "priority", "issuetype",
        "labels", "components", "attachment", "issuelinks",
        "comment", "parent",
    ]

    def _build_ticket_fields(self, config: Optional[Any] = None) -> List[str]:
        """
        Build the Jira fields list from base fields + configured custom fields.

        Custom field IDs come from config.jira_field_map so the client stays
        portable across Jira tenants without code changes.
        """
        base = list(self._BASE_FIELDS)
        jira_field_map: dict = getattr(config, "jira_field_map", {}) if config else {}
        custom = sorted({
            v for v in jira_field_map.values()
            if isinstance(v, str) and v.startswith("customfield_")
        })
        return base + custom

    async def get_ticket_data(self, ticket_id: str, config: Optional[Any] = None) -> Dict[str, Any]:
        """
        Fetch a Jira ticket and return structured data with:
          - key:         ticket key (e.g. "IDEA-1234")
          - fields:      all ticket fields (summary, description, links, etc.)
          - attachments: list of attachment metadata dicts
          - themes:      list of linked issues (key, summary, status)
        """
        issue = await self.client.get_issue_by_key(
            ticket_id, fields=self._build_ticket_fields(config)
        )
        fields = issue.get("fields", {})

        attachments = issue.get("fields", {}).get("attachment", [])

        # -- Themes: linked issues filtered by "implements" relationship --
        themes: List[Dict[str, Any]] = []
        issuelinks = issue.get("fields", {}).get("issuelinks", [])
        for link in issuelinks:
            if (
                link.get("type", {}).get("outward") == "implements"
                and "outwardIssue" in link
            ):
                fields_data = link["outwardIssue"].get("fields", {})
                summary = fields_data.get("summary")
                key = link["outwardIssue"].get("key")
                status = fields_data.get("status", {}).get("name")
                themes.append({"key": key, "summary": summary, "status": status})

            if (
                link.get("type", {}).get("inward") == "implemented by"
                and "inwardIssue" in link
            ):
                fields_data = link["inwardIssue"].get("fields", {})
                summary = fields_data.get("summary")
                key = link["inwardIssue"].get("key")
                status = fields_data.get("status", {}).get("name")
                themes.append({"key": key, "summary": summary, "status": status})

        return {
            "key": issue.get("key", ticket_id),
            "fields": fields,
            "attachments": attachments,
            "themes": themes,
        }

    # ------------------------------------------------------------------
    # Attachment handling
    # ------------------------------------------------------------------

    async def download_attachment(self, url_or_att: Any, dest_path: str = "") -> Any:
        """
        Download a single attachment.

        Supports two calling conventions:
          - download_attachment(url: str, dest_path: str) — saves to disk
          - download_attachment(att: dict) — returns raw bytes (pipeline compat)

        Reuses the authenticated httpx client created by authenticate() to
        avoid the overhead and auth inconsistency of spawning a new client
        per download.
        """
        if isinstance(url_or_att, dict):
            url = url_or_att.get("content", "")
        else:
            url = url_or_att

        if not url:
            raise ValueError("No download URL found in attachment")

        http_client = self.client._client
        if http_client is None:
            raise RuntimeError("Call authenticate() before downloading attachments.")

        response = await http_client.get(url, timeout=120.0)
        response.raise_for_status()

        if dest_path:
            with open(dest_path, "wb") as f:
                f.write(response.content)
            return None
        return response.content

    async def fetch_attachment_content(
        self,
        attachments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Download each attachment and extract its text content using MarkItDown.

        Returns a list of dicts with keys:
          - filename:     original attachment filename
          - mime_type:    MIME type reported by Jira
          - text_content: extracted Markdown text (empty string on failure)
          - error:        None | 'no_content_url' | 'markitdown_not_installed'
                          | '<ExcType>: <message>'

        Behaviour when markitdown is unavailable:
          - ingestion continues without raising
          - each attachment gets error='markitdown_not_installed'

        Reuses the authenticated Jira session — no new clients created.
        """
        try:
            from markitdown import MarkItDown  # type: ignore
            md: Optional[Any] = MarkItDown()
        except ImportError:
            logger.warning("markitdown not installed; attachment text extraction disabled.")
            md = None

        http_client = self.client._client
        if http_client is None:
            raise RuntimeError("Call authenticate() before fetching attachment content.")

        results: List[Dict[str, Any]] = []
        for att in attachments:
            url = att.get("content", "")
            filename = att.get("filename", "")
            mime_type = att.get("mimeType", "")

            if not url:
                results.append({
                    "filename": filename,
                    "mime_type": mime_type,
                    "text_content": "",
                    "error": "no_content_url",
                })
                continue

            if md is None:
                results.append({
                    "filename": filename,
                    "mime_type": mime_type,
                    "text_content": "",
                    "error": "markitdown_not_installed",
                })
                continue

            try:
                response = await http_client.get(url, timeout=120.0)
                response.raise_for_status()
                ext = f".{filename.rsplit('.', 1)[-1]}" if "." in filename else ""
                stream_info = self._build_stream_info(
                    mime_type=mime_type, ext=ext, filename=filename
                )
                result = md.convert_stream(
                    io.BytesIO(response.content),
                    stream_info=stream_info,
                )
                results.append({
                    "filename": filename,
                    "mime_type": mime_type,
                    "text_content": result.text_content or "",
                    "error": None,
                })
            except Exception as exc:
                logger.exception("Attachment extraction failed for %s", filename)
                results.append({
                    "filename": filename,
                    "mime_type": mime_type,
                    "text_content": "",
                    "error": f"{type(exc).__name__}: {exc}",
                })

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_stream_info(mime_type: str, ext: str, filename: str) -> Any:
        """Build a MarkItDown StreamInfo for format detection. Returns None if unavailable."""
        try:
            from markitdown import StreamInfo  # type: ignore
        except ImportError:
            return None
        return StreamInfo(
            mimetype=mime_type or None,
            extension=ext or None,
            filename=filename or None,
        )
