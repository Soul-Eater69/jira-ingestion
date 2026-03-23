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
from typing import Any, Dict, List

import httpx

from .jira_client import JIRARestClient
from .value_stream_protocol import ValueStreamFetcher

logger = logging.getLogger(__name__)


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

    async def get_ticket_data(self, ticket_id: str) -> Dict[str, Any]:
        """
        Fetch a Jira ticket and return structured data with:
          - attachments: list of attachment metadata dicts
          - themes:      list of linked issues (key, summary, status)
        """
        issue = await self.client.get_issue_by_key(
            ticket_id, fields=["attachment", "issuelinks"]
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

        return {"attachments": attachments, "themes": themes, "fields": fields}

    # ------------------------------------------------------------------
    # Attachment handling
    # ------------------------------------------------------------------

    async def download_attachment(self, url_or_att: Any, dest_path: str = "") -> Any:
        """
        Download a single attachment.

        Supports two calling conventions:
          - download_attachment(url: str, dest_path: str) — saves to disk
          - download_attachment(att: dict) — returns raw bytes (pipeline compat)
        """
        headers = {"Authorization": f"Bearer {self.token}"}
        if isinstance(url_or_att, dict):
            url = url_or_att.get("content", "")
        else:
            url = url_or_att

        async with httpx.AsyncClient(verify=self.verify_ssl) as client:
            response = await client.get(url, headers=headers, timeout=60.0)
            response.raise_for_status()

        if dest_path:
            with open(dest_path, "wb") as f:
                f.write(response.content)
        else:
            return response.content

    async def fetch_attachment_content(
        self,
        attachments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Download each attachment and extract its text content using MarkItDown.

        Args:
            attachments: List of attachment dicts from get_ticket_data().
                Each dict should have at least 'content' (URL),
                'filename', and optionally 'mimeType'.

        Returns:
            List of dicts with keys:
                - filename:     original attachment filename
                - mime_type:    MIME type reported by Jira
                - text_content: extracted Markdown text (empty string on failure)
                - error:        error message if extraction failed, else None
        """
        try:
            from markitdown import MarkItDown  # type: ignore

            md = MarkItDown()
        except ImportError:
            logger.warning("markitdown not installed; attachment text extraction disabled.")
            md = None

        headers = {"Authorization": f"Bearer {self.token}"}
        results: List[Dict[str, Any]] = []

        async with httpx.AsyncClient(verify=self.verify_ssl) as client:
            for att in attachments:
                url = att.get("content", "")
                filename = att.get("filename", "")
                mime_type = att.get("mimeType", "")

                if not url:
                    results.append({
                        "filename": filename,
                        "mime_type": mime_type,
                        "text_content": "",
                        "error": "No content URL",
                    })
                    continue

                try:
                    response = await client.get(url, headers=headers, timeout=60.0)
                    response.raise_for_status()

                    # Determine file extension for better format detection
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
                        "text_content": result.text_content,
                        "error": None,
                    })
                except Exception as exc:
                    results.append({
                        "filename": filename,
                        "mime_type": mime_type,
                        "text_content": "",
                        "error": str(exc),
                    })

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_stream_info(
        mime_type: str, ext: str, filename: str
    ) -> Any:
        """Build a MarkItDown StreamInfo for format detection."""
        from markitdown import StreamInfo  # type: ignore

        return StreamInfo(
            mimetype=mime_type or None,
            extension=ext or None,
            filename=filename or None,
        )
