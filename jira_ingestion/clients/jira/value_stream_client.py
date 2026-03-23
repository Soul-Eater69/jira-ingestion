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
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


class JiraValueStreamClient:
    """Async Jira client that exposes ticket data and attachment text extraction."""

    def __init__(self, base_url: str, token: str, verify_ssl: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.verify_ssl = verify_ssl
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def authenticate(self) -> None:
        """Create an authenticated aiohttp session and verify credentials."""
        connector = aiohttp.TCPConnector(ssl=self.verify_ssl)
        self._session = aiohttp.ClientSession(
            connector=connector,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        # Verify credentials by hitting the /myself endpoint
        try:
            async with self._session.get(
                f"{self.base_url}/rest/api/2/myself"
            ) as resp:
                resp.raise_for_status()
                me = await resp.json()
                logger.info("Authenticated as %s", me.get("displayName", me.get("name")))
        except aiohttp.ClientResponseError as exc:
            await self.close()
            raise RuntimeError(
                f"Jira authentication failed ({exc.status}): {exc.message}"
            ) from exc

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "JiraValueStreamClient":
        await self.authenticate()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Core API calls
    # ------------------------------------------------------------------

    async def get_ticket_data(self, ticket_id: str) -> dict:
        """
        Fetch a Jira ticket and return structured data with:
          - themes:      list of linked issues (key, summary, status, link_type)
          - attachments: list of attachment metadata dicts
          - fields:      raw Jira fields for downstream processing
        """
        if self._session is None:
            raise RuntimeError("Call authenticate() first.")

        url = f"{self.base_url}/rest/api/2/issue/{ticket_id}"
        params = {"expand": "attachment,issuelinks,comment,renderedFields"}

        async with self._session.get(url, params=params) as resp:
            resp.raise_for_status()
            raw = await resp.json()

        fields = raw.get("fields", {})

        # -- Themes: all linked issues (VS links, related ideas, etc.) --
        themes: list[dict] = []
        for link in fields.get("issuelinks", []):
            linked = link.get("outwardIssue") or link.get("inwardIssue")
            if not linked:
                continue
            themes.append(
                {
                    "key": linked["key"],
                    "summary": linked["fields"].get("summary", ""),
                    "status": linked["fields"].get("status", {}).get("name", ""),
                    "link_type": link.get("type", {}).get("name", ""),
                    "direction": "outward" if link.get("outwardIssue") else "inward",
                }
            )

        # -- Attachments: Jira metadata (no download yet) --
        attachments: list[dict] = []
        reporter_name = (fields.get("reporter") or {}).get("displayName", "")
        for att in fields.get("attachment", []):
            attachments.append(
                {
                    "id": att["id"],
                    "filename": att["filename"],
                    "mimeType": att.get("mimeType", ""),
                    "size": att.get("size", 0),
                    "content": att["content"],  # download URL
                    "created": att.get("created", ""),
                    "author": (att.get("author") or {}).get("displayName", ""),
                    "is_reporter_upload": (
                        (att.get("author") or {}).get("displayName", "") == reporter_name
                    ),
                }
            )

        return {
            "key": raw["key"],
            "themes": themes,
            "attachments": attachments,
            "fields": fields,  # raw fields for full pipeline use
        }

    async def fetch_attachment_content(self, attachments: list[dict]) -> list[dict]:
        """
        Download each attachment and extract its text via MarkItDown.

        Returns a list of dicts:
            {"filename": str, "mime_type": str, "text_content": str, "error": str|None}
        """
        # Import here so the rest of the module works even if markitdown isn't installed
        try:
            from markitdown import MarkItDown  # type: ignore
            md = MarkItDown()
        except ImportError:
            logger.warning("markitdown not installed; attachment text extraction disabled.")
            md = None

        results: list[dict] = []
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
                file_bytes = await self._download_file(url)
                if md is None:
                    raise ImportError("markitdown not available")
                text = self._markitdown_extract(md, file_bytes, filename, mime_type)
                results.append({"filename": filename, "mime_type": mime_type, "text_content": text, "error": None})
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to extract %s: %s", filename, exc)
                results.append({"filename": filename, "mime_type": mime_type, "text_content": "", "error": str(exc)})

        return results

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    async def _download_file(self, url: str) -> bytes:
        """Download a file and return its raw bytes."""
        if self._session is None:
            raise RuntimeError("Call authenticate() first.")
        async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=60.0)) as resp:
            resp.raise_for_status()
            return await resp.read()

    @staticmethod
    def _markitdown_extract(md: Any, file_bytes: bytes, filename: str, mime_type: str = "") -> str:
        """Convert file bytes to Markdown text via MarkItDown."""
        try:
            from markitdown import StreamInfo  # type: ignore
            ext = f".{filename.rsplit('.', 1)[-1]}" if "." in filename else ""
            stream_info = StreamInfo(
                mimetype=mime_type or None,
                extension=ext or None,
                filename=filename or None,
            )
            result = md.convert_stream(io.BytesIO(file_bytes), stream_info=stream_info)
        except ImportError:
            # Fallback for older markitdown versions without StreamInfo
            stream = io.BytesIO(file_bytes)
            stream.name = filename
            result = md.convert_stream(stream)
        return result.text_content or ""

    # ------------------------------------------------------------------
    # Convenience: download raw bytes for structured extraction
    # ------------------------------------------------------------------

    async def download_attachment(self, attachment: dict) -> bytes:
        """Download a single attachment and return its raw bytes."""
        return await self._download_file(attachment["content"])
