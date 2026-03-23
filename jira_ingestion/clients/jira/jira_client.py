"""Low-level async Jira REST client."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
        response = await self._client.get(
            f"{self.base_url}/rest/api/2/myself"
        )
        response.raise_for_status()
        me = response.json()
        logger.info("Authenticated as %s", me.get("displayName", me.get("name")))

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

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
