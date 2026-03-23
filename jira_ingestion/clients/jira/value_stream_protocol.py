"""Protocol (abstract base) for value-stream fetchers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ValueStreamFetcher(ABC):
    """Interface that any value-stream data source must implement."""

    @abstractmethod
    async def authenticate(self) -> None: ...

    @abstractmethod
    async def get_ticket_data(self, ticket_id: str) -> Dict[str, Any]: ...

    @abstractmethod
    async def fetch_attachment_content(
        self, attachments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def download_attachment(self, url: str, dest_path: str) -> None: ...
