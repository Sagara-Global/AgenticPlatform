"""ArtifactStore — binary artifact persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ArtifactStore(ABC):
    """Port for storing/retrieving binary artifacts (documents, images, etc.)."""

    @abstractmethod
    async def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """Store binary data."""
        ...

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        """Retrieve binary data. Returns None if not found."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete an artifact."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if an artifact exists."""
        ...
