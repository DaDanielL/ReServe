"""Interface for the future real precision-switching implementation."""

from enum import Enum
from typing import Protocol


class PrecisionMode(str, Enum):
    HIGH = "high"
    LOW = "low"


class PrecisionController(Protocol):
    async def request_mode(self, mode: PrecisionMode) -> None:
        """Coordinate real weight changes with engine work and KV ownership."""
        ...

    def memory_state(self) -> dict:
        """Report active/reclaimable KV bytes and any outstanding restore transfers."""
        ...
