from dataclasses import dataclass
from typing import Protocol

from ..workload import Request


@dataclass(frozen=True)
class Generation:
    # All timestamps use time.perf_counter() in the same process.
    first_token_at: float
    finished_at: float
    output_tokens: int


class Backend(Protocol):
    async def generate(self, request: Request) -> Generation: ...
