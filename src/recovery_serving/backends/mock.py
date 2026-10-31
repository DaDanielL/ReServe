"""Artificial sleeps for harness testing. No LLM, CUDA, KV cache, or precision changes."""

import asyncio
from time import perf_counter

from ..workload import Request
from .base import Generation


class MockBackend:
    def __init__(self, prefill_s: float, token_s: float):
        self.prefill_s = prefill_s
        self.token_s = token_s

    async def generate(self, request: Request) -> Generation:
        await asyncio.sleep(self.prefill_s + self.token_s)
        first_token_at = perf_counter()
        await asyncio.sleep((request.output_tokens - 1) * self.token_s)
        return Generation(first_token_at, perf_counter(), request.output_tokens)
