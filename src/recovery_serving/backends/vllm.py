"""Integration boundary; this module deliberately does not pretend to switch precision."""

from .base import Generation


class VLLMBackend:
    async def generate(self, request) -> Generation:
        raise NotImplementedError(
            "Add a streaming client for a fixed-precision baseline, then implement "
            "internal scheduler/cache instrumentation in a pinned vLLM fork. "
            "Client request ordering alone cannot control KV placement."
        )
