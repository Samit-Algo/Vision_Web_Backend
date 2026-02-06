"""
Async helpers for running async code from sync context.
Used when tools are sync (called by LLM) but underlying operations are async.
"""

import asyncio
from typing import TypeVar

T = TypeVar("T")


def run_async(coro) -> T:
    """
    Run coroutine from sync code. Works inside or outside a running event loop.
    Use when a sync tool needs to call async repository/API.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)
