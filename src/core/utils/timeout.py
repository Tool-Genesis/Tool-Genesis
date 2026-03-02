import functools
import threading
import queue
import asyncio
from typing import Optional

def with_timeout(timeout: Optional[float] = None):
    r"""Decorator that adds timeout functionality to functions.

    - If `timeout` is None, it tries to read `self.timeout` from the bound instance.
    - Sync functions run in a daemon thread; timeout returns a string message.
    - Async functions use `asyncio.wait_for`; timeout returns a string message.
    - Exceptions raised by the wrapped function are propagated (no IndexError).

    Supports both usages:
        @with_timeout
        @with_timeout()
        @with_timeout(5)
    """

    # Allow bare decorator usage: @with_timeout
    if callable(timeout):
        func, timeout = timeout, None
        return _decorate(func, timeout)

    def decorator(func):
        return _decorate(func, timeout)
    return decorator


def _decorate(func, timeout: Optional[float]):
    is_coro = asyncio.iscoroutinefunction(func)

    if is_coro:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Resolve effective timeout (param > instance attr > None)
            effective_timeout = timeout
            if effective_timeout is None and args:
                effective_timeout = getattr(args[0], "timeout", None)

            if effective_timeout is None:
                return await func(*args, **kwargs)

            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=effective_timeout)
            except asyncio.TimeoutError:
                return (
                    f"Function `{func.__name__}` execution timed out, "
                    f"exceeded {effective_timeout} seconds."
                )
        return async_wrapper

    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Resolve effective timeout (param > instance attr > None)
            effective_timeout = timeout
            if effective_timeout is None and args:
                effective_timeout = getattr(args[0], "timeout", None)

            if effective_timeout is None:
                return func(*args, **kwargs)

            # Use a queue to capture result or exception from the worker thread
            q: "queue.Queue[tuple[str, object]]" = queue.Queue(maxsize=1)

            def target():
                try:
                    res = func(*args, **kwargs)
                    q.put(("ok", res))
                except BaseException as e:
                    q.put(("err", e))

            t = threading.Thread(target=target, daemon=True)
            t.start()
            t.join(effective_timeout)

            if t.is_alive():
                # Thread will continue in background (daemon=True)
                return (
                    f"Function `{func.__name__}` execution timed out, "
                    f"exceeded {effective_timeout} seconds."
                )

            # If finished, propagate result or exception
            try:
                status, payload = q.get_nowait()
            except queue.Empty:
                # Extremely rare: thread finished but didn't put; fallback
                status, payload = "ok", None

            if status == "err":
                raise payload  # re-raise original exception
            return payload

        return wrapper