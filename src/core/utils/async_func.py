import asyncio
from functools import wraps
def async_retry(backoff=2, max_delay=60):

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = backoff
            attempt = 0
            while True:
                try:
                    attempt += 1
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    print(f"Attempt {attempt} failed with error: {e}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, max_delay)
        return wrapper
    return decorator