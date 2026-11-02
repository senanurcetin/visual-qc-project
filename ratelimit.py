"""Best-effort per-client request limits for the public write endpoints.

State is in process memory: on serverless each instance keeps its own window, so this slows abuse
down rather than capping it globally (the row quotas in store.py are the hard bound). Clients are
identified by the connecting address; behind Vercel the edge-set X-Vercel-Forwarded-For is used.
A client-supplied X-Forwarded-For is never trusted, so it cannot be used to dodge the limit.
"""
from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Callable
from functools import wraps

from flask import jsonify, request


class RateLimiter:
    def __init__(self, calls: int, seconds: float, clock: Callable[[], float] = time.monotonic, max_keys: int = 10_000):
        self.calls, self.seconds, self.clock, self.max_keys = calls, seconds, clock, max_keys
        self.hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> float | None:
        """None if the call is allowed (and counted), else the seconds until it would be."""
        now = self.clock()
        window = self.hits.get(key)
        if window is None:
            # Never admit more than max_keys active clients. Prune expired keys
            # before admission; rejecting a new key is safer than unbounded
            # memory growth or evicting another client's active rate limit.
            if len(self.hits) >= self.max_keys:
                expired = [k for k, w in self.hits.items() if not w or now - w[-1] >= self.seconds]
                for expired_key in expired:
                    del self.hits[expired_key]
                if len(self.hits) >= self.max_keys:
                    retry = min(
                        (self.seconds - (now - w[-1]) for w in self.hits.values() if w),
                        default=self.seconds,
                    )
                    return max(0.0, retry)
            window = deque()
            self.hits[key] = window
        while window and now - window[0] >= self.seconds:
            window.popleft()
        if len(window) >= self.calls:
            return self.seconds - (now - window[0])
        window.append(now)
        return None


_limiters: dict[str, RateLimiter] = {}


def reset_limiters() -> None:
    _limiters.clear()


def client_key() -> str:
    if os.environ.get("VERCEL"):
        forwarded = request.headers.get("X-Vercel-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def rate_limited(name: str, calls: int, seconds: float = 60.0):
    """Decorator: 429 + Retry-After once a client exceeds `calls` per `seconds` on this route."""
    def decorator(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            limiter = _limiters.setdefault(name, RateLimiter(calls, seconds))
            retry = limiter.check(client_key())
            if retry is not None:
                response = jsonify({"error": "too many requests", "retry_after_seconds": round(retry, 1)})
                response.status_code = 429
                response.headers["Retry-After"] = str(max(1, int(retry + 0.999)))
                return response
            return view(*args, **kwargs)
        return wrapper
    return decorator
