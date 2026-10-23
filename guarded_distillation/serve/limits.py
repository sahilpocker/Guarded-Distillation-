"""Rate limits for the public endpoint. The expensive thing is a live Claude call (a cache miss), so those get strict
per-visitor and global limits on top of the daily dollar cap; plain requests get a generous per-visitor ceiling."""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque


class SlidingWindow:
    def __init__(self, max_events: int, window_s: float):
        self.max, self.window = max_events, window_s
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self.lock:
            q = self.hits[key]
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.max:
                return False
            q.append(now)
            if len(self.hits) > 5000:                      # bounded memory: forget idle visitors
                for k in [k for k, v in self.hits.items() if not v or v[-1] <= now - self.window][:2500]:
                    del self.hits[k]
            return True

    def retry_after(self, key: str, now: float | None = None) -> int:
        now = time.time() if now is None else now
        q = self.hits.get(key)
        return max(1, int(q[0] + self.window - now) + 1) if q else 1


class Limits:
    def __init__(self):
        e = os.environ.get
        self.requests = SlidingWindow(int(e("REQ_PER_IP_PER_MIN", 300)), 60)
        self.live_ip = SlidingWindow(int(e("LIVE_PER_IP_PER_HOUR", 10)), 3600)
        self.live_all = SlidingWindow(int(e("LIVE_GLOBAL_PER_HOUR", 60)), 3600)

    def allow_live(self, ip: str) -> bool:
        return self.live_ip.allow(ip) and self.live_all.allow("*")
