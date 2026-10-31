"""RateLimiter, retry/backoff, .part + atomic rename, listing cache with TTL.

Shared by every downloader. RMI-style servers rate-limit the whole egress IP,
so the limiter is per-host and conservative by default.

Not implemented yet — see the Target design tab of the redesign plan.
"""
from __future__ import annotations
