from __future__ import annotations

import json
import logging
import os

import redis

_log = logging.getLogger("agent.memory.redis_memory")


class RedisMemory:
    def __init__(self) -> None:
        use_fake = os.getenv("REDIS_USE_FAKEREDIS", "").lower() in ("1", "true", "yes")

        if use_fake:
            import fakeredis

            self.client = fakeredis.FakeRedis(decode_responses=False)
        else:
            self.client = redis.Redis.from_url(os.environ["REDIS_URL"])

        self.ttl = int(os.getenv("REDIS_TTL_SECONDS", "86400"))

    def add_message(self, session_id: str, role: str, content: str) -> None:
        from datetime import datetime, timezone

        payload = {"role": role, "content": content, "timestamp": datetime.now(timezone.utc).isoformat()}
        key = f"chat:{session_id}"
        blob = json.dumps(payload, ensure_ascii=False)
        self.client.rpush(key, blob)
        self.client.expire(key, self.ttl)
        _log.info({"event": "memory_push", **{"session_id": session_id, "role": role}})

    def get_history(self, session_id: str, last_n: int = 5) -> list:
        span = max(1, int(last_n)) if last_n else 5

        entries = []
        blobs = self.client.lrange(f"chat:{session_id}", -span, -1)
        for blob in blobs:
            if isinstance(blob, (bytes, bytearray)):
                blob = blob.decode("utf-8")
            entries.append(json.loads(blob))
        _log.info({"event": "memory_tail", "session_id": session_id, "count": len(entries)})
        return entries

    def get_full_history(self, session_id: str) -> list:
        rows = []
        for blob in self.client.lrange(f"chat:{session_id}", 0, -1):
            if isinstance(blob, (bytes, bytearray)):
                blob = blob.decode("utf-8")
            rows.append(json.loads(blob))
        _log.info({"event": "memory_dump", "session_id": session_id, "count": len(rows)})
        return rows

    def clear_session(self, session_id: str) -> None:
        self.client.delete(f"chat:{session_id}")
        _log.info({"event": "memory_cleared", **{"session_id": session_id}})