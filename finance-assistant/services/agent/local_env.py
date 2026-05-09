"""Rewrite Compose-only hostnames in os.environ when running bare-metal."""

from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse


def apply_local_dev_url_overrides() -> None:
    """Maps Docker service DNS names to localhost for local runs."""

    def rewrite(raw: str, default_port: int) -> str:
        stripped = raw.strip().rstrip("/")
        parsed = urlparse(stripped if "://" in stripped else f"http://{stripped}")
        host = (parsed.hostname or "").lower()
        if host not in frozenset({"mock-api", "ollama"}):
            return stripped.rstrip("/")
        port = parsed.port or default_port
        return urlunparse(("http", f"127.0.0.1:{port}", parsed.path or "", "", "", "")).rstrip("/")

    if banking := os.environ.get("BANKING_API_URL", "").strip():
        os.environ["BANKING_API_URL"] = rewrite(banking, 8001)

    if olla := os.environ.get("OLLAMA_BASE_URL", "").strip():
        os.environ["OLLAMA_BASE_URL"] = rewrite(olla, 11434)

    if rus := os.environ.get("REDIS_URL", "").strip():
        p = urlparse(rus)
        if (p.hostname or "").lower() == "redis":
            port = p.port or 6379
            netloc = f"127.0.0.1:{port}"
            if p.username is not None:
                pwpart = p.password if p.password is not None else ""
                netloc = f"{p.username}:{pwpart}@{netloc}" if pwpart else f"{p.username}@{netloc}"
            fixed = urlunparse(
                (
                    p.scheme or "redis",
                    netloc,
                    p.path or "",
                    "",
                    p.query or "",
                    p.fragment or "",
                )
            )
            os.environ["REDIS_URL"] = fixed
