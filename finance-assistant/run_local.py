"""Start mock-api, ingest RAG corpus, agent API, and Streamlit locally (no Docker)."""

from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
MOCK_API = ROOT / "services" / "mock-api"
AGENT_DIR = ROOT / "services" / "agent"
FRONTEND = ROOT / "services" / "frontend"


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    return env


def with_pythonpath(env: dict[str, str], service_root: Path) -> dict[str, str]:
    """Single directory on PYTHONPATH so uvicorn workers never pick up another service's `main.py`."""

    out = env.copy()
    out["PYTHONPATH"] = str(service_root.resolve())
    return out


def ensure_dotenv() -> None:
    dot = ROOT / ".env"
    example = ROOT / ".env.example"
    if not dot.exists() and example.exists():
        shutil.copyfile(example, dot)
        print(f"Copied {example.name} → .env (edit if needed)")
    elif not dot.exists():
        sys.stderr.write("Missing .env and .env.example — create .env manually.\n")
        sys.exit(1)


_PROCS: list[subprocess.Popen] = []


def terminate_all() -> None:
    for p in reversed(_PROCS):
        if p.poll() is None:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(p.pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                )
            else:
                try:
                    p.terminate()
                    p.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    p.kill()
                if p.poll() is None:
                    p.kill()


def _listening_pids_on_ports_win(ports: set[int]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {port: set() for port in ports}
    try:
        r = subprocess.run(
            ["cmd", "/c", "netstat", "-ano"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except OSError:
        return out
    for line in (r.stdout or "").splitlines():
        if "LISTENING" not in line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        addr = parts[1]
        if ":" not in addr:
            continue
        tail = addr.rsplit(":", 1)[-1].strip()
        if not tail.isdigit():
            continue
        lp = int(tail)
        if lp not in ports:
            continue
        pid = parts[-1].strip()
        if pid.isdigit():
            out[lp].add(int(pid))
    return out


def free_local_stack_ports_windows(ports: tuple[int, ...] = (8000, 8001, 8501)) -> None:
    to_free = set(ports)
    pids_seen: set[int] = set()

    for _ in range(12):
        mapping = _listening_pids_on_ports_win(to_free)
        batch: set[int] = set()
        for pset in mapping.values():
            batch |= pset
        killable = batch - pids_seen
        if not killable:
            break
        for pid in killable:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            pids_seen.add(pid)
        time.sleep(0.4)


def _http_reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=0.4) as r:  # noqa: S310
            r.read(1)
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError):
        return False


def preflight_conflict_or_die(*, after_auto_free: bool = False) -> None:
    checks = (
        ("http://127.0.0.1:8001/health", 8001),
        ("http://127.0.0.1:8000/health", 8000),
        ("http://127.0.0.1:8501/", 8501),
    )
    occupied = [port for url, port in checks if _http_reachable(url)]
    if not occupied:
        return

    ports_txt = ", ".join(str(p) for p in sorted(set(occupied)))
    manual = (
        "  PowerShell\n"
        "    Get-NetTCPConnection -LocalPort 8001,8000,8501 -State Listen "
        "| Select-Object -Expand OwningProcess -Unique\n"
        "    Stop-Process -Id <PID> -Force\n"
        "  cmd\n"
        "    netstat -ano | findstr \":8001\"\n"
        "    taskkill /PID <PID> /T /F\n"
    )

    if after_auto_free:
        sys.stderr.write(
            f"Ports still in use ({ports_txt}) after automatic cleanup on Windows.\n"
            "Another program may hold these ports, or taskkill could not terminate the process.\n\n"
            f"{manual}\n"
            "You can retry with:  python run_local.py --free-ports [...]\n",
        )
    else:
        extra = ""
        if sys.platform == "win32":
            extra = (
                "On Windows, stale listeners should have just been cleared; "
                "if you still see this, run:\n"
                "  python run_local.py --free-ports [...]\n"
            )
        sys.stderr.write(
            f"Conflict on ports {ports_txt}: something responds on stack URLs.\n"
            "Stop that process manually (see below).\n\n"
            f"{manual}\n"
            f"{extra}",
        )
    sys.exit(3)


def ensure_stack_ports_free() -> None:
    probe = (
        "http://127.0.0.1:8001/health",
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8501/",
    )
    if not any(_http_reachable(url) for url in probe):
        return
    if sys.platform == "win32":
        print("Stale listener(s) on stack ports — stopping PIDs listening on 8000/8001/8501 …")
        free_local_stack_ports_windows()
        time.sleep(0.6)
    preflight_conflict_or_die(after_auto_free=sys.platform == "win32")


def spawn(name: str, args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen:
    popen_kw: dict = {
        "args": args,
        "cwd": str(cwd),
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": None,
        "stderr": None,
    }

    if sys.platform == "win32":
        popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    p = subprocess.Popen(**popen_kw)
    _PROCS.append(p)
    print(f"Started [{name}] pid={p.pid}")
    time.sleep(0.35)
    if p.poll() is not None:
        print(f"ERROR: [{name}] exited immediately with code {p.returncode}", file=sys.stderr)
        terminate_all()
        sys.exit(1)
    return p


def wait_http_ready(
    base_url: str,
    name: str,
    proc: subprocess.Popen,
    *,
    path: str = "/health",
    seconds: float = 15.0,
    interval: float = 0.25,
) -> None:
    url = f"{base_url.rstrip('/')}{path}"
    deadline = time.monotonic() + seconds

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            sys.stderr.write(
                f"ERROR: [{name}] exited early (code {proc.returncode}). "
                "If you see WinError 10048, another process is using that port "
                "(often a leftover mock-api on 8001). Close it or find the PID:\n"
                "  PowerShell: Get-NetTCPConnection -LocalPort 8001 | Select-Object OwningProcess\n"
                "  cmd: netstat -ano | findstr :8001\n",
            )
            terminate_all()
            sys.exit(1)

        try:
            with urllib.request.urlopen(url, timeout=1.5) as r:  # noqa: S310
                if getattr(r, "status", 200) == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(interval)

    sys.stderr.write(f"ERROR: [{name}] did not become ready at {url} within {seconds}s.\n")
    terminate_all()
    sys.exit(1)


def ollama_reachable(http_url: str) -> bool:
    tags_url = f"{http_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=3) as r:  # noqa: S310
            return getattr(r, "status", 200) == 200
    except (urllib.error.URLError, OSError):
        return False


def warn_ollama(http_url: str) -> None:
    if ollama_reachable(http_url):
        return
    sys.stderr.write(
        "[warn] Ollama is not reachable. Install from https://ollama.com and run:\n"
        "       ollama pull llama3.2\n"
        "       ollama pull nomic-embed-text\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the finance-assistant stack locally.")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip RAG ingest (reuse existing persisted Chroma).",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Mock API + Agent only (no Streamlit).",
    )
    parser.add_argument(
        "--free-ports",
        action="store_true",
        help="Windows: forcibly stop any process listening on ports 8000, 8001, 8501 (stale orphans).",
    )
    args = parser.parse_args()

    if args.free_ports:
        if sys.platform == "win32":
            print("Stopping anything listening on 8000 / 8001 / 8501 …")
            free_local_stack_ports_windows()
            time.sleep(0.5)
        else:
            sys.stderr.write("--free-ports is only automated on Windows; on other OSes free ports manually.\n")

    ensure_dotenv()
    load_dotenv(ROOT / ".env")
    ensure_stack_ports_free()
    env_base = base_env()
    python = sys.executable

    ollama_url = env_base.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    warn_ollama(ollama_url)

    if not args.skip_ingest and not ollama_reachable(ollama_url):
        sys.stderr.write(
            "\nRAG ingest needs Ollama for embeddings. Start Ollama (see https://ollama.com) and run:\n"
            "  ollama pull llama3.2\n"
            "  ollama pull nomic-embed-text\n"
            "\nOr skip ingest if Chroma is already populated:  python run_local.py --skip-ingest\n",
        )
        sys.exit(1)

    subprocess.run(
        [python, "database/seed.py"],
        cwd=str(MOCK_API),
        env=with_pythonpath(env_base, MOCK_API),
        check=True,
    )
    mock_proc = spawn(
        name="mock-api",
        args=[python, "run_server.py"],
        cwd=MOCK_API,
        env=with_pythonpath(env_base, MOCK_API),
    )
    wait_http_ready("http://127.0.0.1:8001", "mock-api", mock_proc)

    if not args.skip_ingest:
        subprocess.run(
            [python, "rag/ingest.py"],
            cwd=str(AGENT_DIR),
            env=with_pythonpath(env_base, AGENT_DIR),
            check=True,
        )

    agent_proc = spawn(
        name="agent",
        args=[python, "run_server.py"],
        cwd=AGENT_DIR,
        env=with_pythonpath(env_base, AGENT_DIR),
    )
    wait_http_ready("http://127.0.0.1:8000", "agent", agent_proc)

    if not args.no_ui:
        fe = spawn(
            name="frontend",
            args=[
                python,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.port",
                "8501",
                "--server.address",
                "127.0.0.1",
                "--server.headless",
                "true",
            ],
            cwd=FRONTEND,
            env={
                **with_pythonpath(env_base, FRONTEND),
                "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
            },
        )
        wait_http_ready("http://127.0.0.1:8501", "frontend", fe, path="/", seconds=45.0)

    print()
    print("Frontend   http://127.0.0.1:8501")
    print("Agent API  http://127.0.0.1:8000/docs")
    print("Mock bank  http://127.0.0.1:8001/docs")
    print()
    print("Press Ctrl+C to stop.")

    def _stop(signum=None, frame=None) -> None:  # noqa: ANN001, ARG001
        print("\nStopping…")
        terminate_all()
        sys.exit(0)

    if sys.platform != "win32":
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

    try:
        while True:
            for p in list(_PROCS):
                rc = p.poll()
                if rc is not None:
                    print(f"Child pid {p.pid} exited ({rc}); shutting others down.")
                    terminate_all()
                    sys.exit(rc or 1)
            time.sleep(0.35)
    except KeyboardInterrupt:
        _stop()


if __name__ == "__main__":
    main()
