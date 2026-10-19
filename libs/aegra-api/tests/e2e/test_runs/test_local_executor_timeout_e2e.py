import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
from langgraph_sdk import get_client

REPO_ROOT = Path(__file__).resolve().parents[5]
API_ROOT = REPO_ROOT / "libs" / "aegra-api"
API_SRC = API_ROOT / "src"


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _default_postgres_value(key: str) -> str:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        ci_defaults = {
            "POSTGRES_USER": "user",
            "POSTGRES_PASSWORD": "password",
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "aegra",
        }
        return ci_defaults[key]

    local_defaults = {
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "postgres",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "aegra",
    }
    return local_defaults[key]


def _can_connect_postgres(config: dict[str, str]) -> bool:
    try:
        with psycopg.connect(
            host=config["POSTGRES_HOST"],
            port=config["POSTGRES_PORT"],
            user=config["POSTGRES_USER"],
            password=config["POSTGRES_PASSWORD"],
            dbname=config["POSTGRES_DB"],
            connect_timeout=2,
        ):
            return True
    except psycopg.Error:
        return False


def _resolve_postgres_env() -> dict[str, str]:
    env = os.environ.copy()
    env_file_values = _read_env_file(REPO_ROOT / ".env")

    host = env.get("POSTGRES_HOST") or env_file_values.get("POSTGRES_HOST") or _default_postgres_value("POSTGRES_HOST")
    port = env.get("POSTGRES_PORT") or env_file_values.get("POSTGRES_PORT") or _default_postgres_value("POSTGRES_PORT")
    dbname = env.get("POSTGRES_DB") or env_file_values.get("POSTGRES_DB") or _default_postgres_value("POSTGRES_DB")

    user_candidates: list[tuple[str, str]] = []
    primary_user = env.get("POSTGRES_USER") or env_file_values.get("POSTGRES_USER")
    primary_password = env.get("POSTGRES_PASSWORD") or env_file_values.get("POSTGRES_PASSWORD")
    if primary_user and primary_password:
        user_candidates.append((primary_user, primary_password))
    user_candidates.extend(
        [
            ("user", "password"),
            ("postgres", "postgres"),
        ]
    )

    seen: set[tuple[str, str]] = set()
    for user, password in user_candidates:
        if (user, password) in seen:
            continue
        seen.add((user, password))

        config = {
            "POSTGRES_USER": user,
            "POSTGRES_PASSWORD": password,
            "POSTGRES_HOST": host,
            "POSTGRES_PORT": port,
            "POSTGRES_DB": dbname,
        }
        if _can_connect_postgres(config):
            return config

    raise AssertionError("Could not resolve working Postgres credentials for local timeout E2E server")


def _timeout_server_env(port: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(_resolve_postgres_env())

    env["AEGRA_CONFIG"] = str(REPO_ROOT / "aegra.json")
    env["AUTH_TYPE"] = "noop"
    env["HOST"] = "127.0.0.1"
    env["PORT"] = str(port)
    env["SERVER_URL"] = f"http://127.0.0.1:{port}"
    env["REDIS_BROKER_ENABLED"] = "false"
    env["BG_JOB_TIMEOUT_SECS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((str(API_SRC), str(REPO_ROOT / "examples")))
    return env


def _wait_for_server(base_url: str, *, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: str | None = None

    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=1.0)
            if response.status_code == 200:
                return
            last_error = f"health returned {response.status_code}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)

    raise AssertionError(f"Timed out waiting for timeout test server: {last_error}")


def _stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


async def _poll_terminal_run(
    *,
    client: Any,
    thread_id: str,
    run_id: str,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        run = await client.runs.get(thread_id, run_id)
        if run["status"] in {"success", "error", "interrupted"}:
            return run
        await asyncio.sleep(0.2)

    raise AssertionError(f"Run {run_id} did not reach terminal state within {timeout_seconds}s")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_local_executor_timeout_enforced_e2e() -> None:
    port = 2127
    base_url = f"http://127.0.0.1:{port}"
    try:
        env = _timeout_server_env(port)
    except AssertionError as exc:
        pytest.skip(str(exc))

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "aegra_api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=API_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    try:
        _wait_for_server(base_url)
        client = get_client(url=base_url)

        assistant = await client.assistants.create(
            graph_id="stress_test",
            config={"tags": ["local-timeout-e2e"]},
            if_exists="do_nothing",
        )
        thread = await client.threads.create()
        run = await client.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=assistant["assistant_id"],
            input={"messages": [{"role": "user", "content": '{"delay": 2.0, "steps": 3}'}]},
        )

        final_run = await _poll_terminal_run(
            client=client,
            thread_id=thread["thread_id"],
            run_id=run["run_id"],
        )

        assert final_run["status"] == "error"
        assert "Job exceeded maximum execution time" in str(final_run.get("error_message", ""))
    finally:
        _stop_server(process)
