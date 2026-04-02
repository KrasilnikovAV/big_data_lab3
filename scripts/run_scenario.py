from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def _resolve_json_path(payload: dict, path: str) -> object:
    current: object = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Path '{path}' is missing in response body.")
        current = current[part]
    return current


def _resolve_scenario_path(scenario_path: Path) -> Path:
    if scenario_path.exists():
        return scenario_path

    checked: list[Path] = [scenario_path]

    if not scenario_path.is_absolute():
        fallback_paths = [
            Path("/app") / scenario_path,
            Path(__file__).resolve().parents[1] / scenario_path,
        ]
        for path in fallback_paths:
            checked.append(path)
            if path.exists():
                return path

    checked_paths = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"Scenario file was not found. Checked: {checked_paths}. Current working dir: {Path.cwd()}"
    )


def _perform_request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    payload: dict | None,
    timeout: int,
    retries: int,
    delay_seconds: float,
) -> requests.Response:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            if method == "GET":
                return session.get(url, timeout=timeout)
            if method == "POST":
                return session.post(url, json=payload or {}, timeout=timeout)
            raise ValueError(f"Unsupported method '{method}' in scenario.")
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(delay_seconds)

    assert last_error is not None
    raise last_error


def run_scenario(
    scenario_path: Path,
    base_url_override: str | None = None,
    retries: int = 5,
    delay_seconds: float = 2.0,
) -> int:
    resolved_scenario_path = _resolve_scenario_path(scenario_path)
    scenario = json.loads(resolved_scenario_path.read_text(encoding="utf-8"))
    base_url = (base_url_override or scenario["base_url"]).rstrip("/")
    checks = scenario.get("checks", [])
    session = requests.Session()
    session.trust_env = False

    for check in checks:
        method = check.get("method", "GET").upper()
        path = check.get("path", "")
        url = f"{base_url}{path}"
        response = _perform_request_with_retry(
            session=session,
            method=method,
            url=url,
            payload=check.get("json"),
            timeout=15,
            retries=retries,
            delay_seconds=delay_seconds,
        )

        expected_status = int(check["expected_status"])
        if response.status_code != expected_status:
            raise AssertionError(
                f"{check['name']}: status {response.status_code} != {expected_status}"
            )

        contains = check.get("contains")
        if contains and contains not in response.text:
            raise AssertionError(f"{check['name']}: '{contains}' was not found in response.")

        json_path = check.get("expected_json_path")
        if json_path:
            payload = response.json()
            value = _resolve_json_path(payload, json_path)
            expected_length = check.get("expected_length")
            if expected_length is not None and len(value) != int(expected_length):
                raise AssertionError(
                    f"{check['name']}: length {len(value)} != {expected_length}"
                )

        print(f"[OK] {check['name']}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run functional scenario against model API.")
    parser.add_argument(
        "--scenario",
        default="scenario.json",
        help="Path to scenario definition in JSON format.",
    )
    parser.add_argument("--base-url", default=None, help="Override base URL from scenario.json.")
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="How many times to retry transient connection failures for each check.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Delay between retries in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_scenario(
        Path(args.scenario),
        args.base_url,
        retries=max(1, args.retries),
        delay_seconds=max(0.1, args.retry_delay),
    )


if __name__ == "__main__":
    raise SystemExit(main())
