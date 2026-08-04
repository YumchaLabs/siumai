#!/usr/bin/env python3
"""Run release-plz with bounded crates.io rate-limit retries."""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RETRY_AFTER_PATTERN = re.compile(
    r"after (?P<date>[A-Za-z]{3}, [0-9]{1,2} [A-Za-z]{3} "
    r"[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} GMT)"
)
RATE_LIMIT_PATTERNS = (
    "status 429 Too Many Requests",
    "published too many new crates",
)


def positive_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def is_crates_io_rate_limit(output: str) -> bool:
    return any(pattern in output for pattern in RATE_LIMIT_PATTERNS)


def extract_retry_after(output: str) -> datetime | None:
    match = RETRY_AFTER_PATTERN.search(output)
    if match is None:
        return None
    try:
        retry_after = parsedate_to_datetime(match.group("date"))
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_after.tzinfo is None:
        retry_after = retry_after.replace(tzinfo=timezone.utc)
    return retry_after.astimezone(timezone.utc)


def retry_delay_seconds(output: str, now: datetime, minimum: int) -> int:
    retry_after = extract_retry_after(output)
    if retry_after is None:
        return minimum
    delay = math.ceil((retry_after - now.astimezone(timezone.utc)).total_seconds()) + 10
    return max(minimum, delay)


def run_release(github_token: str) -> tuple[int, str]:
    command = ["release-plz", "release", "--git-token", github_token]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    lines: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return process.wait(), "".join(lines)


def main() -> int:
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        print("Missing required env var: GITHUB_TOKEN", file=sys.stderr)
        return 2

    try:
        max_attempts = positive_env_int("RELEASE_PLZ_MAX_ATTEMPTS", 10)
        minimum_sleep = positive_env_int("RELEASE_PLZ_MIN_SLEEP_SECONDS", 60)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    for attempt in range(1, max_attempts + 1):
        print(f"::group::release-plz release attempt {attempt}/{max_attempts}")
        status, output = run_release(github_token)
        print("::endgroup::")

        if status == 0:
            print("release-plz release succeeded.")
            return 0
        if not is_crates_io_rate_limit(output):
            print("release-plz release failed (non-429).", file=sys.stderr)
            return status
        if attempt == max_attempts:
            break

        delay = retry_delay_seconds(
            output,
            datetime.now(timezone.utc),
            minimum_sleep,
        )
        print(f"Hit crates.io rate limit (429). Sleeping {delay}s before retry.")
        time.sleep(delay)

    print(
        f"release-plz release kept hitting crates.io 429 after {max_attempts} attempts.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as error:
        print(f"Missing executable: {error.filename}", file=sys.stderr)
        raise SystemExit(127) from error
