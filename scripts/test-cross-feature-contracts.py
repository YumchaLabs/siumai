#!/usr/bin/env python3
"""Run no-network cross-feature facade contract profiles."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROFILES: dict[str, tuple[str, tuple[str, ...]]] = {
    "openai-realtime": ("openai-realtime,registry", ("facade_contract",)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run no-network contract tests for facade feature combinations."
    )
    parser.add_argument(
        "profile",
        nargs="?",
        default="all",
        choices=("all", *PROFILES),
        help="contract profile to run (default: all)",
    )
    return parser.parse_args()


def has_nextest(repo_root: Path) -> bool:
    result = subprocess.run(
        ["cargo", "nextest", "--version"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def run_profile(repo_root: Path, profile: str, use_nextest: bool) -> int:
    features, tests = PROFILES[profile]
    test_args = [argument for test in tests for argument in ("--test", test)]
    if use_nextest:
        command = [
            "cargo",
            "nextest",
            "run",
            "-j",
            "1",
            "-p",
            "siumai",
            "--no-default-features",
            "--features",
            features,
            "--no-fail-fast",
            *test_args,
        ]
    else:
        command = [
            "cargo",
            "test",
            "-j",
            "1",
            "-p",
            "siumai",
            "--no-default-features",
            "--features",
            features,
            "--no-fail-fast",
            *test_args,
        ]

    print(
        f"[test-cross-feature-contracts] profile={profile} features={features}",
        flush=True,
    )
    return subprocess.run(command, cwd=repo_root, check=False).returncode


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    use_nextest = has_nextest(repo_root)
    profiles = PROFILES if args.profile == "all" else (args.profile,)

    for profile in profiles:
        return_code = run_profile(repo_root, profile, use_nextest)
        if return_code != 0:
            return return_code

    print("[test-cross-feature-contracts] OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as error:
        print(f"[test-cross-feature-contracts] missing executable: {error.filename}", file=sys.stderr)
        raise SystemExit(127) from error
