#!/usr/bin/env python3
"""Run the maintained local Siumai test suites serially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FAST_PACKAGES = (
    "siumai-core",
    "siumai-runtime",
    "siumai-transport",
    "siumai-registry",
    "siumai",
)
FLAGSHIP_PACKAGES = (
    "siumai-protocol-openai",
    "siumai-provider-openai",
    "siumai-openai-compatible",
    "siumai-protocol-anthropic",
    "siumai-anthropic-compatible",
    "siumai-provider-anthropic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run current, no-network Siumai test suites without parallel Cargo jobs."
    )
    parser.add_argument("suite", choices=("fast", "flagship", "full"))
    parser.add_argument(
        "--runner",
        choices=("auto", "nextest", "cargo-test"),
        default="auto",
        help="Rust test runner (default: auto-detect nextest)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without executing them",
    )
    return parser.parse_args()


def has_nextest() -> bool:
    result = subprocess.run(
        ["cargo", "nextest", "--version"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def resolve_runner(requested: str) -> str:
    if requested == "auto":
        return "nextest" if has_nextest() else "cargo-test"
    return requested


def package_test_command(
    runner: str,
    packages: tuple[str, ...],
    *,
    single_threaded_tests: bool = False,
) -> list[str]:
    package_args = [argument for package in packages for argument in ("-p", package)]
    if runner == "nextest":
        return [
            "cargo",
            "nextest",
            "run",
            "-j",
            "1",
            "--no-fail-fast",
            *package_args,
        ]
    command = [
        "cargo",
        "test",
        "-j",
        "1",
        "--no-fail-fast",
        *package_args,
    ]
    if single_threaded_tests:
        command.extend(("--", "--test-threads=1"))
    return command


def workspace_test_command(runner: str) -> list[str]:
    if runner == "nextest":
        return [
            "cargo",
            "nextest",
            "run",
            "--profile",
            "ci",
            "-j",
            "1",
            "--workspace",
            "--all-features",
            "--no-fail-fast",
        ]
    return [
        "cargo",
        "test",
        "-j",
        "1",
        "--workspace",
        "--all-features",
        "--no-fail-fast",
        "--",
        "--test-threads=1",
    ]


def commands_for(args: argparse.Namespace, runner: str) -> list[list[str]]:
    if args.suite == "fast":
        return [package_test_command(runner, FAST_PACKAGES)]

    if args.suite == "flagship":
        return [
            package_test_command(
                runner,
                FLAGSHIP_PACKAGES,
                single_threaded_tests=True,
            )
        ]

    return [workspace_test_command(runner)]


def run_commands(commands: list[list[str]], dry_run: bool) -> int:
    for command in commands:
        print(f"[test-workspace] {subprocess.list2cmdline(command)}", flush=True)
        if dry_run:
            continue
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            return result.returncode
    print("[test-workspace] OK")
    return 0


def main() -> int:
    args = parse_args()
    runner = resolve_runner(args.runner)
    print(f"[test-workspace] suite={args.suite} runner={runner}", flush=True)
    return run_commands(commands_for(args, runner), args.dry_run)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as error:
        print(f"[test-workspace] missing executable: {error.filename}", file=sys.stderr)
        raise SystemExit(127) from error
