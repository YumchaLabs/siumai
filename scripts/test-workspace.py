#!/usr/bin/env python3
"""Run the maintained local Siumai test suites serially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

SMOKE_PROFILES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "openai": (
        ("openai-native", "openai-compat"),
        ("registry-openai",),
    ),
    "openai-compatible": (
        ("openai-native", "openai-compat", "groq", "xai", "deepseek"),
        ("compatible-stack",),
    ),
    "all-providers": (
        ("all",),
        ("all-providers",),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run current, no-network Siumai test suites without parallel Cargo jobs."
    )
    parser.add_argument("suite", choices=("fast", "smoke", "full"))
    parser.add_argument(
        "--profile",
        choices=tuple(SMOKE_PROFILES),
        default="openai",
        help="provider profile used by the smoke suite (default: openai)",
    )
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


def package_test_command(runner: str, packages: tuple[str, ...]) -> list[str]:
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
    return ["cargo", "test", "-j", "1", "--no-fail-fast", *package_args]


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
            "--test-threads",
            "1",
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


def common_checks() -> list[list[str]]:
    return [
        [sys.executable, "-B", "scripts/check_workspace_boundaries.py"],
        [sys.executable, "-B", "scripts/check_fixture_inventory.py"],
    ]


def commands_for(args: argparse.Namespace, runner: str) -> list[list[str]]:
    if args.suite == "fast":
        return [
            package_test_command(
                runner,
                (
                    "siumai-core",
                    "siumai-runtime",
                    "siumai-transport",
                    "siumai-registry",
                    "siumai",
                ),
            ),
        ]

    if args.suite == "smoke":
        provider_profiles, facade_profiles = SMOKE_PROFILES[args.profile]
        commands = [
            *common_checks(),
            package_test_command(
                runner,
                (
                    "siumai-core",
                    "siumai-runtime",
                    "siumai-transport",
                    "siumai-registry",
                ),
            ),
        ]
        commands.extend(
            [sys.executable, "-B", "scripts/test-provider-contracts.py", profile]
            for profile in provider_profiles
        )
        commands.extend(
            [sys.executable, "-B", "scripts/test-cross-feature-contracts.py", profile]
            for profile in facade_profiles
        )
        return commands

    return [
        [
            sys.executable,
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            "scripts/tests",
            "-p",
            "test_*.py",
        ],
        *common_checks(),
        [sys.executable, "-B", "scripts/test-cross-feature-contracts.py"],
        workspace_test_command(runner),
    ]


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
