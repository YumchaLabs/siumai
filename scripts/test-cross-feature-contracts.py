#!/usr/bin/env python3
"""Run no-network cross-feature facade contract profiles."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROFILES: dict[str, tuple[bool, str, tuple[str, ...]]] = {
    "no-default": (False, "", ("facade_contract",)),
    "default": (True, "", ("facade_contract",)),
    "one-provider": (False, "openai", ("facade_contract",)),
    "registry-openai": (False, "registry,openai", ("facade_contract",)),
    "compatible-stack": (
        False,
        "registry,openai,openai-compatible,groq,xai,deepseek",
        ("facade_contract",),
    ),
    "multi-provider": (
        False,
        "registry,openai,openai-compatible,google,cohere,deepgram,elevenlabs",
        ("facade_contract",),
    ),
    "all-providers": (False, "registry,all-providers", ("facade_contract",)),
    "openai-realtime": (False, "openai-realtime,registry", ("facade_contract",)),
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


def build_command(profile: str, use_nextest: bool) -> list[str]:
    use_default_features, features, tests = PROFILES[profile]
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
            "--no-fail-fast",
        ]
    else:
        command = [
            "cargo",
            "test",
            "-j",
            "1",
            "-p",
            "siumai",
            "--no-fail-fast",
        ]
    if not use_default_features:
        command.append("--no-default-features")
    if features:
        command.extend(("--features", features))
    command.extend(test_args)
    return command


def run_profile(repo_root: Path, profile: str, use_nextest: bool) -> int:
    use_default_features, features, _tests = PROFILES[profile]
    command = build_command(profile, use_nextest)

    print(
        "[test-cross-feature-contracts] "
        f"profile={profile} default_features={use_default_features} "
        f"features={features or '-'}",
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
