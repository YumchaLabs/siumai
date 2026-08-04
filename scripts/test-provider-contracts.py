#!/usr/bin/env python3
"""Run provider-package, no-network contract profiles."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROFILES: dict[str, tuple[str, str]] = {
    "openai-native": ("siumai-provider-openai", "openai"),
    "openai-compat": ("siumai-provider-openai-compatible", "openai-standard"),
    "azure": ("siumai-provider-azure", "azure"),
    "anthropic": ("siumai-provider-anthropic", "anthropic"),
    "google": ("siumai-provider-gemini", "google"),
    "google-vertex": ("siumai-provider-google-vertex", "google-vertex,gcp"),
    "ollama": ("siumai-provider-ollama", "ollama"),
    "xai": ("siumai-provider-xai", "xai"),
    "groq": ("siumai-provider-groq", "groq"),
    "minimaxi": ("siumai-provider-minimaxi", "minimaxi"),
    "deepseek": ("siumai-provider-deepseek", "deepseek"),
    "cohere": ("siumai-provider-cohere", "cohere"),
    "togetherai": ("siumai-provider-togetherai", "togetherai"),
    "bedrock": ("siumai-provider-amazon-bedrock", "bedrock"),
    "gateway": ("siumai-provider-gateway", "gateway"),
    "deepgram": ("siumai-provider-deepgram", "deepgram"),
    "elevenlabs": ("siumai-provider-elevenlabs", "elevenlabs"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run provider-package no-network contract tests."
    )
    parser.add_argument(
        "profile",
        nargs="?",
        default="all",
        choices=("all", *PROFILES),
        help="provider contract profile to run (default: all)",
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
    package, features = PROFILES[profile]
    if use_nextest:
        command = [
            "cargo",
            "nextest",
            "run",
            "-j",
            "1",
            "-p",
            package,
            "--no-default-features",
            "--features",
            features,
            "--no-fail-fast",
        ]
    else:
        command = [
            "cargo",
            "test",
            "-j",
            "1",
            "-p",
            package,
            "--no-default-features",
            "--features",
            features,
            "--no-fail-fast",
        ]

    print(
        f"[test-provider-contracts] profile={profile} package={package} features={features}",
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

    print("[test-provider-contracts] OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as error:
        print(
            f"[test-provider-contracts] missing executable: {error.filename}",
            file=sys.stderr,
        )
        raise SystemExit(127) from error
