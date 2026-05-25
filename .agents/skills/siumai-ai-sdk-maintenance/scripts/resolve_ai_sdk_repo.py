#!/usr/bin/env python3
"""Resolve a local Vercel AI SDK reference checkout.

The script prints JSON so agents can consume it without parsing prose.
It does not modify the filesystem.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


def is_ai_sdk_repo(path: Path) -> bool:
    packages = path / "packages"
    return (
        path.is_dir()
        and (packages / "ai").is_dir()
        and any(
            child.is_dir()
            and child.name.startswith(("openai", "anthropic", "google", "provider"))
            for child in packages.iterdir()
        )
    )


def parents_from(start: Path) -> Iterable[Path]:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    yield current
    yield from current.parents


def candidate_paths(start: Path) -> list[Path]:
    candidates: list[Path] = []

    for env_name in ("AI_SDK_REPO", "VERCEL_AI_REPO"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())

    names = ("ai", "vercel-ai", "ai-sdk")
    relative_candidates = (
        Path("repo-ref") / "ai",
        Path("repo-ref") / "vercel-ai",
        Path("repo-ref") / "ai-sdk",
    )

    for parent in parents_from(start):
        for rel in relative_candidates:
            candidates.append(parent / rel)
        if parent.parent != parent:
            for rel in relative_candidates:
                candidates.append(parent.parent / rel)
        for name in names:
            candidates.append(parent / name)
            if parent.parent != parent:
                candidates.append(parent.parent / name)

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate.absolute())
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        default=os.getcwd(),
        help="Start directory for nearby repo discovery. Defaults to cwd.",
    )
    args = parser.parse_args()

    checked: list[str] = []
    for candidate in candidate_paths(Path(args.start)):
        checked.append(str(candidate))
        if is_ai_sdk_repo(candidate):
            print(
                json.dumps(
                    {
                        "found": True,
                        "path": str(candidate.resolve()),
                        "source": "env-or-nearby",
                    },
                    indent=2,
                )
            )
            return 0

    print(
        json.dumps(
            {
                "found": False,
                "path": None,
                "message": "Set AI_SDK_REPO to a local Vercel AI SDK checkout or place it near the Siumai repo as repo-ref/ai.",
                "checked": checked[:50],
            },
            indent=2,
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
