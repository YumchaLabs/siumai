#!/usr/bin/env python3
"""Run Siumai's standard local model-catalog drift audit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def build_command(repo_root: Path, arguments: list[str]) -> list[str]:
    audit_script = (
        repo_root
        / ".agents"
        / "skills"
        / "siumai-ai-sdk-maintenance"
        / "scripts"
        / "audit_model_catalogs.py"
    )
    return [
        sys.executable,
        str(audit_script),
        "--include-green",
        "--show-skipped",
        "--defer",
        "deepinfra",
        *arguments,
    ]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        build_command(repo_root, sys.argv[1:]),
        cwd=repo_root,
        check=False,
    ).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as error:
        print(f"[audit-model-catalogs] missing executable: {error.filename}", file=sys.stderr)
        raise SystemExit(127) from error
