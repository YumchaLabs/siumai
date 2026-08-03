#!/usr/bin/env python3
"""Validate Siumai package boundaries from Cargo's structured metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = REPO_ROOT / "config" / "architecture" / "dependency-policy.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def cargo_metadata() -> dict[str, Any]:
    process = subprocess.run(
        [
            "cargo",
            "metadata",
            "--locked",
            "--no-deps",
            "--format-version",
            "1",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"cargo metadata failed: {detail}")
    value = json.loads(process.stdout)
    if not isinstance(value, dict):
        raise ValueError("cargo metadata must return a JSON object")
    return value


def workspace_packages(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    member_ids = set(metadata.get("workspace_members", []))
    packages = {
        package["name"]: package
        for package in metadata.get("packages", [])
        if package.get("id") in member_ids
    }
    if not packages:
        raise ValueError("cargo metadata did not contain workspace packages")
    return packages


def workspace_dependencies(
    package: dict[str, Any], workspace_names: set[str]
) -> set[str]:
    return {
        dependency["name"]
        for dependency in package.get("dependencies", [])
        if dependency.get("name") in workspace_names
    }


def transition_allows(
    source: str, dependency: str, transitions: list[dict[str, Any]]
) -> bool:
    for transition in transitions:
        if transition.get("dependency") != dependency:
            continue
        sources = transition.get("sources", [])
        if "*" in sources or source in sources:
            return True
    return False


def validate(
    metadata: dict[str, Any], policy: dict[str, Any], *, target: bool
) -> list[str]:
    errors: list[str] = []
    packages = workspace_packages(metadata)
    workspace_names = set(packages)
    expected_msrv = policy.get("msrv")

    for name, package in sorted(packages.items()):
        if package.get("rust_version") != expected_msrv:
            errors.append(
                f"{name}: rust_version={package.get('rust_version')!r}; "
                f"expected {expected_msrv!r}"
            )

    rule_key = (
        "target_allowed_workspace_dependencies"
        if target
        else "current_allowed_workspace_dependencies"
    )
    for name, rule in policy.get("package_rules", {}).items():
        package = packages.get(name)
        if package is None:
            errors.append(f"policy references missing workspace package {name!r}")
            continue
        actual = workspace_dependencies(package, workspace_names)
        allowed = set(rule.get(rule_key, []))
        unexpected = sorted(actual - allowed)
        if unexpected:
            errors.append(
                f"{name}: unexpected workspace dependencies for "
                f"{'target' if target else 'current'} policy: "
                + ", ".join(unexpected)
            )

    provider_transitions = policy.get("provider_dependency_transitions", [])
    for name, package in sorted(packages.items()):
        if not name.startswith("siumai-provider-"):
            continue
        provider_dependencies = sorted(
            dependency
            for dependency in workspace_dependencies(package, workspace_names)
            if dependency.startswith("siumai-provider-")
            and (
                target
                or not transition_allows(name, dependency, provider_transitions)
            )
        )
        if provider_dependencies:
            errors.append(
                f"{name}: provider-to-provider dependencies are forbidden: "
                + ", ".join(provider_dependencies)
            )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
        help="dependency policy JSON",
    )
    parser.add_argument(
        "--target",
        action="store_true",
        help="enforce the final package graph instead of the migration ratchet",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        errors = validate(cargo_metadata(), load_json(args.policy), target=args.target)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"[workspace-boundaries] ERROR: {error}", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(f"[workspace-boundaries] ERROR: {error}", file=sys.stderr)
        return 1

    mode = "target" if args.target else "migration"
    print(f"[workspace-boundaries] OK ({mode} policy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
