#!/usr/bin/env python3
"""Reject workspace dependency directions that violate Siumai's architecture."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent

FOUNDATION_DEPENDENCIES = {
    "siumai-core": frozenset(),
    "siumai-transport": frozenset({"siumai-core"}),
    "siumai-registry": frozenset({"siumai-core"}),
    "siumai-runtime": frozenset({"siumai-core"}),
}
HOST_LAYER_PACKAGES = frozenset(
    {"siumai-registry", "siumai-runtime", "siumai-mcp", "siumai-server"}
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


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


def is_branded_provider(name: str) -> bool:
    return name.startswith("siumai-provider-")


def is_provider_neutral(name: str) -> bool:
    return (
        name in FOUNDATION_DEPENDENCIES
        or name in HOST_LAYER_PACKAGES
        or name.startswith("siumai-protocol-")
        or name.endswith("-compatible")
    )


def is_wire_or_compatibility_layer(name: str) -> bool:
    return name.startswith("siumai-protocol-") or name.endswith("-compatible")


def validate(metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    packages = workspace_packages(metadata)
    workspace_names = set(packages)
    dependencies = {
        name: workspace_dependencies(package, workspace_names)
        for name, package in packages.items()
    }

    for name, allowed in FOUNDATION_DEPENDENCIES.items():
        actual = dependencies.get(name)
        if actual is None:
            continue
        unexpected = sorted(actual - allowed)
        if unexpected:
            errors.append(
                f"{name}: foundation package cannot depend on: "
                + ", ".join(unexpected)
            )

    for name, actual in sorted(dependencies.items()):
        if name in FOUNDATION_DEPENDENCIES:
            continue

        branded_dependencies = sorted(
            dependency for dependency in actual if is_branded_provider(dependency)
        )

        if name != "siumai" and "siumai" in actual:
            errors.append(f"{name}: facade back-edge to siumai is forbidden")

        if is_provider_neutral(name) and branded_dependencies:
            errors.append(
                f"{name}: provider-neutral package cannot depend on branded providers: "
                + ", ".join(branded_dependencies)
            )

        if is_wire_or_compatibility_layer(name):
            host_dependencies = sorted(actual & HOST_LAYER_PACKAGES)
            if host_dependencies:
                errors.append(
                    f"{name}: wire or compatibility package cannot depend on "
                    "host-layer packages: " + ", ".join(host_dependencies)
                )

        if not is_branded_provider(name):
            continue

        if branded_dependencies:
            errors.append(
                f"{name}: provider-to-provider dependencies are forbidden: "
                + ", ".join(branded_dependencies)
            )

        host_dependencies = sorted(actual & HOST_LAYER_PACKAGES)
        if host_dependencies:
            errors.append(
                f"{name}: branded provider cannot depend on host-layer packages: "
                + ", ".join(host_dependencies)
            )

    return errors


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    try:
        errors = validate(cargo_metadata())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"[workspace-boundaries] ERROR: {error}", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(f"[workspace-boundaries] ERROR: {error}", file=sys.stderr)
        return 1

    print("[workspace-boundaries] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
