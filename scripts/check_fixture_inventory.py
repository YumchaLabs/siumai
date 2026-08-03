#!/usr/bin/env python3
"""Ensure every retained wire fixture has exactly one inventory owner."""

from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "config" / "fixtures" / "manifest.json"
REQUIRED_FIELDS = {
    "path",
    "owner",
    "protocol",
    "behaviors",
    "source_kind",
    "source_date",
    "disposition",
}


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("fixture manifest must be a JSON object")
    return manifest


def validate(root: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    sets = manifest.get("sets", [])
    if not isinstance(sets, list) or not sets:
        return ["fixture manifest must define at least one set"]

    for index, fixture_set in enumerate(sets):
        if not isinstance(fixture_set, dict):
            errors.append(f"set {index} must be an object")
            continue
        missing = sorted(REQUIRED_FIELDS - set(fixture_set))
        if missing:
            errors.append(f"set {index} is missing: {', '.join(missing)}")
        if not fixture_set.get("behaviors"):
            errors.append(f"set {index} must classify at least one behavior")

    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        matches = [
            fixture_set
            for fixture_set in sets
            if isinstance(fixture_set, dict)
            and fnmatch.fnmatchcase(relative, str(fixture_set.get("path", "")))
        ]
        if not matches:
            errors.append(f"unclassified fixture: {relative}")
        elif len(matches) > 1:
            patterns = ", ".join(str(item["path"]) for item in matches)
            errors.append(f"fixture matches multiple sets: {relative}: {patterns}")

    for fixture_set in sets:
        if not isinstance(fixture_set, dict) or "path" not in fixture_set:
            continue
        pattern = str(fixture_set["path"])
        if not any(
            fnmatch.fnmatchcase(path.relative_to(root).as_posix(), pattern)
            for path in root.rglob("*")
            if path.is_file()
        ):
            errors.append(f"fixture set matches no files: {pattern}")

    return errors


def main() -> int:
    try:
        manifest = load_manifest()
        fixture_root = REPO_ROOT / str(manifest["fixture_root"])
        errors = validate(fixture_root, manifest)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[fixture-inventory] ERROR: {error}", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(f"[fixture-inventory] ERROR: {error}", file=sys.stderr)
        return 1

    fixture_count = sum(1 for path in fixture_root.rglob("*") if path.is_file())
    print(
        f"[fixture-inventory] OK ({fixture_count} files, "
        f"{len(manifest['sets'])} inherited sets)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
