#!/usr/bin/env python3
"""Inspect Cargo's bounded package file list without shell pipelines.

Cargo remains authoritative for package membership and publish semantics. This
script only rejects paths that must not enter a release artifact.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import BinaryIO


MAX_ENTRIES = 100_000
MAX_PATH_BYTES = 4_096
MAX_CARGO_STDOUT_BYTES = 64 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_COMPONENTS = frozenset(
    {
        "target",
        "repo-ref",
        ".git",
        ".hg",
        ".svn",
        ".codex",
        ".codex-helper",
        ".idea",
        ".vscode",
        ".vs",
        ".fleet",
        ".zed",
    }
)

FORBIDDEN_BASENAMES = frozenset(
    {
        ".ds_store",
        ".env",
        ".gitconfig",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "_netrc",
        "credentials",
        "credentials.json",
        "credentials.toml",
        "credentials.yaml",
        "credentials.yml",
        "desktop.ini",
        "id_dsa",
        "id_ed25519",
        "id_ecdsa",
        "id_rsa",
        "thumbs.db",
    }
)

PRIVATE_CONFIG_RE = re.compile(
    r"(?:config|settings)\.local(?:\.(?:json|toml|ya?ml|ini))?\Z"
)
PRIVATE_DATA_RE = re.compile(
    r"(?:"
    r"secrets?\.(?:json|toml|ya?ml|ini|pem|key)"
    r"|service[-_]account(?:[-_][^.]+)?\.(?:json|toml|ya?ml)"
    r"|private[-_]key(?:[-_][^.]+)?\.(?:pem|key)"
    r")\Z"
)
CANARY_ARTIFACT_RE = re.compile(
    r"(?:"
    r"\.?canary[-_]?(?:artifact|output|result|response|request|payload|session|trace|log)(?:[._-].*)?"
    r"|canary\.(?:jsonl?|log|txt)"
    r"|(?:live|active|temporary|temp|sub2api)[-_]?canary(?:[._-].*)?"
    r"|\.canary(?:[._-].*)?"
    r")\Z"
)


class PackageFileListError(ValueError):
    """Raised when a package file-list entry violates the release boundary."""


class CargoPackageListError(RuntimeError):
    """Raised when Cargo cannot produce a trustworthy package file list."""


def _fail(line_number: int, reason: str) -> None:
    raise PackageFileListError(f"line {line_number}: {reason}")


def _has_control_character(path: str) -> bool:
    return any(unicodedata.category(character) in {"Cc", "Cf"} for character in path)


def _validate_path(path: str, line_number: int) -> None:
    if not path:
        _fail(line_number, "empty path")
    if _has_control_character(path):
        _fail(line_number, "control character in path")

    normalized = path.replace("\\", "/")
    if normalized.startswith(("/", "~/")) or normalized == "~":
        _fail(line_number, "absolute or user-relative path")
    if re.match(r"[A-Za-z]:", normalized):
        _fail(line_number, "Windows drive-qualified path")

    components = tuple(component for component in normalized.split("/") if component != "")
    if not components:
        _fail(line_number, "empty path")
    if ".." in components:
        _fail(line_number, "parent-directory traversal")

    folded_components = tuple(component.casefold() for component in components)
    forbidden_component = next(
        (component for component in folded_components if component in FORBIDDEN_COMPONENTS),
        None,
    )
    if forbidden_component is not None:
        _fail(line_number, f"forbidden release directory {forbidden_component!r}")

    basename = folded_components[-1]
    if basename in FORBIDDEN_BASENAMES:
        _fail(line_number, "credential, private configuration, or editor-state file")
    if basename.startswith(".env.") and basename not in {
        ".env.example",
        ".env.sample",
        ".env.template",
    }:
        _fail(line_number, "private environment configuration")
    if PRIVATE_CONFIG_RE.fullmatch(basename) or PRIVATE_DATA_RE.fullmatch(basename):
        _fail(line_number, "credential or private configuration file")
    if basename.endswith((".p12", ".pfx", ".jks", ".keystore")):
        _fail(line_number, "credential store")

    if (
        basename.endswith(("~", ".tmp", ".temp", ".swp", ".swo", ".bak"))
        or (basename.endswith(".orig") and basename != "cargo.toml.orig")
        or basename.startswith(".#")
        or (basename.startswith("#") and basename.endswith("#"))
        or basename.endswith(".code-workspace")
    ):
        _fail(line_number, "temporary or editor-state file")
    if CANARY_ARTIFACT_RE.fullmatch(basename):
        _fail(line_number, "temporary or live canary artifact")


def validate_package_file_list(stream: BinaryIO) -> int:
    """Validate package paths from a binary stream and return their count."""

    entry_count = 0
    while True:
        raw_line = stream.readline(MAX_PATH_BYTES + 2)
        if not raw_line:
            break

        entry_count += 1
        if entry_count > MAX_ENTRIES:
            raise PackageFileListError(
                f"package file list exceeds {MAX_ENTRIES} entries"
            )

        if raw_line.endswith(b"\n"):
            raw_path = raw_line[:-1]
            if raw_path.endswith(b"\r"):
                raw_path = raw_path[:-1]
        else:
            raw_path = raw_line

        if len(raw_path) > MAX_PATH_BYTES:
            _fail(entry_count, f"path exceeds {MAX_PATH_BYTES} bytes")
        try:
            path = raw_path.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise PackageFileListError(
                f"line {entry_count}: path is not valid UTF-8"
            ) from error

        _validate_path(path, entry_count)

    if entry_count == 0:
        raise PackageFileListError("package file list is empty")
    return entry_count


def cargo_package_list_command(*, allow_dirty: bool) -> list[str]:
    """Return the Cargo-native package-list command for this workspace."""

    command = ["cargo", "package", "--workspace", "--list", "--locked"]
    if allow_dirty:
        command.append("--allow-dirty")
    return command


def capture_cargo_package_list(*, allow_dirty: bool) -> bytes:
    """Run Cargo and return its bounded package-list output after success.

    The output is deliberately not validated until Cargo exits successfully:
    a partial list from a failed package command cannot prove a release-safe
    package set.
    """

    process = subprocess.Popen(
        cargo_package_list_command(allow_dirty=allow_dirty),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise CargoPackageListError("cargo package --list did not expose standard output")

    output = bytearray()
    output_exceeded_limit = False
    try:
        while chunk := process.stdout.read(READ_CHUNK_BYTES):
            remaining = MAX_CARGO_STDOUT_BYTES - len(output)
            if remaining <= 0:
                output_exceeded_limit = True
                continue
            if len(chunk) > remaining:
                output.extend(chunk[:remaining])
                output_exceeded_limit = True
            else:
                output.extend(chunk)
    finally:
        process.stdout.close()

    return_code = process.wait()
    if return_code != 0:
        raise CargoPackageListError(
            f"cargo package --list failed with exit status {return_code}"
        )
    if output_exceeded_limit:
        raise CargoPackageListError(
            f"cargo package --list output exceeds {MAX_CARGO_STDOUT_BYTES} bytes"
        )
    return bytes(output)


def check_cargo_package_file_list(*, allow_dirty: bool) -> int:
    """Validate Cargo's complete successful package-list output."""

    output = capture_cargo_package_list(allow_dirty=allow_dirty)
    return validate_package_file_list(io.BytesIO(output))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="pass Cargo's local dirty-worktree allowance without publishing",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        entry_count = check_cargo_package_file_list(allow_dirty=args.allow_dirty)
    except (CargoPackageListError, PackageFileListError) as error:
        print(f"package file-list check failed: {error}", file=sys.stderr)
        return 1

    print(f"validated {entry_count} packaged paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
