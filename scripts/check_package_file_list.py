#!/usr/bin/env python3
"""Reject private files from Cargo's authoritative package file list."""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import BinaryIO


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_CARGO_STDOUT_BYTES = 16 * 1024 * 1024
MAX_PATH_BYTES = 4_096
READ_CHUNK_BYTES = 64 * 1024
FORBIDDEN_COMPONENTS = frozenset(
    {
        ".codex",
        ".codex-helper",
        ".fleet",
        ".git",
        ".hg",
        ".idea",
        ".svn",
        ".vs",
        ".vscode",
        ".zed",
        "repo-ref",
        "target",
    }
)
FORBIDDEN_NAMES = frozenset(
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
        "secrets.json",
        "secrets.toml",
        "secrets.yaml",
        "secrets.yml",
        "thumbs.db",
    }
)
PRIVATE_CONFIG_PATTERN = re.compile(
    r"(?:config|settings)\.local(?:\.(?:json|toml|ya?ml|ini))?\Z"
)
PRIVATE_DATA_PATTERN = re.compile(
    r"(?:"
    r"secrets?\.(?:json|toml|ya?ml|ini|pem|key)"
    r"|service[-_]account(?:[-_][^.]+)?\.(?:json|toml|ya?ml)"
    r"|private[-_]key(?:[-_][^.]+)?\.(?:pem|key)"
    r")\Z"
)
CANARY_ARTIFACT_PATTERN = re.compile(
    r"(?:"
    r"\.?canary(?:[-_](?:artifact|output|result|response|request|payload|session|trace|log))?"
    r"|(?:live|active|temporary|temp|sub2api)[-_]canary"
    r"(?:[-_](?:artifact|output|result|response|request|payload|session|trace|log))?"
    r")\.(?:jsonl?|ndjson|log|txt|trace)\Z"
)


class PackageFileListError(RuntimeError):
    """Raised when Cargo fails or a private path would be packaged."""


def cargo_package_list_command(*, allow_dirty: bool) -> list[str]:
    command = ["cargo", "package", "--workspace", "--list", "--locked"]
    if allow_dirty:
        command.append("--allow-dirty")
    return command


def is_private_path(raw_path: str) -> bool:
    path = PurePosixPath(raw_path.replace("\\", "/"))
    components = tuple(component.casefold() for component in path.parts)
    name = path.name.casefold()

    if any(component in FORBIDDEN_COMPONENTS for component in components):
        return True
    if name in FORBIDDEN_NAMES:
        return True
    if name.startswith(".env."):
        return True
    if PRIVATE_CONFIG_PATTERN.fullmatch(name) or PRIVATE_DATA_PATTERN.fullmatch(name):
        return True
    if CANARY_ARTIFACT_PATTERN.fullmatch(name):
        return True
    if name.endswith((".p12", ".pfx", ".jks", ".keystore")):
        return True
    return (
        name.endswith((".bak", ".orig", ".swo", ".swp", ".temp", ".tmp", "~"))
        and name != "cargo.toml.orig"
    ) or name.startswith(".#") or (
        name.startswith("#") and name.endswith("#")
    ) or name.endswith(".code-workspace")


def validate_package_file_list(stream: BinaryIO) -> int:
    count = 0
    while raw_line := stream.readline(MAX_PATH_BYTES + 2):
        count += 1
        raw_path = raw_line.rstrip(b"\r\n")
        if len(raw_path) > MAX_PATH_BYTES:
            raise PackageFileListError(
                f"package path at line {count} exceeds {MAX_PATH_BYTES} bytes"
            )
        try:
            path = raw_path.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise PackageFileListError(
                f"package path at line {count} is not valid UTF-8"
            ) from error
        if not path:
            raise PackageFileListError(f"package path at line {count} is empty")
        if is_private_path(path):
            raise PackageFileListError(
                f"private or local file would be packaged at line {count}"
            )

    if count == 0:
        raise PackageFileListError("cargo package --list returned no files")
    return count


def capture_cargo_package_file_list(*, allow_dirty: bool) -> bytes:
    process = subprocess.Popen(
        cargo_package_list_command(allow_dirty=allow_dirty),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise PackageFileListError("cargo package --list did not expose stdout")

    output = bytearray()
    exceeded_limit = False
    try:
        while chunk := process.stdout.read(READ_CHUNK_BYTES):
            remaining = MAX_CARGO_STDOUT_BYTES - len(output)
            if remaining <= 0:
                exceeded_limit = True
                continue
            output.extend(chunk[:remaining])
            exceeded_limit |= len(chunk) > remaining
    finally:
        process.stdout.close()

    return_code = process.wait()
    if return_code != 0:
        raise PackageFileListError(
            f"cargo package --list failed with exit status {return_code}"
        )
    if exceeded_limit:
        raise PackageFileListError(
            f"cargo package --list output exceeds {MAX_CARGO_STDOUT_BYTES} bytes"
        )
    return bytes(output)


def check_cargo_package_file_list(*, allow_dirty: bool) -> int:
    output = capture_cargo_package_file_list(allow_dirty=allow_dirty)
    return validate_package_file_list(io.BytesIO(output))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow Cargo to inspect a dirty local worktree without publishing",
    )
    return parser.parse_args()


def main() -> int:
    try:
        count = check_cargo_package_file_list(allow_dirty=parse_args().allow_dirty)
    except PackageFileListError as error:
        print(f"package file-list check failed: {error}", file=sys.stderr)
        return 1

    print(f"validated {count} packaged paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
