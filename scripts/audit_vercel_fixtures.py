"""
Audit fixture coverage and drift against the upstream Vercel AI SDK repo.

This script is intentionally lightweight:
- It treats upstream fixtures as "covered" if either:
  - a file with the same name exists anywhere under `siumai/tests/fixtures`, OR
  - a directory with the same case id exists anywhere under `siumai/tests/fixtures`
    (case id = filename without extension, e.g. `anthropic-json-tool.1`).
- It detects "drift" for 1:1 file matches (single match by basename) by comparing SHA-256.

Run:
  python scripts/audit_vercel_fixtures.py --ai-root ../ai --siumai-root .
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class UpstreamFixture:
    package: str
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def case_id(self) -> str:
        # "foo.1.chunks.txt" -> "foo.1"
        # "foo.1.json" -> "foo.1"
        parts = self.name.split(".")
        if len(parts) >= 3 and parts[-2:] == ["chunks", "txt"]:
            return ".".join(parts[:-2])
        if len(parts) >= 2:
            return ".".join(parts[:-1])
        return self.name


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_upstream_fixtures(ai_root: Path) -> list[UpstreamFixture]:
    packages_root = ai_root / "packages"
    out: list[UpstreamFixture] = []
    for fixtures_dir in packages_root.rglob("__fixtures__"):
        if not fixtures_dir.is_dir():
            continue
        try:
            package = fixtures_dir.relative_to(packages_root).parts[0]
        except Exception:
            package = "unknown"
        for file in fixtures_dir.rglob("*"):
            if file.is_file():
                out.append(UpstreamFixture(package=package, path=file))
    return sorted(out, key=lambda f: str(f.path))


def build_siumai_fixture_index(siumai_root: Path) -> tuple[dict[str, list[Path]], set[str]]:
    fixtures_root = siumai_root / "siumai" / "tests" / "fixtures"
    file_index: dict[str, list[Path]] = {}
    dir_names: set[str] = set()

    for p in fixtures_root.rglob("*"):
        if p.is_dir():
            dir_names.add(p.name)
        elif p.is_file():
            file_index.setdefault(p.name, []).append(p)
    return file_index, dir_names


def normalize_case_id(package: str, case_id: str) -> str:
    # Our fixture case directories usually omit the provider prefix,
    # while upstream uses e.g. `openai-foo.1.json` / `xai-foo.1.json`.
    if package == "openai" and case_id.startswith("openai-"):
        return case_id[len("openai-") :]
    if package == "xai" and case_id.startswith("xai-"):
        return case_id[len("xai-") :]
    return case_id


def case_dir_exists(dir_names: set[str], normalized_case_id: str) -> bool:
    if normalized_case_id in dir_names:
        return True
    # Some suites fan out a single upstream fixture into multiple cases (e.g. json vs text documents).
    prefix = normalized_case_id + "-"
    return any(name.startswith(prefix) for name in dir_names)


def summarize_paths(paths: Iterable[Path], limit: int) -> list[str]:
    out = []
    for p in list(paths)[:limit]:
        out.append(str(p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-root", type=Path, required=True)
    ap.add_argument("--siumai-root", type=Path, required=True)
    ap.add_argument(
        "--ignore-packages",
        type=str,
        default="langchain",
        help="Comma-separated package ids to ignore (default: langchain)",
    )
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    ignore = {p.strip() for p in args.ignore_packages.split(",") if p.strip()}
    upstream = [f for f in iter_upstream_fixtures(args.ai_root) if f.package not in ignore]
    file_index, dir_names = build_siumai_fixture_index(args.siumai_root)

    covered_by_file = []
    covered_by_file_multi = []
    covered_by_case_dir = []
    missing = []
    ambiguous = []
    drift = []

    for f in upstream:
        candidates = file_index.get(f.name, [])
        if len(candidates) == 1:
            covered_by_file.append(f)
            si = candidates[0]
            if sha256(f.path) != sha256(si):
                drift.append((f, si))
            continue
        if len(candidates) > 1:
            up_hash = sha256(f.path)
            matching = [c for c in candidates if sha256(c) == up_hash]
            if matching:
                covered_by_file_multi.append((f, matching, candidates))
            else:
                ambiguous.append((f, candidates))
            continue
        normalized_case_id = normalize_case_id(f.package, f.case_id)
        if case_dir_exists(dir_names, normalized_case_id):
            covered_by_case_dir.append(f)
        else:
            missing.append(f)

    print(f"Upstream fixtures (excluding {sorted(ignore)}): {len(upstream)}")
    print(f"Covered by file match: {len(covered_by_file)}")
    print(f"Covered by file match (multi): {len(covered_by_file_multi)}")
    print(f"Covered by case-dir match: {len(covered_by_case_dir)}")
    print(f"Ambiguous file matches: {len(ambiguous)}")
    print(f"Missing: {len(missing)}")
    print(f"Drift (hash mismatch on 1:1 file match): {len(drift)}")

    if drift:
        print("\n=== Drift (showing up to limit) ===")
        for up, si in drift[: args.limit]:
            print(f"- UP: {up.path}")
            print(f"  SI: {si}")

    if missing:
        print("\n=== Missing (showing up to limit) ===")
        for up in missing[: args.limit]:
            print(f"- UP: {up.path}")

    if ambiguous:
        print("\n=== Ambiguous (showing up to limit) ===")
        for up, cands in ambiguous[: args.limit]:
            print(f"- UP: {up.path}")
            for si in summarize_paths(cands, 5):
                print(f"  - SI: {si}")

    if covered_by_file_multi:
        print("\n=== Multi-match (hash OK) (showing up to limit) ===")
        for up, matching, all_cands in covered_by_file_multi[: args.limit]:
            print(f"- UP: {up.path}")
            for si in summarize_paths(matching, 5):
                print(f"  - MATCH: {si}")
            other = [c for c in all_cands if c not in matching]
            for si in summarize_paths(other, 5):
                print(f"  - OTHER: {si}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
