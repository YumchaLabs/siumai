#!/usr/bin/env python3
"""Audit Siumai model catalogs against a local Vercel AI SDK checkout.

The audit is intentionally source-based and side-effect free:
- extracts AI SDK `export type *Model*Id = ...` string-literal unions
- extracts Rust string literals from Siumai model catalog files
- reports upstream ids that are not present in the mapped Siumai sources
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from resolve_ai_sdk_repo import candidate_paths, is_ai_sdk_repo
except ImportError as exc:  # pragma: no cover - only happens if moved incorrectly.
    raise SystemExit(f"Cannot import resolve_ai_sdk_repo.py: {exc}") from exc


@dataclass(frozen=True)
class AuditTarget:
    name: str
    package: str
    type_regex: str
    rust_paths: tuple[str, ...]
    note: str = ""


TARGETS: tuple[AuditTarget, ...] = (
    AuditTarget(
        name="alibaba",
        package="alibaba",
        type_regex=r"Alibaba(Chat|Video)ModelId",
        rust_paths=(
            "siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/alibaba.rs",
        ),
    ),
    AuditTarget(
        name="anthropic",
        package="anthropic",
        type_regex=r"AnthropicModelId",
        rust_paths=(
            "siumai-provider-anthropic/src/providers/anthropic/model_constants.rs",
            "siumai-provider-anthropic/src/providers/anthropic/models.rs",
        ),
    ),
    AuditTarget(
        name="cohere",
        package="cohere",
        type_regex=r"Cohere(Chat|Embedding|Reranking)ModelId",
        rust_paths=("siumai-provider-cohere/src/providers/cohere/models.rs",),
    ),
    AuditTarget(
        name="deepinfra",
        package="deepinfra",
        type_regex=r"DeepInfra(Chat|Completion|Embedding|Image)ModelId",
        rust_paths=(
            "siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/deepinfra.rs",
        ),
        note="Large curated catalogs may need a policy decision before bulk updates.",
    ),
    AuditTarget(
        name="deepseek",
        package="deepseek",
        type_regex=r"DeepSeekChatModelId",
        rust_paths=(
            "siumai-provider-deepseek/src/providers/deepseek/models.rs",
            "siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/deepseek.rs",
        ),
    ),
    AuditTarget(
        name="fireworks",
        package="fireworks",
        type_regex=r"Fireworks(Chat|Completion|Embedding|Image)ModelId",
        rust_paths=(
            "siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/fireworks.rs",
        ),
    ),
    AuditTarget(
        name="google",
        package="google",
        type_regex=r"Google(ModelId|EmbeddingModelId|ImageModelId|VideoModelId|InteractionsModelId)",
        rust_paths=(
            "siumai/src/provider_ext/gemini/models.rs",
            "siumai-provider-gemini/src/providers/gemini/interactions.rs",
            "siumai-provider-gemini/src/providers/gemini/model_constants.rs",
        ),
    ),
    AuditTarget(
        name="google-vertex",
        package="google-vertex",
        type_regex=(
            r"GoogleVertex(ModelId|EmbeddingModelId|ImageModelId|VideoModelId|AnthropicModelId|XaiModelId|MaasModelId)"
        ),
        rust_paths=(
            "siumai-provider-google-vertex/src/providers/vertex/models.rs",
            "siumai-provider-google-vertex/src/providers/anthropic_vertex/models.rs",
            "siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/google_vertex_xai.rs",
            "siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/vertex_maas.rs",
        ),
    ),
    AuditTarget(
        name="groq",
        package="groq",
        type_regex=r"Groq(Chat|Transcription)ModelId",
        rust_paths=("siumai-provider-groq/src/providers/groq/models.rs",),
    ),
    AuditTarget(
        name="mistral",
        package="mistral",
        type_regex=r"Mistral(Chat|Embedding)ModelId",
        rust_paths=(
            "siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/mistral.rs",
        ),
    ),
    AuditTarget(
        name="moonshotai",
        package="moonshotai",
        type_regex=r"MoonshotAIChatModelId",
        rust_paths=(
            "siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/moonshotai.rs",
            "siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/moonshot.rs",
        ),
    ),
    AuditTarget(
        name="openai",
        package="openai",
        type_regex=(
            r"OpenAI(Chat|Completion|Embedding|Image|Speech|Transcription|Responses)ModelId"
        ),
        rust_paths=(
            "siumai-provider-openai/src/providers/openai/model_constants.rs",
            "siumai-provider-openai/src/providers/openai/models.rs",
        ),
    ),
    AuditTarget(
        name="perplexity",
        package="perplexity",
        type_regex=r"PerplexityLanguageModelId",
        rust_paths=(
            "siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/perplexity.rs",
        ),
    ),
    AuditTarget(
        name="togetherai",
        package="togetherai",
        type_regex=r"TogetherAI(Chat|Completion|Embedding|Image|Reranking)ModelId",
        rust_paths=(
            "siumai-provider-togetherai/src/providers/togetherai/models.rs",
            "siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/togetherai.rs",
            "siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/together.rs",
        ),
    ),
    AuditTarget(
        name="xai",
        package="xai",
        type_regex=r"Xai(Chat|Responses|Image|Video)ModelId",
        rust_paths=(
            "siumai-provider-xai/src/providers/xai/models.rs",
            "siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/xai.rs",
        ),
    ),
)


SKIPPED_PACKAGES: dict[str, str] = {
    "amazon-bedrock": "Siumai intentionally avoids inventing a static Bedrock catalog; it returns configured model ids.",
    "azure": "Azure OpenAI deployments are user-defined names rather than fixed upstream model ids.",
    "gateway": "AI Gateway is an aggregator surface, not a Siumai provider package catalog.",
    "openai-compatible": "The generic OpenAI-compatible package accepts arbitrary strings.",
}


TYPE_ALIAS_RE = re.compile(
    r"export\s+type\s+([A-Za-z0-9_]*Model[A-Za-z0-9_]*Id)\s*=\s*(.*?);",
    re.DOTALL,
)
TS_STRING_RE = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'")


def find_siumai_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "Cargo.toml").is_file() and (
            candidate / ".agents" / "skills" / "siumai-ai-sdk-maintenance"
        ).is_dir():
            return candidate
    return current


def resolve_ai_sdk(start: Path, override: str | None) -> Path:
    if override:
        path = Path(override).expanduser()
        if is_ai_sdk_repo(path):
            return path.resolve()
        raise SystemExit(f"AI SDK path does not look valid: {path}")

    for candidate in candidate_paths(start):
        if is_ai_sdk_repo(candidate):
            return candidate.resolve()
    raise SystemExit("Cannot find AI SDK repo. Set AI_SDK_REPO or pass --ai-sdk-repo.")


def iter_ts_source_files(package_dir: Path) -> Iterable[Path]:
    src = package_dir / "src"
    if not src.is_dir():
        return ()
    return (
        path
        for path in sorted(src.rglob("*.ts"))
        if not path.name.endswith((".test.ts", ".test-d.ts"))
    )


def parse_ts_model_id_unions(path: Path) -> dict[str, set[str]]:
    text = path.read_text(encoding="utf-8")
    out: dict[str, set[str]] = {}
    for match in TYPE_ALIAS_RE.finditer(text):
        type_name = match.group(1)
        body = match.group(2)
        ids = {m.group(1).encode("utf-8").decode("unicode_escape") for m in TS_STRING_RE.finditer(body)}
        if ids:
            out.setdefault(type_name, set()).update(ids)
    return out


def rust_string_literals(text: str) -> set[str]:
    literals: set[str] = set()
    current: list[str] = []
    state = "code"
    i = 0

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line_comment"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block_comment"
                i += 2
                continue
            if ch == '"':
                current = []
                state = "string"
                i += 1
                continue
            i += 1
            continue

        if state == "line_comment":
            if ch in "\r\n":
                state = "code"
            i += 1
            continue

        if state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
            i += 1
            continue

        if state == "string":
            if ch == "\\" and nxt:
                current.append(ch)
                current.append(nxt)
                i += 2
                continue
            if ch == '"':
                value = "".join(current)
                try:
                    value = value.encode("utf-8").decode("unicode_escape")
                except UnicodeDecodeError:
                    pass
                literals.add(value)
                state = "code"
                i += 1
                continue
            current.append(ch)
            i += 1
            continue

    return literals


def collect_upstream(ai_sdk: Path, target: AuditTarget) -> tuple[set[str], dict[str, list[str]], list[str]]:
    package_dir = ai_sdk / "packages" / target.package
    if not package_dir.is_dir():
        return set(), {}, [f"missing AI SDK package: {target.package}"]

    type_re = re.compile(target.type_regex)
    ids: set[str] = set()
    type_sources: dict[str, list[str]] = {}
    warnings: list[str] = []

    for path in iter_ts_source_files(package_dir):
        for type_name, type_ids in parse_ts_model_id_unions(path).items():
            if not type_re.fullmatch(type_name):
                continue
            ids.update(type_ids)
            rel = path.relative_to(ai_sdk).as_posix()
            type_sources.setdefault(type_name, []).append(rel)

    if not ids:
        warnings.append(f"no upstream ids found for {target.name} using {target.type_regex}")
    return ids, type_sources, warnings


def collect_rust(root: Path, target: AuditTarget) -> tuple[set[str], list[str]]:
    ids: set[str] = set()
    warnings: list[str] = []
    for rel in target.rust_paths:
        path = root / rel
        if not path.is_file():
            warnings.append(f"missing Siumai source: {rel}")
            continue
        ids.update(rust_string_literals(path.read_text(encoding="utf-8")))
    return ids, warnings


def packages_with_model_unions(ai_sdk: Path) -> set[str]:
    packages = ai_sdk / "packages"
    out: set[str] = set()
    for package in sorted(path for path in packages.iterdir() if path.is_dir()):
        for ts in iter_ts_source_files(package):
            if parse_ts_model_id_unions(ts):
                out.add(package.name)
                break
    return out


def audit(root: Path, ai_sdk: Path, selected: set[str] | None, deferred: set[str]) -> dict[str, object]:
    target_names = {target.name for target in TARGETS}
    unknown = selected.difference(target_names) if selected else set()
    unknown.update(deferred.difference(target_names))
    if unknown:
        raise SystemExit(f"Unknown provider target(s): {', '.join(sorted(unknown))}")

    results: list[dict[str, object]] = []
    for target in TARGETS:
        if selected and target.name not in selected:
            continue
        upstream_ids, type_sources, upstream_warnings = collect_upstream(ai_sdk, target)
        rust_ids, rust_warnings = collect_rust(root, target)
        missing = sorted(upstream_ids - rust_ids)
        status = "green"
        if missing:
            status = "deferred" if target.name in deferred else "red"
        results.append(
            {
                "provider": target.name,
                "package": target.package,
                "status": status,
                "upstream_count": len(upstream_ids),
                "rust_literal_count": len(rust_ids),
                "missing_count": len(missing),
                "missing": missing,
                "type_sources": type_sources,
                "rust_paths": list(target.rust_paths),
                "warnings": upstream_warnings + rust_warnings,
                "note": target.note,
            }
        )

    mapped_packages = {target.package for target in TARGETS}
    union_packages = packages_with_model_unions(ai_sdk)
    unmapped = sorted(union_packages - mapped_packages - set(SKIPPED_PACKAGES))

    return {
        "siumai_root": str(root),
        "ai_sdk_repo": str(ai_sdk),
        "results": results,
        "skipped_packages": SKIPPED_PACKAGES,
        "unmapped_ai_sdk_packages_with_model_unions": unmapped,
    }


def print_text(report: dict[str, object], include_green: bool, show_skipped: bool) -> None:
    print(f"Siumai root: {report['siumai_root']}")
    print(f"AI SDK repo: {report['ai_sdk_repo']}")
    print()

    results = report["results"]
    assert isinstance(results, list)
    reds = [item for item in results if item["status"] == "red"]
    deferred = [item for item in results if item["status"] == "deferred"]
    greens = [item for item in results if item["status"] == "green"]

    if reds:
        print("Red model catalog drift:")
        for item in reds:
            print(
                f"- {item['provider']}: missing {item['missing_count']} of {item['upstream_count']} upstream ids"
            )
            note = item.get("note")
            if note:
                print(f"  note: {note}")
            for model_id in item["missing"]:
                print(f"  - {model_id}")
            for warning in item["warnings"]:
                print(f"  warning: {warning}")
        print()
    else:
        print("No red model catalog drift found.")
        print()

    if deferred:
        print("Deferred model catalog drift:")
        for item in deferred:
            print(
                f"- {item['provider']}: deferred {item['missing_count']} of {item['upstream_count']} upstream ids"
            )
            note = item.get("note")
            if note:
                print(f"  note: {note}")
            for model_id in item["missing"]:
                print(f"  - {model_id}")
            for warning in item["warnings"]:
                print(f"  warning: {warning}")
        print()

    if include_green and greens:
        print("Green targets:")
        for item in greens:
            print(f"- {item['provider']}: {item['upstream_count']} upstream ids covered")
        print()

    if show_skipped:
        print("Skipped packages:")
        skipped = report["skipped_packages"]
        assert isinstance(skipped, dict)
        for package, reason in sorted(skipped.items()):
            print(f"- {package}: {reason}")
        print()

        unmapped = report["unmapped_ai_sdk_packages_with_model_unions"]
        assert isinstance(unmapped, list)
        if unmapped:
            print("Unmapped AI SDK packages with model-id unions:")
            for package in unmapped:
                print(f"- {package}")
            print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--siumai-root", default=None, help="Path to the Siumai repo root.")
    parser.add_argument("--ai-sdk-repo", default=None, help="Path to a local Vercel AI SDK checkout.")
    parser.add_argument(
        "--provider",
        action="append",
        help="Audit only one provider target. May be repeated.",
    )
    parser.add_argument(
        "--defer",
        action="append",
        default=[],
        help="Treat a provider target with missing ids as deferred instead of red. May be repeated.",
    )
    parser.add_argument("--include-green", action="store_true", help="Print green targets in text output.")
    parser.add_argument("--show-skipped", action="store_true", help="Print skipped/unmapped AI SDK packages.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    root = find_siumai_root(Path(args.siumai_root) if args.siumai_root else Path.cwd())
    ai_sdk = resolve_ai_sdk(root, args.ai_sdk_repo)
    selected = set(args.provider) if args.provider else None
    report = audit(root, ai_sdk, selected, set(args.defer))

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report, include_green=args.include_green, show_skipped=args.show_skipped)

    red_count = sum(1 for item in report["results"] if item["status"] == "red")
    return 1 if red_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
