# Fearless Refactor V4 - Evidence And Gates

Last updated: 2026-05-17

## Evidence Anchors

- Architecture docs: `docs/architecture/module-split-design.md`
- ADRs:
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0006-family-model-first-trait-policy.md`
  - `docs/adr/0007-llmclient-demotion-policy.md`
- Task ledger: `docs/workstreams/fearless-refactor-v4/todo.md`
- Follow-ons: `docs/workstreams/fearless-refactor-v4/follow-ons.md`
- Validation matrix: `docs/workstreams/fearless-refactor-v4/validation-matrix.md`
- Typed metadata matrix: `docs/workstreams/fearless-refactor-v4/typed-metadata-boundary-matrix.md`
- Provider capability matrix:
  `docs/workstreams/fearless-refactor-v4/provider-capability-alignment-matrix.md`
- Hosted search scope: `docs/workstreams/fearless-refactor-v4/hosted-search-surface.md`
- Current OpenAI-compatible boundary files:
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/runtime.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/compatibility.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/types.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/completion/mod.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/completion/streaming.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/settings.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/settings/tests.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/config.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/config/builtin_providers.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/config/family_defaults.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/builder.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/builder/reasoning.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/builder/tests.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/ext/request_options.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/ext/request_options/tests.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/mod.rs`

## V4 Core Closeout Decision

The V4 core architecture workstream is closed as of 2026-05-17. The remaining backlog is
matrix-driven follow-on work, not unfinished V4 core architecture. New work should start from a
concrete provider behavior gap, public API gap, or documented provider evidence.

## Track J Closeout Decision

Track J - OpenAI-compatible internal boundary cleanup is closed as of 2026-05-17. The remaining
large files are intentional capability-local implementations, static provider data, or public
facades. Future work should start only from a concrete behavior bug, API drift, or real
ownership/coupling problem; do not continue the lane by mechanically splitting files.

## Required Gates

- Historical code gates are recorded below for the lanes that changed code.
- Docs-only closeout gate:
  - `git diff --check`

## Validation Log

- OpenAI-compatible provider model catalog split:
  - `cargo fmt -p siumai-provider-openai-compatible`
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast providers::openai_compatible::providers::models`
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
  - `git diff --check`
- OpenAI-compatible client shell split:
  - `cargo fmt -p siumai-provider-openai-compatible`
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast openai_client`
    - Result: 84 tests passed, 142 skipped.
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
    - Result: 227 tests passed, 0 skipped.
  - `git diff --check`
    - Result: passed; Git reported Windows LF-to-CRLF working-copy warnings only.
- OpenAI-compatible client test module isolation:
  - `cargo fmt -p siumai-provider-openai-compatible`
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast openai_client`
    - Result: 85 tests passed, 142 skipped.
- OpenAI-compatible completion runtime split:
  - `cargo fmt -p siumai-provider-openai-compatible`
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast completion`
    - Result: 12 tests passed, 216 skipped.
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast openai_client`
    - Result: 86 tests passed, 142 skipped.
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
    - Result: 228 tests passed, 0 skipped.
- OpenAI-compatible simple provider settings adapter convergence:
  - `cargo fmt -p siumai-provider-openai-compatible`
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast settings`
    - Result: 16 tests passed, 213 skipped.
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
    - Result: 229 tests passed, 0 skipped.
- OpenAI-compatible provider family-default split:
  - `cargo fmt -p siumai-provider-openai-compatible`
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast config`
    - Result: 31 tests passed, 199 skipped.
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
    - Result: 230 tests passed, 0 skipped.
- OpenAI-compatible built-in provider registry split:
  - `cargo fmt -p siumai-provider-openai-compatible`
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast config`
    - Result: 32 tests passed, 199 skipped.
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
    - Result: 231 tests passed, 0 skipped.
- OpenAI-compatible builder reasoning mapping split:
  - `cargo fmt -p siumai-provider-openai-compatible`
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast builder`
    - Result: 22 tests passed, 210 skipped.
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
    - Result: 232 tests passed, 0 skipped.
- OpenAI-compatible request-option extension test isolation:
  - `cargo fmt -p siumai-provider-openai-compatible`
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast request_options`
    - Result: 20 tests passed, 213 skipped.
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
    - Result: 233 tests passed, 0 skipped.
- OpenAI-compatible builder test module isolation:
  - `cargo fmt -p siumai-provider-openai-compatible`
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast builder`
    - Result: 22 tests passed, 211 skipped.
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
    - Result: 233 tests passed, 0 skipped.
- OpenAI-compatible settings test module isolation:
  - `cargo fmt -p siumai-provider-openai-compatible`
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast settings`
    - Result: 16 tests passed, 217 skipped.
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
    - Result: 233 tests passed, 0 skipped.
- OpenAI-compatible internal boundary cleanup closeout:
  - Closeout audit result: no further OpenAI-compatible split is justified without a concrete
    behavior/API/coupling trigger.
  - Latest code gate before closeout:
    `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
    - Result: 233 tests passed, 0 skipped.
  - Docs-only closeout gate:
    `git diff --check`
- V4 core ledger closeout:
  - `todo.md` has no remaining `[~]` or `[ ]` implementation items for the V4 core ledger.
  - Remaining typed metadata, provider capability, hosted-search, and OpenAI-compatible internal
    boundary work is explicitly routed through `follow-ons.md`.
  - Docs-only closeout gate:
    `git diff --check`
