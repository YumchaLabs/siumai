# Fearless Refactor V4 - Evidence And Gates

Last updated: 2026-05-17

## Evidence Anchors

- Architecture docs: `docs/architecture/module-split-design.md`
- ADRs:
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0006-family-model-first-trait-policy.md`
  - `docs/adr/0007-llmclient-demotion-policy.md`
- Task ledger: `docs/workstreams/fearless-refactor-v4/todo.md`
- Validation matrix: `docs/workstreams/fearless-refactor-v4/validation-matrix.md`
- Current OpenAI-compatible boundary files:
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/runtime.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/compatibility.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/types.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/completion/mod.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/completion/streaming.rs`
  - `siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/mod.rs`

## Required Gates

- `cargo fmt -p siumai-provider-openai-compatible`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast completion`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast openai_client`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
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
