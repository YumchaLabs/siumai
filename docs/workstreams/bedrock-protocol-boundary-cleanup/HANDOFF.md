# Bedrock Protocol Boundary Cleanup - Handoff

Last updated: 2026-05-17

## Current State

The workstream is active. Scope is intentionally narrow: improve Bedrock chat protocol locality
without changing public module paths or wire behavior.

## Current Task

BPC-030: reassess production request planning, response metadata, and stream conversion after test
isolation.

BPC-020 is complete:

- Inline Bedrock chat standard tests moved from
  `siumai-provider-amazon-bedrock/src/standards/bedrock/chat.rs` to
  `siumai-provider-amazon-bedrock/src/standards/bedrock/chat/tests.rs`.
- `chat.rs` now keeps production protocol code plus `#[cfg(test)] mod tests;`.
- Source guards still read the production file through `include_str!("../chat.rs")`.
- Focused Bedrock gate passed: 74 tests passed, 0 skipped.

## Guardrails

- Preserve public module paths.
- Preserve request, response, and stream behavior.
- Keep source guards that prevent request/response metadata direction regressions.
- Do not split production code by file size alone.

## Validation Commands

- `cargo fmt -p siumai-provider-amazon-bedrock`
- `cargo nextest run -p siumai-provider-amazon-bedrock --all-features --no-fail-fast bedrock`
- `git diff --check`
