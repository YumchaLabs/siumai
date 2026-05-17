# Bedrock Protocol Boundary Cleanup - Handoff

Last updated: 2026-05-17

## Current State

The workstream is closed. Bedrock chat protocol locality improved without changing public module
paths or wire behavior.

## Current Task

BPC-030 is complete.

- Bedrock stream JSON conversion now lives in
  `siumai-provider-amazon-bedrock/src/standards/bedrock/chat/streaming.rs`.
- `chat.rs` keeps the public `BedrockEventConverter` path stable via `pub use`.
- The extraction is justified by locality: Bedrock stream DTOs, accumulator state, raw chunk
  handling, terminal stream events, and `JsonEventConverter` implementation move together.
- Request planning stays in `chat.rs` because it is still tightly coupled to Bedrock message,
  tool, reasoning, cache point, and file conversion helpers.
- Non-stream response shaping stays in `chat.rs` because it shares Bedrock metadata, usage, finish
  reason, and tool-id normalization helpers with the standard shell.
- The source guard now scans both `chat.rs` and `chat/streaming.rs` for response/stream
  provider-option direction regressions.
- Focused Bedrock gate passed after extraction: 74 tests passed, 0 skipped.

BPC-020 is complete:

- Inline Bedrock chat standard tests moved from
  `siumai-provider-amazon-bedrock/src/standards/bedrock/chat.rs` to
  `siumai-provider-amazon-bedrock/src/standards/bedrock/chat/tests.rs`.
- `chat.rs` now keeps production protocol code plus `#[cfg(test)] mod tests;`.
- Source guards still read the production file through `include_str!("../chat.rs")`; the
  response/stream guard also includes `chat/streaming.rs`.
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

## Next Recommendation

No immediate follow-on is recommended. The remaining request/response code should not be split just
to reduce file size; open a new workstream only for a concrete Bedrock behavior issue or a helper
with proven reuse across Bedrock chat, embedding, image, or rerank.

## Residual Risk

Focused Bedrock no-network tests passed, but this lane did not run live AWS Bedrock integration
tests. Wire behavior is protected by existing contract tests and source guards, not by live service
round trips.
