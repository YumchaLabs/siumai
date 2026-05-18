# GIR-020 Request Conversion

Date: 2026-05-18

## Summary

Implemented provider-owned model-mode request conversion for Google Interactions. The conversion
turns stable `ChatRequest` structures plus `providerOptions.google` into the
`/v1beta/interactions` request body shape without enabling network runtime execution.

## Scope

- Added `providers/gemini/interactions/request.rs` for request wire shapes and conversion helpers.
- Kept `GoogleInteractionsLanguageModel` fail-fast for runtime execution.
- Covered model-mode system instructions, response formats, generation config, media blocks,
  model-mode tools/tool choice, tool calls/results, reasoning signatures, `previousInteractionId`,
  `store`, deprecated `imageConfig`, and warnings.
- Left agent-mode conversion explicitly deferred to GIR-030.

## Evidence

- `cargo fmt -p siumai-provider-gemini -- --check`
- `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_request --no-fail-fast`
- `cargo nextest run -p siumai-provider-gemini --all-features --no-fail-fast`
- `git diff --check`

## Next Slice

GIR-030 should add agent-mode request conversion and warning behavior after checking the AI SDK
agent semantics.
