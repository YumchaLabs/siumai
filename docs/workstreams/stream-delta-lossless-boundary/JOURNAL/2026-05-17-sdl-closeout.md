# 2026-05-17 - SDL Closeout

## Shipped

- Added `stream-delta-lossless-boundary` workstream docs and linked it from `docs/README.md`.
- Added generated-field extraction helpers for OpenAI-compatible field access.
- Kept legacy accessor method names as wrappers while routing production streaming through the
  generated-field helpers.
- Preserved empty and whitespace-only chat content/reasoning deltas.
- Preserved empty JSON-string Responses-style deltas.
- Tightened shared stream factory coverage for whitespace-only event data versus true empty frames.
- Added OpenAI-compatible provider public-path coverage for chat text/reasoning delta losslessness.
- Fixed completion stream `[DONE]` ownership and made completion `choices[].text` field-presence
  based, with empty and whitespace-only delta coverage.

## Gates

- `cargo fmt -p siumai-core -p siumai-protocol-openai -p siumai-provider-openai-compatible`
- `cargo nextest run -p siumai-protocol-openai --all-features`
- `cargo nextest run -p siumai-core streaming::factory`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features`

## Decision

Close the lane without a split follow-up. The target seam is now explicit and tested for the
OpenAI-compatible chat and completion paths.
