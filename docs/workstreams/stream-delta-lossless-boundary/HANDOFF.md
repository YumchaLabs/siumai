# Stream Delta Lossless Boundary - Handoff

Status: Closed
Last updated: 2026-05-17

## Final State

The immediate issue #19 fix is already committed as:

- `c64b8980 fix(openai-compatible): preserve whitespace stream deltas`

This workstream hardens the architecture seam so generated stream deltas remain lossless by
construction:

- OpenAI-compatible generated content/reasoning extraction now uses explicit generated-field
  helpers instead of ambiguous control-field helpers.
- Known generated JSON paths preserve empty strings and whitespace-only strings.
- Shared stream factory tests prove empty raw SSE frames are skipped while whitespace-bearing frames
  reach converters.
- OpenAI-compatible provider chat and completion stream public paths have regression coverage.
- Completion streaming now declares `[DONE]` as converter-owned stream end framing.

## Validation

Passed:

- `cargo nextest run -p siumai-protocol-openai --all-features`
- `cargo nextest run -p siumai-core streaming::factory`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features`

- `git diff --check`

## Follow-ons

No split follow-up. Future provider stream work should apply the same invariant when adding new
stream adapters: generated text/reasoning/tool-argument deltas are payload and must not be trimmed
or dropped because they are whitespace-only.
