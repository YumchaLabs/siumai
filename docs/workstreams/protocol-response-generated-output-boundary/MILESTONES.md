# Protocol Response Generated-Output Boundary — Milestones

Status: Active
Last updated: 2026-05-19

## M0 — Scope And Response-Lossiness Baseline

Exit criteria:

- Workstream docs exist and agree on status, scope, and task order.
- Initial response/parser hotspot scan is summarized.
- First executable task is selected.
- `WORKSTREAM.json` parses.

Current result:

- Complete. PRG-020 is the first executable code task.

## M1 — OpenAI Responses Proof Slice

Exit criteria:

- OpenAI Responses response-owned legacy content construction is behind a named adapter module or a
  narrower first sub-slice is split.
- Source guard proves response parser code does not emit non-empty request-side provider options.
- Existing OpenAI Responses fixture behavior remains stable.
- Lossiness notes classify which output shapes can or cannot project to `GenerateTextContentPart`.

## M2 — Second Provider Proof

Exit criteria:

- Anthropic and/or Gemini prove the response-adapter pattern outside OpenAI Responses.
- Citation/source/reasoning metadata remains provider-owned and response-side.
- Any provider-specific mismatch is documented as a named exception rather than left implicit.

## M3 — Bridge, Stream, And Provider-Owned Response Paths

Exit criteria:

- Bridge response/stream paths have a clear ownership decision.
- Provider-owned response parsers such as Gemini Interactions and Bedrock are either migrated
  narrowly, classified as already sufficiently guarded, or split into follow-ons.
- No bridge/provider path becomes a generic dumping ground for provider response semantics.

## M4 — Integration, Docs, And Closeout

Exit criteria:

- Architecture/migration docs distinguish protocol-native parsing, legacy `ChatResponse`
  compatibility payloads, and high-level generated-output projection.
- Final targeted nextest/fmt gates pass for touched crates.
- Remaining broad parser migration is split or explicitly deferred with lossiness evidence.
- `WORKSTREAM.json` is updated to `closed`, `deferred`, or another truthful final status.
