# Google Interactions Runtime Alignment - Handoff

Status: Active
Last updated: 2026-05-18

## Current State

The workstream is open. Siumai already exposes `google.interactions(...)` through the Gemini/Google
provider surface, including model ids, agent names, typed provider options, metadata, builder
construction, and a fail-fast `GoogleInteractionsLanguageModel` handle. Runtime execution is not
implemented yet and should remain fail-fast until GIR-020 through GIR-050 land.

## Active Task

- Task ID: GIR-020
- Owner: unassigned
- Files:
  - `siumai-provider-gemini/src/providers/gemini/interactions.rs`
  - `siumai-provider-gemini/src/provider_options/gemini/mod.rs`
  - provider-local fixtures/tests
- Validation:
  - `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_request`
- Status: NEEDS_CONTEXT
- Review: `review-workstream` before accepting implementation.
- Evidence: request converter tests and captured request body assertions.

## Decisions Since Last Update

- This runtime lane was split out of `ai-sdk-provider-interface-convergence` because Interactions is
  not ordinary Gemini chat. It needs request conversion, polling, cancellation, stream reconnect,
  signature round-trip, and interaction-id compaction.
- The fail-fast handle remains the correct current behavior until a runtime slice lands with tests.
- Interactions code must stay provider-owned in `siumai-provider-gemini`.

## Blockers

- No external blocker. Implementation needs careful fixture selection from `repo-ref/ai`.

## Next Recommended Action

Start GIR-020 with a request-conversion tracer bullet:

- one model-mode request with system + user text;
- one provider-options request with `responseFormat`, `store`, and `mediaResolution`;
- one prior assistant reasoning/tool-call round-trip with `provider_metadata.google.signature`;
- one compaction case with `previousInteractionId`.
