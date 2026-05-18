# Google Interactions Runtime Alignment - Handoff

Status: Active
Last updated: 2026-05-18

## Current State

The workstream is open. Siumai exposes `google.interactions(...)` through the Gemini/Google provider
surface, including model ids, agent names, typed provider options, metadata, builder construction,
and a fail-fast `GoogleInteractionsLanguageModel` handle.

GIR-020 is implemented: stable `ChatRequest` values can now be prepared into model-mode
`/v1beta/interactions` request bodies through provider-owned conversion code, including
model-mode tools/tool choice. Runtime execution still fail-fasts by design until response parsing
and non-stream execution land.

## Active Task

- Task ID: GIR-030
- Owner: unassigned
- Files:
  - `siumai-provider-gemini/src/providers/gemini/interactions.rs`
  - `siumai-provider-gemini/src/providers/gemini/interactions/request.rs`
  - provider-local fixtures/tests
- Validation:
  - `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_agent`
- Status: READY
- Review: `review-workstream` before accepting implementation.
- Evidence: agent-mode request body assertions and warning decisions.

## Decisions Since Last Update

- This runtime lane was split out of `ai-sdk-provider-interface-convergence` because Interactions is
  not ordinary Gemini chat. It needs request conversion, polling, cancellation, stream reconnect,
  signature round-trip, and interaction-id compaction.
- The fail-fast handle remains the correct current behavior until a runtime slice lands with tests.
- Interactions code must stay provider-owned in `siumai-provider-gemini`.
- Model-mode request conversion lives in `providers/gemini/interactions/request.rs` so later
  response, polling, and streaming runtime slices do not turn `interactions.rs` into a monolith.

## Blockers

- No external blocker. GIR-030 needs AI SDK agent request semantics checked before widening
  supported agent options.

## Next Recommended Action

Start GIR-030 with an agent-mode request-conversion tracer bullet:

- one agent request with user text and `agent` set instead of `model`;
- warning or rejection coverage for unsupported generation config and tools;
- provider-option handling for `store`, `previousInteractionId`, and `responseFormat` where the AI
  SDK reference permits it.
