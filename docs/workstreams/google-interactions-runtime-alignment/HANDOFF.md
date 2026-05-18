# Google Interactions Runtime Alignment - Handoff

Status: Active
Last updated: 2026-05-18

## Current State

The workstream is open. Siumai exposes `google.interactions(...)` through the Gemini/Google provider
surface, including model ids, agent names, typed provider options, metadata, builder construction,
and a fail-fast `GoogleInteractionsLanguageModel` handle.

GIR-020, GIR-030, and GIR-040 are implemented: stable `ChatRequest` values can now be prepared
into model-mode and agent-mode `/v1beta/interactions` request bodies, and completed Interactions
responses now parse back into stable `ChatResponse` values through provider-owned conversion code.
Runtime execution still fail-fasts by design until non-stream execution and polling land.

## Active Task

- Task ID: GIR-050
- Owner: unassigned
- Files:
  - `siumai-provider-gemini/src/providers/gemini/interactions.rs`
  - `siumai-provider-gemini/src/providers/gemini/interactions/request.rs`
  - `siumai-provider-gemini/src/providers/gemini/interactions/response.rs`
  - provider-local fixtures/tests
- Validation:
  - `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_non_stream`
- Status: READY
- Review: `review-workstream` before accepting implementation.
- Evidence: response fixture tests for completed Interactions responses, plus request conversion
  coverage already in place for model and agent bodies.

## Decisions Since Last Update

- This runtime lane was split out of `ai-sdk-provider-interface-convergence` because Interactions is
  not ordinary Gemini chat. It needs request conversion, polling, cancellation, stream reconnect,
  signature round-trip, and interaction-id compaction.
- The fail-fast handle remains the correct current behavior until a runtime slice lands with tests.
- Interactions code must stay provider-owned in `siumai-provider-gemini`.
- Model-mode request conversion lives in `providers/gemini/interactions/request.rs` so later
  response, polling, and streaming runtime slices do not turn `interactions.rs` into a monolith.
- Agent-mode request conversion follows the AI SDK semantics: send `agent` with `background: true`;
  warn/drop model-only `generation_config`, `tools`, structured output, and `imageConfig`.
- Completed-response parsing now lives in `providers/gemini/interactions/response.rs` and keeps the
  provider-owned `interactionId`/`serviceTier` metadata path ready for later polling and
  compaction.

## Blockers

- No external blocker. GIR-050 needs transport execution and polling helpers against the AI SDK
  `google-interactions-language-model.ts` reference.

## Next Recommended Action

Start GIR-050 with a non-stream execution tracer bullet:

- POST `/interactions` in model mode;
- poll `GET /interactions/{id}` until terminal when the API returns `in_progress` or
  `requires_action`;
- preserve the response parsing path from GIR-040 for the terminal payload.
