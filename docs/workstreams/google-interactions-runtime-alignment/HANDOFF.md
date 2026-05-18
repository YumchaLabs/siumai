# Google Interactions Runtime Alignment - Handoff

Status: Active
Last updated: 2026-05-18

## Current State

The workstream is open. Siumai exposes `google.interactions(...)` through the Gemini/Google provider
surface, including model ids, agent names, typed provider options, metadata, builder construction,
and a fail-fast `GoogleInteractionsLanguageModel` handle.

GIR-020 and GIR-030 are implemented: stable `ChatRequest` values can now be prepared into
model-mode and agent-mode `/v1beta/interactions` request bodies through provider-owned conversion
code. Runtime execution still fail-fasts by design until response parsing and non-stream execution
land.

## Active Task

- Task ID: GIR-040
- Owner: unassigned
- Files:
  - `siumai-provider-gemini/src/providers/gemini/interactions.rs`
  - `siumai-provider-gemini/src/providers/gemini/interactions/response.rs`
  - provider-local fixtures/tests
- Validation:
  - `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_response`
- Status: READY
- Review: `review-workstream` before accepting implementation.
- Evidence: response fixture tests for completed Interactions responses.

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

## Blockers

- No external blocker. GIR-040 needs response fixtures and output parsing against the AI SDK
  `parse-google-interactions-outputs.ts` reference.

## Next Recommended Action

Start GIR-040 with a response-parsing tracer bullet:

- one completed response with text output, usage, finish reason, service tier, and interaction id;
- one response with reasoning signature and function call;
- one response with function result / media block coverage.
