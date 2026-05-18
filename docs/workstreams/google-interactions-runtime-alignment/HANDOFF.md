# Google Interactions Runtime Alignment - Handoff

Status: Active
Last updated: 2026-05-18

## Current State

The workstream is open. Siumai exposes `google.interactions(...)` through the Gemini/Google provider
surface, including model ids, agent names, typed provider options, metadata, builder construction,
and a provider-owned `GoogleInteractionsLanguageModel` handle.

GIR-020, GIR-030, GIR-040, and GIR-050 are implemented: stable `ChatRequest` values can now be prepared
into model-mode and agent-mode `/v1beta/interactions` request bodies, and completed Interactions
responses now parse back into stable `ChatResponse` values through provider-owned conversion code.
Non-stream runtime execution now posts to `/interactions`, parses terminal model responses, and polls
background agent responses through `GET /interactions/{id}` until a terminal status. Streaming still
fail-fasts by design until the SSE runtime lands.

## Active Task

- Task ID: GIR-060
- Owner: unassigned
- Files:
  - `siumai-provider-gemini/src/providers/gemini/interactions.rs`
  - `siumai-provider-gemini/src/providers/gemini/interactions/request.rs`
  - `siumai-provider-gemini/src/providers/gemini/interactions/response.rs`
  - `siumai-provider-gemini/src/providers/gemini/interactions/runtime.rs`
  - provider-local fixtures/tests
- Validation:
  - `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_stream`
- Status: READY
- Review: `review-workstream` before accepting implementation.
- Evidence: request conversion, completed-response parsing, non-stream POST, agent polling, missing
  id errors, timeout behavior, package tests, fmt, and clippy are already covered.

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
- Non-stream runtime execution now lives in `providers/gemini/interactions/runtime.rs`, sends
  `Api-Revision: 2026-05-20`, uses the shared HTTP execution layer/custom transport, and preserves
  request/response envelopes on `ChatResponse`.
- Agent polling honors `providerOptions.google.pollingTimeoutMs` and returns `TimeoutError` if the
  terminal status is not reached in time.

## Blockers

- No external blocker. GIR-060 needs stream event conversion against the AI SDK
  `build-google-interactions-stream-transform.ts` and event schema references.

## Next Recommended Action

Start GIR-060 with a stream conversion tracer bullet:

- POST `/interactions` in model mode with `stream: true`;
- convert Interactions SSE events into stable stream parts;
- preserve warnings, `interactionId`, service tier, sources, tool calls/results, and final
  `StreamEnd` metadata.
