# Google Interactions Runtime Alignment - Handoff

Status: Active
Last updated: 2026-05-18

## Current State

The workstream is open. Siumai exposes `google.interactions(...)` through the Gemini/Google provider
surface, including model ids, agent names, typed provider options, metadata, builder construction,
and a provider-owned `GoogleInteractionsLanguageModel` handle.

GIR-020, GIR-030, GIR-040, GIR-050, GIR-060, and GIR-070 are implemented: stable `ChatRequest` values can now
be prepared into model-mode and agent-mode `/v1beta/interactions` request bodies, completed
Interactions responses parse back into stable `ChatResponse` values through provider-owned
conversion code, non-stream execution posts to `/interactions`, and background agent responses poll
through `GET /interactions/{id}` until a terminal status. Model-mode streaming now posts
`stream: true` to `/interactions` and converts Interactions SSE events into stable stream parts.
Agent-mode streaming now creates a background interaction, opens `GET /interactions/{id}?stream=true`,
reconnects with `last_event_id`, and sends best-effort `POST /interactions/{id}/cancel` when a
cancellable stream is aborted before completion.

## Active Task

- Task ID: GIR-080
- Owner: unassigned
- Files:
  - `siumai/tests/provider_public_path_parity_test.rs`
  - `siumai-provider-gemini/src/providers/gemini/interactions.rs`
  - public facade/builder path tests
- Validation:
  - `cargo nextest run -p siumai --features google google_interactions --test provider_public_path_parity_test --no-fail-fast`
- Status: READY
- Review: `review-workstream` before accepting implementation.
- Evidence: request conversion, completed-response parsing, non-stream POST, agent polling,
  model-mode stream POST, stream event conversion, agent stream reconnect/cancel, package tests,
  fmt, and clippy are already covered.

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
- Model-mode streaming now lives in `providers/gemini/interactions/stream.rs`, posts to
  `/interactions` with `stream: true`, emits stable typed stream parts for text, reasoning,
  function calls, built-in tool results, sources, images, usage, finish metadata, and preserves
  request/response envelopes.
- Agent-mode streaming now follows the AI SDK runtime shape: `POST /interactions` with
  `background: true`, then `GET /interactions/{id}?stream=true`, using the JSON `event_id` as
  `last_event_id` on reconnect. Cancel handles issue best-effort `POST /interactions/{id}/cancel`.
- The core custom transport seam now includes `execute_get_stream` so provider-owned GET SSE
  endpoints can be tested without network access.

## Blockers

- No external blocker. GIR-080 needs public-path parity assertions updated now that agent streaming
  is implemented instead of deferred.

## Next Recommended Action

Start GIR-080 with public facade parity:

- remove/narrow obsolete fail-fast Interactions public-path expectations;
- prove `Provider::google()`, `provider_ext::google`, and direct handle paths reach the implemented
  request/response/stream runtime;
- keep any truly unsupported subfeatures explicit with focused assertions.
