# Google Interactions Runtime Alignment - Handoff

Status: Completed
Last updated: 2026-05-18

## Current State

The workstream is closed. Siumai exposes `google.interactions(...)` through the Gemini/Google provider
surface, including model ids, agent names, typed provider options, metadata, builder construction,
and a provider-owned `GoogleInteractionsLanguageModel` handle.

GIR-020 through GIR-080 are implemented: stable `ChatRequest` values can now be prepared into
model-mode and agent-mode `/v1beta/interactions` request bodies, completed Interactions responses
parse back into stable `ChatResponse` values through provider-owned conversion code, non-stream
execution posts to `/interactions`, and background agent responses poll through
`GET /interactions/{id}` until a terminal status. Model-mode streaming now posts `stream: true` to
`/interactions` and converts Interactions SSE events into stable stream parts. Agent-mode streaming
now creates a background interaction, opens `GET /interactions/{id}?stream=true`, reconnects with
`last_event_id`, and sends best-effort `POST /interactions/{id}/cancel` when a cancellable stream is
aborted before completion. Public facade tests now prove `Provider::google()`,
`provider_ext::google`, and direct handle construction reach the implemented Interactions runtime
instead of fail-fast behavior.

## Closeout

- Task ID: GIR-090
- Status: DONE
- Decision: close the lane without splitting an Interactions-specific follow-on.
- Evidence: request conversion, completed-response parsing, non-stream POST, agent polling,
  model-mode stream POST, stream event conversion, agent stream reconnect/cancel, public facade
  parity, package tests, fmt, and clippy are covered in `EVIDENCE_AND_GATES.md`.

## Decisions Since Last Update

- This runtime lane was split out of `ai-sdk-provider-interface-convergence` because Interactions is
  not ordinary Gemini chat. It needs request conversion, polling, cancellation, stream reconnect,
  signature round-trip, and interaction-id compaction.
- The temporary fail-fast boundary was removed for implemented chat paths after equivalent runtime
  and public-path tests landed.
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
- Public facade parity now covers model non-stream, model stream, and agent background GET stream
  paths across `Provider::google()`, `provider_ext::google`, and direct
  `GoogleInteractionsLanguageModel` construction.

## Blockers

- None.

## Next Recommended Action

Return to the parent AI SDK provider-interface convergence program and choose the next provider or
shared seam by fresh inventory, not by continuing this closed lane.
