# Google Interactions Runtime Alignment - TODO

Status: Active
Last updated: 2026-05-18

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## M0 - Scope And Evidence Freeze

- [x] GIR-010 [owner=planner] [deps=none] [scope=docs/workstreams/google-interactions-runtime-alignment]
  Goal: Freeze problem, target state, non-goals, and evidence anchors for Google Interactions
  runtime support.
  Validation: `DESIGN.md`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`,
  `WORKSTREAM.json`, and `HANDOFF.md` exist and agree.
  Evidence: `docs/workstreams/google-interactions-runtime-alignment/DESIGN.md`
  Handoff: Opened from AIPC-080 after the package surface was exposed but runtime execution was
  intentionally deferred.

## M1 - Request Conversion Proof

- [x] GIR-020 [owner=codex] [deps=GIR-010] [scope=siumai-provider-gemini,fixtures]
  Goal: Convert `ChatRequest` plus Google Interactions provider options into the
  `/v1beta/interactions` request shape for model-mode calls.
  Validation: `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_request`
  Review: `review-workstream` before accepting completion.
  Evidence: provider-local converter tests and captured request body assertions passed on
  2026-05-18.
  Handoff: Implemented as provider-owned request conversion under
  `providers/gemini/interactions/request.rs`. Covered system instruction precedence, response
  format entries, files, tool calls/results, reasoning signatures, `previousInteractionId`,
  `store`, `mediaResolution`, model-mode tools/tool choice, and deprecated `imageConfig`
  fallback. Agent conversion remains deferred to GIR-030.

- [x] GIR-030 [owner=codex] [deps=GIR-020] [scope=siumai-provider-gemini,fixtures]
  Goal: Add agent-mode request conversion and warning behavior.
  Validation: `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_agent`
  Review: `review-workstream` before accepting completion.
  Evidence: provider-local tests for agent request bodies and warning decisions passed on
  2026-05-18.
  Handoff: Agent requests send `agent` and `background: true`, omit `model` and `stream`, preserve
  `agent_config`, and warn/drop unsupported tools, generation config, structured output, and
  `imageConfig`.

## M2 - Response And Polling Runtime

- [x] GIR-040 [owner=codex] [deps=GIR-020] [scope=siumai-provider-gemini]
  Goal: Parse completed Interactions responses into stable `ChatResponse` content, usage, finish
  reason, and `provider_metadata.google`.
  Validation: `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_response`
  Review: `review-workstream` before accepting completion.
  Evidence: provider-owned response fixture tests passed for completed text, usage, finish reason,
  reasoning signatures, function calls, images, built-in tool calls/results, and source
  extraction.
  Handoff: `providers/gemini/interactions/response.rs` now parses completed Interactions payloads
  into stable `ChatResponse` values and stamps `interactionId` on output parts that participate in
  later compaction.

- [x] GIR-050 [owner=codex] [deps=GIR-040] [scope=siumai-provider-gemini]
  Goal: Implement non-stream model-mode execution and background polling for terminal interactions.
  Validation: `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_non_stream`
  Review: `review-workstream` before accepting completion.
  Evidence: capture-transport tests passed for model POST, agent background polling, missing-id
  error behavior, and `pollingTimeoutMs` timeout behavior on 2026-05-18.
  Handoff: Non-stream Interactions now executes through provider-owned runtime helpers and the
  shared HTTP execution layer. Streaming remains deferred to GIR-060/GIR-070.

## M3 - Streaming Runtime

- [x] GIR-060 [owner=codex] [deps=GIR-040] [scope=siumai-provider-gemini]
  Goal: Convert Interactions SSE events into stable stream parts and provider metadata.
  Validation: `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_stream`
  Review: `review-workstream` before accepting completion.
  Evidence: provider-local capture-transport stream tests passed on 2026-05-18 for model-mode POST,
  typed stream start/metadata, text deltas/end metadata, reasoning signatures, function call
  argument deltas, built-in tool results, deduped sources, image file parts, usage, finish reason,
  request/response envelopes, and service tier metadata.
  Handoff: Implemented in provider-owned `providers/gemini/interactions/stream.rs`; ordinary Gemini
  `:generateContent` streaming remains separate. Agent stream reconnect/cancel remains GIR-070.

- [~] GIR-070 [owner=unassigned] [deps=GIR-060] [scope=siumai-provider-gemini]
  Goal: Add stream reconnect and cancel-on-abort behavior.
  Validation: `cargo nextest run -p siumai-provider-gemini --all-features google_interactions_stream_reconnect`
  Review: `review-workstream` before accepting completion.
  Evidence: no-network tests for `last_event_id`, retry budget, and `POST /cancel`.
  Handoff: This is required before claiming production-ready streaming.

## M4 - Public Path And Closeout

- [ ] GIR-080 [owner=unassigned] [deps=GIR-050,GIR-060] [scope=siumai,siumai-provider-gemini]
  Goal: Replace the current fail-fast public-path guard with request/response/stream parity tests
  for implemented Interactions paths.
  Validation: `cargo nextest run -p siumai --features google google_interactions --test provider_public_path_parity_test --no-fail-fast`
  Review: `review-workstream` before accepting completion.
  Evidence: public-path tests across `Provider::google()`, `provider_ext::google`, and direct handle
  construction.
  Handoff: Keep unsupported Interactions subfeatures explicit if not implemented.

- [ ] GIR-090 [owner=planner] [deps=GIR-080] [scope=docs/workstreams/google-interactions-runtime-alignment]
  Goal: Close the lane or split remaining Interactions runtime gaps into narrower follow-ons.
  Validation: `verify-rust-workstream` records final gate evidence.
  Review: `review-workstream` has no blocking findings.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`
  Handoff: Summarize any intentionally deferred agent/stream cases.
