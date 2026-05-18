# Google Interactions Runtime Alignment - Milestones

Status: Completed
Last updated: 2026-05-18

## M0 - Scope And Evidence Freeze

Exit criteria:

- Runtime problem and target state are explicit.
- AI SDK reference files and AIPC parent lane are linked.
- Non-goals prevent ordinary Gemini runtime churn.

Primary evidence:

- `docs/workstreams/google-interactions-runtime-alignment/DESIGN.md`
- `docs/workstreams/google-interactions-runtime-alignment/TODO.md`

Status: completed

## M1 - Request Conversion Proof

Exit criteria:

- Model-mode Interactions request bodies can be produced from stable `ChatRequest`.
- Agent-mode request bodies and warning semantics are covered.
- Provider options remain under `provider_options.google`.

Primary gates:

```bash
cargo nextest run -p siumai-provider-gemini --all-features google_interactions_request
cargo nextest run -p siumai-provider-gemini --all-features google_interactions_agent
```

Status: completed

Progress:

- GIR-020 completed on 2026-05-18 for model-mode request conversion.
- GIR-030 completed on 2026-05-18 for agent-mode request conversion and warning behavior.

## M2 - Response And Polling Runtime

Exit criteria:

- Completed Interactions responses parse into stable response content and metadata.
- Non-stream execution posts to `/interactions` and polls when needed.
- Cancellation/timeout behavior is explicit at helper level.

Primary gates:

```bash
cargo nextest run -p siumai-provider-gemini --all-features google_interactions_response
cargo nextest run -p siumai-provider-gemini --all-features google_interactions_non_stream
```

Status: completed

Progress:

- GIR-040 completed on 2026-05-18 for completed-response parsing, usage mapping, finish reasons,
  provider metadata, images, and source extraction.
- GIR-050 completed on 2026-05-18 for non-stream `/interactions` execution, agent background
  polling, typed polling timeout behavior, and shared HTTP execution wiring.

## M3 - Streaming Runtime

Exit criteria:

- Interactions SSE events convert into stable stream parts.
- Reconnect with `last_event_id` is covered.
- Abort/cancel behavior is covered.

Primary gates:

```bash
cargo nextest run -p siumai-provider-gemini --all-features google_interactions_stream
cargo nextest run -p siumai-provider-gemini --all-features google_interactions_stream_reconnect
```

Status: completed

Progress:

- GIR-060 completed on 2026-05-18 for model-mode Interactions SSE conversion, typed stream parts,
  provider metadata, usage/finish metadata, and model-mode streaming POST wiring.
- GIR-070 completed on 2026-05-18 for agent-mode background stream creation, resumable GET SSE
  with JSON `event_id` as `last_event_id`, empty stream retry budget, and best-effort remote
  cancel on abort handles.

## M4 - Public Path And Closeout

Exit criteria:

- Public facade tests prove the implemented Interactions paths.
- The current fail-fast guard is either removed for implemented paths or narrowed to truly deferred
  subfeatures.
- Final gate evidence is recorded.

Primary gates:

```bash
cargo nextest run -p siumai --features google google_interactions --test provider_public_path_parity_test --no-fail-fast
cargo nextest run -p siumai --features google --test provider_public_path_parity_test --no-fail-fast
cargo nextest run -p siumai-provider-gemini --all-features --no-fail-fast
```

Status: completed

Progress:

- GIR-080 completed on 2026-05-18 for public facade parity. The old fail-fast guard was replaced
  with no-network tests proving `Provider::google()`, `provider_ext::google`, and direct
  `GoogleInteractionsLanguageModel` paths reach implemented model non-stream, model stream, and
  agent background GET-stream Interactions runtime.
- GIR-090 completed on 2026-05-18. The lane closes without splitting a new Interactions-specific
  follow-on because the target runtime, stream reconnect/cancel, public facade parity, and package
  gates are covered.
