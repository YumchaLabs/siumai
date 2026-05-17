# Stream Delta Lossless Boundary - TODO

Status: Closed
Last updated: 2026-05-17

## M0 - Scope And Evidence Freeze

- [x] SDL-010 [owner=planner] [deps=none] [scope=docs/workstreams/stream-delta-lossless-boundary]
  Goal: Freeze the stream delta lossless invariant, Vercel comparison, non-goals, and evidence
  anchors.
  Validation: `DESIGN.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, and
  `HANDOFF.md` exist and agree.
  Evidence: `docs/workstreams/stream-delta-lossless-boundary/DESIGN.md`
  Handoff: Planner owns this before implementation continues.

## M1 - Lossless Extractor Semantics

- [x] SDL-020 [owner=codex] [deps=SDL-010] [scope=siumai-protocol-openai/src/standards/openai/compat]
  Goal: Separate generated delta field extraction from control string extraction so generated
  content/reasoning paths cannot accidentally use trim-based or control-oriented filtering.
  Validation: `cargo nextest run -p siumai-protocol-openai --all-features`
  Evidence:
  `siumai-protocol-openai/src/standards/openai/compat/types.rs`,
  `siumai-protocol-openai/src/standards/openai/compat/streaming_tests.rs`
  Handoff: Preserve existing provider compatibility mappings while tightening helper names and
  tests.

## M2 - Shared Stream Factory Guard

- [x] SDL-030 [owner=codex] [deps=SDL-020] [scope=siumai-core/src/streaming]
  Goal: Prove shared stream infrastructure treats only true empty raw event data as framing and
  preserves whitespace-bearing JSON frames for provider conversion.
  Validation: `cargo nextest run -p siumai-core streaming::factory`
  Evidence: `siumai-core/src/streaming/factory.rs`
  Handoff: Keep raw SSE keepalive behavior intact.

## M3 - Provider Surface Regression Gate

- [x] SDL-040 [owner=codex] [deps=SDL-020] [scope=siumai-provider-openai-compatible]
  Goal: Prove the public OpenAI-compatible provider path keeps whitespace-only content and
  reasoning deltas.
  Validation: `cargo nextest run -p siumai-provider-openai-compatible --all-features`
  Evidence: chat and completion stream runtime tests covering lossless text/reasoning deltas and
  `[DONE]` handling.
  Handoff: No broad fixture churn required.

## M4 - Closeout

- [x] SDL-050 [owner=planner] [deps=SDL-030,SDL-040] [scope=docs/workstreams/stream-delta-lossless-boundary]
  Goal: Record final gates, update docs, and close or split remaining provider-wide audit work.
  Validation: `git diff --check` plus all focused gates in `EVIDENCE_AND_GATES.md`
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`
  Handoff: Commit with conventional commit once closeout checks pass.

## M5 - Post-Close Responses SSE Bridge Parity

- [x] SDL-060 [owner=codex] [deps=SDL-050] [scope=siumai-protocol-openai/src/standards/openai/responses_sse]
  Goal: Extend the same generated-delta field-presence invariant to OpenAI Responses SSE
  parser/serializer paths used by gateway and bridge routes.
  Validation: `cargo nextest run -p siumai-protocol-openai --all-features`
  Evidence:
  `siumai-protocol-openai/src/standards/openai/responses_sse/converter/convert.rs`,
  `siumai-protocol-openai/src/standards/openai/responses_sse/converter/serialize.rs`,
  `siumai-protocol-openai/src/standards/openai/responses_sse/tests.rs`
  Handoff: Keep control identifiers required, but treat generated `delta` fields as payload even
  when the payload is the empty string.
