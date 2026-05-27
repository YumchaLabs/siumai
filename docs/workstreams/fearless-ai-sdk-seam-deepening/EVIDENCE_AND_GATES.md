# Fearless AI SDK Seam Deepening - Evidence And Gates

Status: Active
Last updated: 2026-05-27

## Baseline Evidence

| Date | Task | Command / Evidence | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-27 | AISD-010 | Architecture review using `improve-codebase-architecture` plus three read-only Explore agents for stream, diagnostics, and tool/usage/capability slices. | Pass | Found six deepening candidates: OpenAI Responses stream state, diagnostics projection, tool ownership, capability gates, usage ledger, and core stream assembly. |
| 2026-05-27 | AISD-010 | `python -m json.tool docs\workstreams\fearless-ai-sdk-seam-deepening\WORKSTREAM.json` | Pass | Workstream metadata parses and records AISD-020 as the next executable task. |
| 2026-05-27 | AISD-010 | `git diff --check -- docs\adr docs\workstreams\fearless-ai-sdk-seam-deepening docs\workstreams\INDEX.md` | Pass | Diff check reported only expected LF-to-CRLF working-copy warnings. |
| 2026-05-27 | AISD-020 | `cargo fmt --check -p siumai-protocol-openai` | Pass | Terminal buffering refactor is formatted after extracting `TerminalEventBuffer`. |
| 2026-05-27 | AISD-020 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_sse --no-fail-fast` | Pass | 71 Responses SSE tests passed, covering terminal buffering, StreamEnd replay, reasoning/text separation, raw replay serialization, provider-executed tool events, and repeated usage behavior. |
| 2026-05-27 | AISD-020 | `cargo fmt --check -p siumai-protocol-openai` | Pass | Replay hint refactor is formatted after extracting `converter::replay`. |
| 2026-05-27 | AISD-020 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_sse --no-fail-fast` | Pass | 71 Responses SSE tests passed after moving PartWithReplay attach/apply rules behind `converter::replay`. |
| 2026-05-27 | AISD-020 | `cargo fmt --check -p siumai-protocol-openai` | Pass | Reasoning lifecycle refactor is formatted after extracting `ReasoningLifecycleState`. |
| 2026-05-27 | AISD-020 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_sse --no-fail-fast` | Pass | 71 Responses SSE tests passed after moving reasoning lifecycle state behind the reasoning module. |
| 2026-05-27 | AISD-020 | `cargo fmt --check -p siumai-protocol-openai` | Pass | Provider/custom tool state refactor is formatted after extracting `ProviderToolState` and `CustomToolState`. |
| 2026-05-27 | AISD-020 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_sse --no-fail-fast` | Pass | 71 Responses SSE tests passed after moving provider-defined tool names, hosted tool-search pairing, and custom tool de-duplication behind named state modules. |
| 2026-05-27 | AISD-020 | `cargo fmt --check -p siumai-protocol-openai` | Pass | Serializer state refactor is formatted after extracting `OpenAiResponsesSerializeStateCell` and state-owned allocation helpers. |
| 2026-05-27 | AISD-020 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_sse --no-fail-fast` | Pass | 71 Responses SSE tests passed after moving sequence, output-index, provider-tool-index, reasoning-item, function-call, and message-item allocation rules into `OpenAiResponsesSerializeState`. |
| 2026-05-27 | AISD-030 | `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai` | Pass | Spec/core/protocol formatting is clean after diagnostics projection changes. |
| 2026-05-27 | AISD-030 | `cargo nextest run -p siumai-spec private_diagnostics --no-fail-fast` | Pass | 9 private diagnostics tests passed for response metadata, stream replay, custom raw/private events, and chat response projection. |
| 2026-05-27 | AISD-030 | `cargo nextest run -p siumai-spec provider_metadata --no-fail-fast` | Pass | 13 provider metadata tests passed, including recursive public projection and merge-time stripping. |
| 2026-05-27 | AISD-030 | `cargo nextest run -p siumai-core provider_metadata --no-fail-fast` | Pass | 12 core provider metadata tests passed, including final content projection stripping raw/private keys. |
| 2026-05-27 | AISD-030 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses provider_metadata --no-fail-fast` | Pass | 23 protocol provider metadata tests passed after the shared public projection change. |
| 2026-05-27 | AISD-030 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses diagnostics --no-fail-fast` | Pass | 1 Responses SSE diagnostics projection test passed. |
| 2026-05-27 | AISD-040 | `cargo check -p siumai-spec -p siumai-core -p siumai-protocol-openai --features siumai-protocol-openai/openai-standard,siumai-protocol-openai/openai-responses` | Pass | Spec/core/protocol compile with the shared `ToolExecutionOwner` contract and OpenAI Responses feature surface. |
| 2026-05-27 | AISD-040 | `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai` | Pass | Spec/core/protocol formatting is clean after provider-executed ownership refactor. |
| 2026-05-27 | AISD-040 | `cargo nextest run -p siumai-spec provider_executed --no-fail-fast` | Pass | 3 spec provider-executed tests passed, including `ToolExecutionOwner` wire-flag conversion and prompt validation. |
| 2026-05-27 | AISD-040 | `cargo nextest run -p siumai-core provider_executed --no-fail-fast` | Pass | 2 core UI provider-executed conversion tests passed. |
| 2026-05-27 | AISD-040 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses provider_executed --no-fail-fast` | Pass | 6 protocol provider-executed tests passed across Responses SSE conversion, request/response transformers, and JSON response encoding. |
| 2026-05-27 | AISD-050 | `cargo check -p siumai-core -p siumai-spec` | Pass | Core/spec compile after moving executor hard guards to `execution::capability`. |
| 2026-05-27 | AISD-050 | `cargo fmt --check -p siumai-core -p siumai-spec` | Pass | Core/spec formatting is clean after named capability requirement extraction. |
| 2026-05-27 | AISD-050 | `cargo nextest run -p siumai-core unsupported_capability --no-fail-fast` | Pass | 2 unsupported-capability policy resolver tests passed. |
| 2026-05-27 | AISD-050 | `cargo nextest run -p siumai-core reject_if_unsupported --no-fail-fast` | Pass | 4 low-level and executor-gate reject tests passed. |
| 2026-05-27 | AISD-050 | `cargo nextest run -p siumai-core core_hard_family_executors_use_named_capability_requirements --no-fail-fast` | Pass | Boundary test proves audio, embedding, files, image, and rerank executors cross the named requirement seam instead of rebuilding policies locally. |
| 2026-05-27 | AISD-060 | `cargo check -p siumai-spec -p siumai-core -p siumai-protocol-openai --features siumai-protocol-openai/openai-standard,siumai-protocol-openai/openai-responses` | Pass | Spec/core/protocol compile after introducing `UsageSnapshotLedger`. |
| 2026-05-27 | AISD-060 | `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai` | Pass | Spec/core/protocol formatting is clean after usage ledger extraction. |
| 2026-05-27 | AISD-060 | `cargo nextest run -p siumai-spec usage --no-fail-fast` | Pass | 10 usage tests passed, including ledger replacement and merge aggregation tests. |
| 2026-05-27 | AISD-060 | `cargo nextest run -p siumai-core streaming::processor --no-fail-fast` | Pass | 17 stream processor tests passed, including the usage ledger boundary test. |
| 2026-05-27 | AISD-060 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_event_converter_repeated_usage_keeps_latest_snapshot --no-fail-fast` | Pass | Responses repeated-usage fixture preserved the latest cumulative snapshot through stream processing. |
| 2026-05-27 | AISD-060 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_serializer_state_uses_usage_snapshot_ledger --no-fail-fast` | Pass | Responses serializer state boundary test proves `UsageSnapshotLedger` owns latest usage replacement. |
| 2026-05-27 | AISD-070 | `cargo check -p siumai-core` | Pass | Core compiles after extracting `AccumulatedStreamRecord`. |
| 2026-05-27 | AISD-070 | `cargo fmt --check -p siumai-core` | Pass | Core formatting is clean after stream response assembly split. |
| 2026-05-27 | AISD-070 | `cargo nextest run -p siumai-core streaming::processor --no-fail-fast` | Pass | 18 stream processor tests passed, including the accumulated-record boundary regression. |
| 2026-05-27 | AISD-070 | `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-fail-fast` | Pass | 48 core provider boundary tests passed after the stream assembly split. |

## Required Gates

### Planning Gate

```text
python -m json.tool docs/workstreams/fearless-ai-sdk-seam-deepening/WORKSTREAM.json
git diff --check -- docs/adr docs/workstreams/fearless-ai-sdk-seam-deepening docs/workstreams/INDEX.md
```

### Slice Gates

- Use crate-scoped `cargo fmt --check -p <crate>` for touched crates.
- Use targeted `cargo nextest run -p <crate> <filter> --no-fail-fast` during iteration.
- Use package gates before marking a task done when the slice changes shared behavior.

### Closeout Gate

```text
cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai
cargo nextest run -p siumai-spec -p siumai-core -p siumai-protocol-openai --no-fail-fast
cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses --no-fail-fast
```

Full workspace `cargo fmt --check` may still fail on Windows with path-length error 206. Record that
failure if it happens and use crate-scoped formatting for touched crates.

## Review Gate

Run `review-workstream` before accepting major slices and before closeout. Review must check:

- ADR-0009 alignment.
- No drift from ADR-0001, ADR-0006, or ADR-0008.
- Changelog coverage for behavior-visible or public contract changes.
- No unverified deletion of compatibility code.
