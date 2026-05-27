# Fearless AI SDK Seam Deepening - Handoff

Status: Closed
Last updated: 2026-05-27

## Final State

The workstream is closed. AISD-010 through AISD-080 are complete. ADR-0009 chose crate-owned seam
deepening; the implementation split OpenAI Responses stream state, diagnostics projection, tool
ownership, capability gates, usage snapshots, and core stream final assembly into named seams with
focused regressions. Hajimi adapter changes remain out of scope.

## Closeout Evidence

- `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai`
- `cargo nextest run -p siumai-spec -p siumai-core -p siumai-protocol-openai --no-fail-fast`
- `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses --no-fail-fast`
- `python -m json.tool docs\workstreams\fearless-ai-sdk-seam-deepening\WORKSTREAM.json`
- Closeout consistency script for `Status: Closed`, completed AISD ledger, and `INDEX.md` closed row
- `git diff --check -- CHANGELOG.md siumai-spec/CHANGELOG.md siumai-core/CHANGELOG.md siumai-protocol-openai/CHANGELOG.md docs/workstreams/fearless-ai-sdk-seam-deepening docs/workstreams/INDEX.md`

## Completed Task

- Task ID: AISD-070
- Result: DONE
- Summary: Core stream final response assembly now consumes an `AccumulatedStreamRecord` snapshot
  instead of reading `StreamProcessor`'s low-level text, reasoning, tool-call, order, and stream-part
  buffers directly. Terminal replay reconciliation, stable tool-call projection, repeated usage,
  and extra terminal parts remain covered by processor and provider-boundary tests.
- Validation:
  - `cargo check -p siumai-core`
  - `cargo fmt --check -p siumai-core`
  - `cargo nextest run -p siumai-core streaming::processor --no-fail-fast`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-fail-fast`

- Task ID: AISD-060
- Result: DONE
- Summary: Same-provider-call stream usage is now tracked by `UsageSnapshotLedger` in
  `siumai-spec::types::usage`. Core stream processing stores finish and terminal usage snapshots in
  the ledger, final response assembly reads from it, and OpenAI Responses serializer state uses the
  same ledger instead of a loose `latest_usage` field. `Usage::merge()` remains the explicit
  multi-call aggregation API.
- Validation:
  - `cargo check -p siumai-spec -p siumai-core -p siumai-protocol-openai --features siumai-protocol-openai/openai-standard,siumai-protocol-openai/openai-responses`
  - `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai`
  - `cargo nextest run -p siumai-spec usage --no-fail-fast`
  - `cargo nextest run -p siumai-core streaming::processor --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_event_converter_repeated_usage_keeps_latest_snapshot --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_serializer_state_uses_usage_snapshot_ledger --no-fail-fast`

- Task ID: AISD-050
- Result: DONE
- Summary: Unsupported capability choreography now lives in
  `siumai-core::execution::capability`. Hard family executors use named requirements for audio,
  embedding, files, image, and rerank; the shared gate owns reject, warning, and provider-fallback
  policy resolution; `ProviderCapabilities::reject_if_unsupported` is documented as a low-level
  compatibility/custom probe; and a boundary test prevents executors from rebuilding local feature
  strings or policies.
- Validation:
  - `cargo check -p siumai-core -p siumai-spec`
  - `cargo fmt --check -p siumai-core -p siumai-spec`
  - `cargo nextest run -p siumai-core unsupported_capability --no-fail-fast`
  - `cargo nextest run -p siumai-core reject_if_unsupported --no-fail-fast`
  - `cargo nextest run -p siumai-core core_hard_family_executors_use_named_capability_requirements --no-fail-fast`

- Task ID: AISD-040
- Result: DONE
- Summary: Provider-executed tool ownership now has a shared `ToolExecutionOwner` semantic
  contract. Spec prompt/UI/stream/tool views expose owner helpers while keeping AI SDK wire flags,
  core UI and stream projection route decisions through the owner contract, and OpenAI protocol
  adapters use the same owner helpers for Responses request/response/SSE conversion.
- Validation:
  - `cargo check -p siumai-spec -p siumai-core -p siumai-protocol-openai --features siumai-protocol-openai/openai-standard,siumai-protocol-openai/openai-responses`
  - `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai`
  - `cargo nextest run -p siumai-spec provider_executed --no-fail-fast`
  - `cargo nextest run -p siumai-core provider_executed --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses provider_executed --no-fail-fast`

- Task ID: AISD-030
- Result: DONE
- Summary: Public provider metadata and private diagnostics now have executable projections; core
  final content strips reserved raw/private provider metadata keys, and protocol has a diagnostics
  regression for OpenAI Responses replay raw items.
- Validation:
  - `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai`
  - `cargo nextest run -p siumai-spec private_diagnostics --no-fail-fast`
  - `cargo nextest run -p siumai-spec provider_metadata --no-fail-fast`
  - `cargo nextest run -p siumai-core provider_metadata --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses provider_metadata --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses diagnostics --no-fail-fast`

- Task ID: AISD-020
- Result: DONE
- Summary: OpenAI Responses SSE converter state now has named owners for terminal buffering, replay
  hints, reasoning lifecycle, provider/custom tool ownership, and serializer allocation rules.
- Validation:
  - `cargo fmt --check -p siumai-protocol-openai`
  - `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_sse --no-fail-fast`

## Decisions

- ADR-0009 chooses crate-owned seam deepening over pushing provider-specific replay behavior into
  `siumai-spec`.
- Work proceeds in vertical slices. Each slice must be independently testable and can delete obsolete
  code only when parity tests prove the deepened module owns the behavior.
- Public behavior should remain stable unless a task explicitly states a breaking cleanup.

## Follow-Ons

- No Siumai follow-up is required for this seam-deepening scope.
- Provider-specific quirks should be opened as provider-scoped workstreams only if a concrete
  behavior gap appears.
- Hajimi adapter changes remain out of scope and should be handled after the Siumai side lands.
