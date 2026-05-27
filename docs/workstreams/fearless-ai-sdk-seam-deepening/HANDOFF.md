# Fearless AI SDK Seam Deepening - Handoff

Status: Active
Last updated: 2026-05-27

## Current State

The workstream is open. AISD-010 created ADR-0009 and the task ledger for all architecture review
candidates. The lane follows the closed `fearless-ai-sdk-contract-hardening` workstream and keeps
Hajimi adapter changes out of scope.

## Completed Task

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

## Active Task

- Task ID: AISD-060
- Owner: codex
- Files:
  - `siumai-spec/src/types/usage.rs`
  - `siumai-core/src/streaming`
  - `siumai-protocol-openai/src/standards/openai/responses_sse`
  - `CHANGELOG.md`
  - crate changelogs
- Validation:
  - `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai`
  - `cargo nextest run -p siumai-spec usage --no-fail-fast`
  - `cargo nextest run -p siumai-core streaming::processor --no-fail-fast`
  - OpenAI Responses repeated-usage fixture
- Status: READY
- Review: not started
- Evidence: `TODO.md`, `EVIDENCE_AND_GATES.md`

## Decisions

- ADR-0009 chooses crate-owned seam deepening over pushing provider-specific replay behavior into
  `siumai-spec`.
- Work proceeds in vertical slices. Each slice must be independently testable and can delete obsolete
  code only when parity tests prove the deepened module owns the behavior.
- Public behavior should remain stable unless a task explicitly states a breaking cleanup.

## Next Recommended Action

Continue AISD-060 with `run-workstream-task`. Inspect usage snapshot handling in spec, core stream
processor, and OpenAI Responses SSE before editing; the goal is a named ledger module where
same-provider-call stream usage snapshots replace each other, while explicit multi-call aggregation
continues to use orchestration-level `Usage::merge()`.
