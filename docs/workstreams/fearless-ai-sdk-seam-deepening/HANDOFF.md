# Fearless AI SDK Seam Deepening - Handoff

Status: Active
Last updated: 2026-05-27

## Current State

The workstream is open. AISD-010 created ADR-0009 and the task ledger for all architecture review
candidates. The lane follows the closed `fearless-ai-sdk-contract-hardening` workstream and keeps
Hajimi adapter changes out of scope.

## Completed Task

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

- Task ID: AISD-050
- Owner: codex
- Files:
  - `siumai-core/src/execution`
  - `siumai-core/src/traits/capabilities.rs`
  - `siumai-core/src/error/helpers.rs`
  - `siumai-spec/src/types/common.rs`
  - `CHANGELOG.md`
  - crate changelogs
- Validation:
  - `cargo fmt --check -p siumai-core -p siumai-spec`
  - `cargo nextest run -p siumai-core unsupported_capability --no-fail-fast`
  - `cargo nextest run -p siumai-core reject_if_unsupported --no-fail-fast`
- Status: IN_PROGRESS
- Review: not started
- Evidence: `TODO.md`, `EVIDENCE_AND_GATES.md`

## Decisions

- ADR-0009 chooses crate-owned seam deepening over pushing provider-specific replay behavior into
  `siumai-spec`.
- Work proceeds in vertical slices. Each slice must be independently testable and can delete obsolete
  code only when parity tests prove the deepened module owns the behavior.
- Public behavior should remain stable unless a task explicitly states a breaking cleanup.

## Next Recommended Action

Continue AISD-050 with `run-workstream-task`. Inspect existing unsupported-capability helpers and
executor guards before editing; the goal is one shared gate module so hard rejects, warnings, and
provider fallback behavior are explicit rather than repeated across executors.
