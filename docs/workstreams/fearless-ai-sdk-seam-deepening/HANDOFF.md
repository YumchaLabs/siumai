# Fearless AI SDK Seam Deepening - Handoff

Status: Active
Last updated: 2026-05-27

## Current State

The workstream is open. AISD-010 created ADR-0009 and the task ledger for all architecture review
candidates. The lane follows the closed `fearless-ai-sdk-contract-hardening` workstream and keeps
Hajimi adapter changes out of scope.

## Completed Task

- Task ID: AISD-020
- Result: DONE
- Summary: OpenAI Responses SSE converter state now has named owners for terminal buffering, replay
  hints, reasoning lifecycle, provider/custom tool ownership, and serializer allocation rules.
- Validation:
  - `cargo fmt --check -p siumai-protocol-openai`
  - `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_sse --no-fail-fast`

## Active Task

- Task ID: AISD-030
- Owner: codex
- Files:
  - `siumai-spec/src/types`
  - `siumai-core/src/streaming`
  - `siumai-protocol-openai/src`
  - `CHANGELOG.md`
  - crate changelogs
- Validation:
  - `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai`
  - `cargo nextest run -p siumai-spec private_diagnostics --no-fail-fast`
  - `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses diagnostics --no-fail-fast`
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

Continue AISD-030 with `run-workstream-task`. Inspect the existing provider metadata, diagnostics,
raw stream part, and custom event paths before editing; the goal is one executable projection seam
for public metadata versus private diagnostics.
