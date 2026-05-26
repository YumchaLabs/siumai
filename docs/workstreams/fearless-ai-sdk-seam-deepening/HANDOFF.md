# Fearless AI SDK Seam Deepening - Handoff

Status: Active
Last updated: 2026-05-27

## Current State

The workstream is open. AISD-010 created ADR-0009 and the task ledger for all architecture review
candidates. The lane follows the closed `fearless-ai-sdk-contract-hardening` workstream and keeps
Hajimi adapter changes out of scope.

## Active Task

- Task ID: AISD-020
- Owner: codex
- Files:
  - `siumai-protocol-openai/src/standards/openai/responses_sse/converter`
  - `CHANGELOG.md`
  - `siumai-protocol-openai/CHANGELOG.md`
- Validation:
  - `cargo fmt --check -p siumai-protocol-openai`
  - `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_sse --no-fail-fast`
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

Continue AISD-020 with `run-workstream-task`. The terminal buffering, replay helper, and reasoning
lifecycle slices are done and verified; next deepen provider tool state or serializer state before
marking AISD-020 complete.
