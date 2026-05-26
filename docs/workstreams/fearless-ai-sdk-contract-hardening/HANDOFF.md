# Fearless AI SDK Contract Hardening - Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open. The bootstrap captured the 12-gap audit, linked the local AI SDK reference
under `repo-ref/ai`, and made changelog tracking part of the task ledger.

No runtime code has been changed yet.

## Active Task

- Task ID: AICH-020
- Owner: unassigned
- Files:
  - `siumai-spec/src/types/streaming.rs`
  - `siumai-spec/src/types/chat/response.rs`
  - `siumai-core/src/streaming`
  - `CHANGELOG.md`
  - `siumai-spec/CHANGELOG.md`
  - `siumai-core/CHANGELOG.md`
- Validation:
  - `cargo nextest run -p siumai-spec stream --no-fail-fast`
  - `cargo nextest run -p siumai-core streaming::processor --no-fail-fast`
- Status: NEEDS_CONTEXT
- Review: not started
- Evidence: `TODO.md`, `EVIDENCE_AND_GATES.md`

## Decisions Since Last Update

- Opened a new workstream instead of reopening closed stream/metadata lanes because the current
  issue spans stream replay, diagnostics, tools, usage, cancellation, error safety, and capability
  contracts.
- Kept Hajimi adapter changes out of scope. Hajimi should wait for Siumai contract fixes before
  adapting.
- First implementation task is stream replay semantics because it affects downstream loss/duplicate
  behavior and provides the cleanest proof slice.

## Blockers

- None.

## Next Recommended Action

- Execute AICH-020 with `run-workstream-task`: define `StreamEnd.response.content` replay/fallback
  semantics in spec docs, add focused processor/stream tests, and update root plus crate changelogs.
