# Fearless AI SDK Contract Hardening - Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open. AICH-010, AICH-020, and AICH-030 are complete. The bootstrap captured the
12-gap audit, linked the local AI SDK reference under `repo-ref/ai`, and made changelog tracking
part of the task ledger. AICH-020 documented `StreamEnd.response.content` as final
replay/fallback. AICH-030 added an OpenAI Responses SSE fixture proving reasoning deltas and
terminal-only final visible text are both preserved.

## Active Task

- Task ID: AICH-040
- Owner: unassigned
- Files:
  - `siumai-core/src/text.rs`
  - `siumai-core/src/traits/chat.rs`
  - `siumai/src/text.rs`
  - `docs`
  - `CHANGELOG.md`
  - `siumai-core/CHANGELOG.md`
- Validation:
  - `cargo nextest run -p siumai-core stream_with_cancel --no-fail-fast`
  - existing OpenAI remote cancel test remains green
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
- AICH-020 established that `StreamEnd.response.content` is final replay/fallback, not a delta.
- AICH-030 established the OpenAI Responses SSE reasoning/text separation fixture at protocol level.

## Blockers

- None.

## Next Recommended Action

- Execute AICH-040 with `run-workstream-task`: document and test `stream_with_cancel` as the
  recommended cancelable text stream entry, with local and remote cancel semantics separated.
