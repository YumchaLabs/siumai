# Fearless AI SDK Contract Hardening - Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open. AICH-010, AICH-020, AICH-030, and AICH-040 are complete. The bootstrap
captured the 12-gap audit, linked the local AI SDK reference under `repo-ref/ai`, and made changelog
tracking part of the task ledger. AICH-020 documented `StreamEnd.response.content` as final
replay/fallback. AICH-030 added an OpenAI Responses SSE fixture proving reasoning deltas and
terminal-only final visible text are both preserved. AICH-040 documented `stream_with_cancel` as the
recommended cancelable stream entry and separated default local cancellation from provider-specific
remote abort.

## Active Task

- Task ID: AICH-050
- Owner: unassigned
- Files:
  - `siumai-spec/src/types/common.rs`
  - `siumai-spec/src/types/chat/content`
  - `siumai-spec/src/types/streaming.rs`
  - `siumai-core/src/streaming`
  - `siumai-protocol-openai`
  - `CHANGELOG.md`
  - `siumai-spec/CHANGELOG.md`
- Validation:
  - `cargo nextest run -p siumai-spec metadata --no-fail-fast`
  - targeted protocol raw-event tests
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
- AICH-040 established that `stream_with_cancel` is the recommended cancelable entry; default
  behavior guarantees local stream-consumption cancellation, while remote abort is provider-specific.

## Blockers

- None.

## Next Recommended Action

- Execute AICH-050 with `run-workstream-task`: define raw/private diagnostics treatment for provider
  metadata, `ResponseMetadata.headers/body`, `ChatStreamPart::Raw`, and `ChatStreamEvent::Custom`.
