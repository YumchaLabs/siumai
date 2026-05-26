# Fearless AI SDK Contract Hardening - Handoff

Status: Active
Last updated: 2026-05-26

## Current State

The workstream is open. AICH-010 through AICH-070 are complete. The bootstrap captured the 12-gap
audit, linked the local AI SDK reference under `repo-ref/ai`, and made changelog tracking part of
the task ledger. AICH-020 documented `StreamEnd.response.content` as final replay/fallback.
AICH-030 added an OpenAI Responses SSE fixture proving reasoning deltas and terminal-only final
visible text are both preserved. AICH-040 documented `stream_with_cancel` as the recommended
cancelable stream entry and separated default local cancellation from provider-specific remote
abort. AICH-050 defined public provider metadata versus private raw diagnostics boundaries.
AICH-060 split safe user-facing error messages from raw diagnostics. AICH-070 added explicit tool
validation/failure helpers, documented provider-executed ownership, and locked `ToolInputStart`
stable projection away from provider replay indexes.

## Active Task

- Task ID: AICH-080
- Owner: unassigned
- Files:
  - `siumai-spec/src/types/usage.rs`
  - `siumai-core/src/streaming/processor.rs`
  - `siumai-protocol-openai`
  - `CHANGELOG.md`
  - `siumai-spec/CHANGELOG.md`
  - `siumai-core/CHANGELOG.md`
- Validation:
  - `cargo nextest run -p siumai-spec usage --no-fail-fast`
  - `cargo nextest run -p siumai-core streaming::processor --no-fail-fast`
  - protocol usage fixtures if touched
- Status: READY
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
- AICH-050 established `ProviderMetadataMap` / stream `providerMetadata` as public provider-scoped
  projection, while `ResponseMetadata.headers/body`, HTTP request/response bodies,
  `ChatStreamPart::Raw`, replay `rawItem`, and raw/private/diagnostic custom events are private
  diagnostics.
- AICH-060 established `LlmErrorExt::user_message()` as safe display copy and moved raw provider
  messages/details to diagnostics fields or verbose rendering.
- AICH-070 established portable tool-name and provider-tool-id validation as an opt-in/fallible
  failure mode while preserving legacy constructors; `providerExecuted: true` now clearly means
  provider/model-service owned execution, and `ToolInputStart` exposes only stable public fields.

## Blockers

- None.

## Next Recommended Action

- Execute AICH-080 with `run-workstream-task`: define usage snapshots as cumulative per-provider-call
  usage and guard stream processor aggregation against double-counting repeated final snapshots.
