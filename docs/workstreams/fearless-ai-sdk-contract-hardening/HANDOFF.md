# Fearless AI SDK Contract Hardening - Handoff

Status: Closed
Last updated: 2026-05-26

## Final State

The workstream is closed. AICH-010 through AICH-100 are complete. The bootstrap captured the 12-gap
audit, linked the local AI SDK reference under `repo-ref/ai`, and made changelog tracking part of
the task ledger. AICH-020 documented `StreamEnd.response.content` as final replay/fallback.
AICH-030 added an OpenAI Responses SSE fixture proving reasoning deltas and terminal-only final
visible text are both preserved. AICH-040 documented `stream_with_cancel` as the recommended
cancelable stream entry and separated default local cancellation from provider-specific remote
abort. AICH-050 defined public provider metadata versus private raw diagnostics boundaries.
AICH-060 split safe user-facing error messages from raw diagnostics. AICH-070 added explicit tool
validation/failure helpers, documented provider-executed ownership, and locked `ToolInputStart`
stable projection away from provider replay indexes. AICH-080 defined usage as a per-provider-call
cumulative snapshot and stopped repeated finish usage snapshots from being over-counted. AICH-090
added explicit unsupported-capability policy behaviors for reject, warn, and provider fallback, and
routed core hard family guards through the shared reject policy.

## Closeout Evidence

- `python -m json.tool docs\workstreams\fearless-ai-sdk-contract-hardening\WORKSTREAM.json`
- `git diff --check -- CHANGELOG.md siumai-spec\CHANGELOG.md siumai-core\CHANGELOG.md docs\workstreams\fearless-ai-sdk-contract-hardening siumai-core\src siumai-core\tests`
- `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai`
- `cargo nextest run -p siumai-core --no-fail-fast`
- `cargo nextest run -p siumai-spec --no-fail-fast`
- `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses --no-fail-fast`
- `cargo nextest run -p siumai-spec -p siumai-core -p siumai-protocol-openai --no-fail-fast`

Full workspace `cargo fmt --check` was attempted and failed with Windows path-length error 206.
This is the same environment limit seen earlier in the lane, so the crate-scoped fmt gate is the
recorded formatting evidence.

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
- AICH-080 established `Usage` as one provider/model-call cumulative snapshot. Stream processor
  usage updates now replace earlier same-call snapshots, while `Usage::merge()` remains explicit
  multi-call aggregation and drops provider-native raw usage.
- AICH-090 established `UnsupportedCapabilityPolicy` / `UnsupportedCapabilityBehavior` as the
  shared decision point before unsupported capabilities become errors or warnings. Hard capability
  guards use `reject`; lossy unsupported option projection should use `warn`; provider-owned
  uncertainty should use `provider-fallback`.

## Blockers

- None.

## Follow-Ons

- No split Siumai follow-up is required for the 12-gap contract-hardening scope.
- Provider-specific unsupported option rewrites should be opened as new provider-scoped workstreams
  only when a concrete behavior gap is found.
- Hajimi adapter changes remain out of scope and should be handled after this Siumai lane lands.
