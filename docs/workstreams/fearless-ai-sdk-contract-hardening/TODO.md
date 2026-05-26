# Fearless AI SDK Contract Hardening - TODO

Status: Active
Last updated: 2026-05-26

## M0 - Scope And Evidence Freeze

- [x] AICH-010 [owner=planner] [deps=none] [scope=docs/workstreams/fearless-ai-sdk-contract-hardening,CHANGELOG.md]
  Goal: Open the durable lane, freeze the 12-gap audit, link AI SDK reference files, and require changelog tracking.
  Validation: Workstream docs exist; `WORKSTREAM.json` parses; `docs/workstreams/INDEX.md` includes the active lane; root changelog has a tracking entry.
  Evidence: `DESIGN.md`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`, `CHANGELOG.md`
  Handoff: Planner completed the bootstrap. First executable implementation task is AICH-020.

## M1 - Stream Replay And Cancellation Contracts

- [x] AICH-020 [owner=codex] [deps=AICH-010] [scope=siumai-spec/src/types/streaming.rs,siumai-spec/src/types/chat/response.rs,siumai-core/src/streaming,CHANGELOG.md,siumai-spec/CHANGELOG.md,siumai-core/CHANGELOG.md]
  Goal: Define and test `StreamEnd.response.content` as final response replay/fallback, not append-only delta.
  Validation: `cargo nextest run -p siumai-spec stream --no-fail-fast`; `cargo nextest run -p siumai-core streaming::processor --no-fail-fast`
  Review: Verify downstream consumers can dedupe text/reasoning/tool calls without losing terminal-only content.
  Evidence: Stream type docs, processor tests, changelog entries.
  Handoff: DONE. `StreamEnd.response.content` is documented as final replay/fallback; processor regression test proves it is not appended as another text delta. Adapter authors should consume deltas for live UI and reconcile `StreamEnd` as final replay.

- [x] AICH-030 [owner=codex] [deps=AICH-020] [scope=siumai-protocol-openai/src/standards/openai/responses_sse,siumai-provider-openai/src,siumai-provider-openai-compatible/src,CHANGELOG.md,siumai-protocol-openai/CHANGELOG.md]
  Goal: Add a fixture for reasoning deltas with final visible text materialized only through terminal response content.
  Validation: `cargo nextest run -p siumai-protocol-openai responses_sse --no-fail-fast`; provider-focused tests if the fixture lives in a provider crate.
  Review: Ensure final text and reasoning are both preserved and not double-counted.
  Evidence: OpenAI Responses or OpenAI-compatible stream fixture.
  Handoff: DONE. OpenAI Responses SSE now has a fixture where reasoning deltas stream first, no text deltas are emitted, and terminal `response.completed` supplies final visible text without losing reasoning.

- [x] AICH-040 [owner=codex] [deps=AICH-010] [scope=siumai-core/src/text.rs,siumai-core/src/traits/chat.rs,siumai/src/text.rs,docs,CHANGELOG.md,siumai-core/CHANGELOG.md]
  Goal: Document and test `stream_with_cancel` as the recommended cancelable text stream entry, with local vs remote cancel semantics separated.
  Validation: `cargo nextest run -p siumai-core stream_with_cancel --no-fail-fast`; existing OpenAI remote cancel test remains green.
  Review: Do not promise provider remote abort unless the provider has a direct test.
  Evidence: API docs and cancellation tests.
  Handoff: DONE. `stream_with_cancel` is documented as the recommended cancelable stream entry; default implementations guarantee local stream-consumption cancellation, while remote abort remains provider-specific and covered by the OpenAI Responses regression test.

## M2 - Diagnostics, Tools, Usage, And Capability Contracts

- [x] AICH-050 [owner=codex] [deps=AICH-020] [scope=siumai-spec/src/types/common.rs,siumai-spec/src/types/chat/content,siumai-spec/src/types/streaming.rs,siumai-core/src/streaming,siumai-protocol-openai,CHANGELOG.md,siumai-spec/CHANGELOG.md]
  Goal: Define raw/private diagnostics treatment for provider metadata, `ResponseMetadata.headers/body`, `ChatStreamPart::Raw`, and `ChatStreamEvent::Custom`.
  Validation: `cargo nextest run -p siumai-spec metadata --no-fail-fast`; targeted protocol raw-event tests.
  Review: Public projections must whitelist safe fields; raw provider data remains available for diagnostics.
  Evidence: Type docs, serialization tests, protocol raw event fixtures.
  Handoff: DONE. `ProviderMetadataMap` is documented as the public provider-scoped projection lane, while `ResponseMetadata.headers/body`, HTTP request/response bodies, `ChatStreamPart::Raw`, replay `rawItem`, and `raw`/`private`/`diagnostic` custom event types are private diagnostics with routing helpers and regression coverage.

- [x] AICH-060 [owner=codex] [deps=AICH-050] [scope=siumai-core/src/error,siumai-spec/src/error,CHANGELOG.md,siumai-core/CHANGELOG.md]
  Goal: Split safe/public error message semantics from raw diagnostic detail so `user_message()` cannot leak provider body, headers, or raw request data.
  Validation: `cargo nextest run -p siumai-core error --no-fail-fast`
  Review: Confirm fallback `Display` paths are not treated as safe public copy.
  Evidence: Error policy docs and tests.
  Handoff: DONE. `LlmErrorExt::user_message()` is now documented and tested as safe display copy that avoids raw `Display` fallback; `summarize_error().message` uses that safe copy while raw provider messages/details remain in diagnostics fields and verbose rendering only.

- [x] AICH-070 [owner=codex] [deps=AICH-010] [scope=siumai-spec/src/types/tools,siumai-spec/src/types/prompt.rs,siumai-core/src/streaming,CHANGELOG.md,siumai-spec/CHANGELOG.md]
  Goal: Harden tool contracts: tool-name validation/failure mode, provider-executed execution owner, and `ToolInputStart` stable field projection.
  Validation: `cargo nextest run -p siumai-spec tools --no-fail-fast`; targeted stream processor tool tests.
  Review: Keep provider-specific replay index out of stable `ToolInputStart` unless promoted by an ADR.
  Evidence: Tool docs, validation tests, provider-executed prompt tests.
  Handoff: DONE. Tool names and provider-tool ids now have explicit validation helpers, fallible constructors, and `validate_contract()` methods while legacy constructors remain infallible. Provider-executed means provider/model-service execution ownership; `ToolInputStart` carries only stable public fields while replay indexes/raw items stay in replay hints.

- [ ] AICH-080 [owner=unassigned] [deps=AICH-020] [scope=siumai-spec/src/types/usage.rs,siumai-core/src/streaming/processor.rs,siumai-protocol-openai,CHANGELOG.md,siumai-spec/CHANGELOG.md,siumai-core/CHANGELOG.md]
  Goal: Define stream usage as a single provider-call cumulative snapshot and prevent accidental over-counting from repeated cumulative finish usage.
  Validation: `cargo nextest run -p siumai-spec usage --no-fail-fast`; `cargo nextest run -p siumai-core streaming::processor --no-fail-fast`; protocol usage fixtures.
  Review: Preserve AI SDK-style `raw` usage for the final provider call while avoiding meaningless raw aggregation.
  Evidence: Usage docs and repeated-finish usage regression test.
  Handoff: If a provider emits usage deltas, document it as provider-specific before merging.

- [ ] AICH-090 [owner=unassigned] [deps=AICH-050,AICH-070,AICH-080] [scope=siumai-spec/src/types/common.rs,siumai-core/src,provider crates as needed,docs,CHANGELOG.md]
  Goal: Make unsupported provider capability behavior explicit: reject, warn, or provider fallback, with shared tests for common behavior.
  Validation: targeted package tests for warnings/errors touched by the slice.
  Review: Do not silently ignore caller settings that change requested semantics.
  Evidence: Warning/error docs, capability tests, changelog entries.
  Handoff: Split provider-specific unsupported behavior into follow-ons if it grows beyond shared contract work.

## M3 - Integration And Closeout

- [ ] AICH-100 [owner=planner] [deps=AICH-020,AICH-030,AICH-040,AICH-050,AICH-060,AICH-070,AICH-080,AICH-090] [scope=docs/workstreams/fearless-ai-sdk-contract-hardening,CHANGELOG.md,crate changelogs]
  Goal: Close or split the lane after fresh verification, review, and changelog reconciliation.
  Validation: `verify-rust-workstream` records fresh final gate evidence; `review-workstream` has no blocking findings.
  Review: Confirm root and touched crate changelogs mention shipped contract changes.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`, changelog diffs.
  Handoff: Remaining Hajimi adapter changes are out of scope and should be handled after this Siumai lane lands.
