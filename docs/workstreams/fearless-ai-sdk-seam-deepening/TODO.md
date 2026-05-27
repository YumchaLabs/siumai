# Fearless AI SDK Seam Deepening - TODO

Status: Active
Last updated: 2026-05-27

## M0 - Planning And Decision Record

- [x] AISD-010 [owner=planner] [deps=none] [scope=docs/adr,docs/workstreams/fearless-ai-sdk-seam-deepening,docs/workstreams/INDEX.md]
  Goal: Open the durable seam-deepening lane and record the architecture decision.
  Validation: `python -m json.tool docs/workstreams/fearless-ai-sdk-seam-deepening/WORKSTREAM.json`; `git diff --check -- docs/adr docs/workstreams/fearless-ai-sdk-seam-deepening docs/workstreams/INDEX.md`.
  Review: Confirm ADR-0009 does not conflict with ADR-0001, ADR-0006, or ADR-0008.
  Evidence: `EVIDENCE_AND_GATES.md`, `docs/adr/0009-ai-sdk-contract-seam-deepening.md`.
  Handoff: DONE. First executable code task is AISD-020.

## M1 - Protocol Stream State

- [x] AISD-020 [owner=codex] [deps=AISD-010] [scope=siumai-protocol-openai/src/standards/openai/responses_sse/converter,CHANGELOG.md,siumai-protocol-openai/CHANGELOG.md]
  Goal: Deepen the OpenAI Responses stream state module so reasoning lifecycle, replay hints, terminal buffering, provider tool ownership, and serializer state are no longer one shallow converter interface.
  Validation: `cargo fmt --check -p siumai-protocol-openai`; `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_sse --no-fail-fast`.
  Review: Confirm no loss of StreamEnd replay, terminal-only final text, raw replay diagnostics, provider-executed tool events, or repeated usage behavior.
  Evidence: `EVIDENCE_AND_GATES.md`, protocol changelog, focused fixtures.
  Handoff: DONE. Terminal buffering is owned by `TerminalEventBuffer`, replay hint attach/apply logic is owned by `converter::replay`, reasoning state is owned by `ReasoningLifecycleState`, provider/custom tool ownership state is owned by `ProviderToolState` and `CustomToolState`, and serializer allocation rules are owned by `OpenAiResponsesSerializeState`.

- [x] AISD-030 [owner=codex] [deps=AISD-020] [scope=siumai-spec/src/types,siumai-core/src/streaming,siumai-protocol-openai/src,CHANGELOG.md,crate changelogs]
  Goal: Make public provider metadata versus private diagnostics an executable projection seam used by protocol/core output paths.
  Validation: `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai`; `cargo nextest run -p siumai-spec private_diagnostics --no-fail-fast`; `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses diagnostics --no-fail-fast`.
  Review: Confirm raw headers, bodies, raw stream parts, replay raw items, and reserved custom events cannot silently enter public provider metadata.
  Evidence: `EVIDENCE_AND_GATES.md`, changelogs.
  Handoff: DONE. Provider metadata public projection strips reserved raw/private diagnostic keys, `ChatResponse` and stream events expose public projections, core final content uses the projection, and protocol has a Responses SSE diagnostics regression.

## M2 - Tool, Capability, And Usage Contract Depth

- [x] AISD-040 [owner=codex] [deps=AISD-010] [scope=siumai-spec/src/types/tools,siumai-spec/src/types/prompt.rs,siumai-core/src/tooling,siumai-core/src/ui,siumai-protocol-openai/src,CHANGELOG.md,crate changelogs]
  Goal: Deepen provider-executed tool ownership into one contract module reused by prompt validation, runtime tooling, UI conversion, and protocol adapters.
  Validation: `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai`; `cargo nextest run -p siumai-spec provider_executed --no-fail-fast`; `cargo nextest run -p siumai-core provider_executed --no-fail-fast`; `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses provider_executed --no-fail-fast`.
  Review: Confirm constructor naming no longer forces callers to remember ownership differences.
  Evidence: `EVIDENCE_AND_GATES.md`, changelogs.
  Handoff: DONE. `ToolExecutionOwner` is the semantic contract; legacy AI SDK wire flags remain compatibility fields, prompt/UI/core/protocol decisions use the owner helper, and the obsolete ignored provider-executed stream-result parameter was removed.

- [x] AISD-050 [owner=codex] [deps=AISD-010] [scope=siumai-core/src/execution,siumai-core/src/traits/capabilities.rs,siumai-core/src/error/helpers.rs,siumai-spec/src/types/common.rs,CHANGELOG.md,siumai-core/CHANGELOG.md]
  Goal: Centralize unsupported capability guard choreography behind a deep module so executors do not repeat feature strings, details, policy creation, and policy resolution.
  Validation: `cargo fmt --check -p siumai-core -p siumai-spec`; `cargo nextest run -p siumai-core unsupported_capability --no-fail-fast`; `cargo nextest run -p siumai-core reject_if_unsupported --no-fail-fast`; focused executor guard tests.
  Review: Confirm every hard family executor gate crosses the same seam.
  Evidence: `EVIDENCE_AND_GATES.md`, changelogs.
  Handoff: DONE. `execution::capability` now owns named hard family requirements, the shared
  reject/warn/provider-fallback policy resolver, and a boundary test proving executors use the same
  seam instead of local feature strings or policy reconstruction.

- [x] AISD-060 [owner=codex] [deps=AISD-010] [scope=siumai-spec/src/types/usage.rs,siumai-core/src/streaming,siumai-protocol-openai/src/standards/openai/responses_sse,CHANGELOG.md,crate changelogs]
  Goal: Give stream usage snapshots a named ledger module that handles replacement separately from explicit multi-call aggregation.
  Validation: `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai`; `cargo nextest run -p siumai-spec usage --no-fail-fast`; `cargo nextest run -p siumai-core streaming::processor --no-fail-fast`; OpenAI Responses repeated-usage fixture.
  Review: Confirm stream paths cannot accidentally call `Usage::merge()` for same-call snapshots.
  Evidence: `EVIDENCE_AND_GATES.md`, changelogs.
  Handoff: DONE. `UsageSnapshotLedger` now owns same-call snapshot replacement; core stream
  processing and OpenAI Responses serializer state record usage through the ledger, while
  `Usage::merge()` remains the explicit orchestration aggregation API.

## M3 - Core Stream Assembly And Closeout

- [ ] AISD-070 [owner=codex] [deps=AISD-020,AISD-060] [scope=siumai-core/src/streaming/processor.rs,siumai-core/src/streaming/processor,CHANGELOG.md,siumai-core/CHANGELOG.md]
  Goal: Separate delta accumulation from final response assembly so terminal replay reconciliation owns a compact accumulated record instead of broad processor internals.
  Validation: `cargo fmt --check -p siumai-core`; `cargo nextest run -p siumai-core streaming::processor --no-fail-fast`; core provider boundary tests.
  Review: Confirm StreamEnd replay, ToolInputStart projection, repeated usage, and terminal extra parts remain covered.
  Evidence: `EVIDENCE_AND_GATES.md`, changelogs.
  Handoff: This task may be skipped or narrowed if AISD-020/AISD-060 make the remaining core seam deep enough.

- [ ] AISD-080 [owner=planner] [deps=AISD-020,AISD-030,AISD-040,AISD-050,AISD-060,AISD-070] [scope=docs/workstreams/fearless-ai-sdk-seam-deepening,CHANGELOG.md,crate changelogs]
  Goal: Review, verify, close, or split any residual provider-specific follow-ons.
  Validation: `verify-rust-workstream` records fresh final gates; `review-workstream` has no blocking findings.
  Review: Confirm ADR/workstream/changelogs agree on final architecture.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`.
  Handoff: Hajimi adapter changes remain out of scope.
