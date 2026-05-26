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

- [ ] AISD-020 [owner=codex] [deps=AISD-010] [scope=siumai-protocol-openai/src/standards/openai/responses_sse/converter,CHANGELOG.md,siumai-protocol-openai/CHANGELOG.md]
  Goal: Deepen the OpenAI Responses stream state module so reasoning lifecycle, replay hints, terminal buffering, provider tool ownership, and serializer state are no longer one shallow converter interface.
  Validation: `cargo fmt --check -p siumai-protocol-openai`; `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses responses_sse --no-fail-fast`.
  Review: Confirm no loss of StreamEnd replay, terminal-only final text, raw replay diagnostics, provider-executed tool events, or repeated usage behavior.
  Evidence: `EVIDENCE_AND_GATES.md`, protocol changelog, focused fixtures.
  Handoff: IN_PROGRESS. Terminal buffering is owned by `TerminalEventBuffer`, replay hint attach/apply logic is owned by `converter::replay`, and reasoning state is owned by `ReasoningLifecycleState`; remaining work should deepen provider tool state and serializer state before marking the task done.

- [ ] AISD-030 [owner=codex] [deps=AISD-020] [scope=siumai-spec/src/types,siumai-core/src/streaming,siumai-protocol-openai/src,CHANGELOG.md,crate changelogs]
  Goal: Make public provider metadata versus private diagnostics an executable projection seam used by protocol/core output paths.
  Validation: `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai`; `cargo nextest run -p siumai-spec private_diagnostics --no-fail-fast`; `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses diagnostics --no-fail-fast`.
  Review: Confirm raw headers, bodies, raw stream parts, replay raw items, and reserved custom events cannot silently enter public provider metadata.
  Evidence: `EVIDENCE_AND_GATES.md`, changelogs.
  Handoff: Delete or demote pass-through helpers only when projection parity is tested.

## M2 - Tool, Capability, And Usage Contract Depth

- [ ] AISD-040 [owner=codex] [deps=AISD-010] [scope=siumai-spec/src/types/tools,siumai-spec/src/types/prompt.rs,siumai-core/src/tooling,siumai-core/src/ui,siumai-protocol-openai/src,CHANGELOG.md,crate changelogs]
  Goal: Deepen provider-executed tool ownership into one contract module reused by prompt validation, runtime tooling, UI conversion, and protocol adapters.
  Validation: `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai`; `cargo nextest run -p siumai-spec provider_executed --no-fail-fast`; `cargo nextest run -p siumai-core provider_executed --no-fail-fast`; `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses provider_executed --no-fail-fast`.
  Review: Confirm constructor naming no longer forces callers to remember ownership differences.
  Evidence: `EVIDENCE_AND_GATES.md`, changelogs.
  Handoff: Preserve compatibility constructors unless tests prove a helper is obsolete.

- [ ] AISD-050 [owner=codex] [deps=AISD-010] [scope=siumai-core/src/execution,siumai-core/src/traits/capabilities.rs,siumai-core/src/error/helpers.rs,siumai-spec/src/types/common.rs,CHANGELOG.md,siumai-core/CHANGELOG.md]
  Goal: Centralize unsupported capability guard choreography behind a deep module so executors do not repeat feature strings, details, policy creation, and policy resolution.
  Validation: `cargo fmt --check -p siumai-core -p siumai-spec`; `cargo nextest run -p siumai-core unsupported_capability --no-fail-fast`; `cargo nextest run -p siumai-core reject_if_unsupported --no-fail-fast`; focused executor guard tests.
  Review: Confirm every hard family executor gate crosses the same seam.
  Evidence: `EVIDENCE_AND_GATES.md`, changelogs.
  Handoff: Delete duplicated helper code in executors once the shared gate proves parity.

- [ ] AISD-060 [owner=codex] [deps=AISD-010] [scope=siumai-spec/src/types/usage.rs,siumai-core/src/streaming,siumai-protocol-openai/src/standards/openai/responses_sse,CHANGELOG.md,crate changelogs]
  Goal: Give stream usage snapshots a named ledger module that handles replacement separately from explicit multi-call aggregation.
  Validation: `cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai`; `cargo nextest run -p siumai-spec usage --no-fail-fast`; `cargo nextest run -p siumai-core streaming::processor --no-fail-fast`; OpenAI Responses repeated-usage fixture.
  Review: Confirm stream paths cannot accidentally call `Usage::merge()` for same-call snapshots.
  Evidence: `EVIDENCE_AND_GATES.md`, changelogs.
  Handoff: Keep `Usage::merge()` as orchestration aggregation.

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
