# Fearless AI SDK Seam Deepening - Milestones

Status: Active
Last updated: 2026-05-27

## M0 - Planning And Decision Record

Exit criteria:

- ADR-0009 records the seam-deepening decision.
- Workstream docs and task ledger exist.
- Workstream index shows one active lane.

## M1 - Protocol Stream State

Exit criteria:

- OpenAI Responses converter state is internally deepened without losing fixture parity.
- Replay, reasoning lifecycle, terminal buffering, provider tool state, and usage replacement have
  named modules or equivalent locality.
- Protocol changelog records behavior-visible changes.

## M2 - Tool, Capability, And Usage Contract Depth

Exit criteria:

- Provider-executed tool ownership is no longer scattered caller knowledge.
- Unsupported capability guard choreography is centralized.
- Stream usage snapshot replacement has a named module distinct from `Usage::merge()`.
- Spec/core/protocol changelogs cover public or behavior-visible contract changes.

## M3 - Core Stream Assembly And Closeout

Exit criteria:

- Core `StreamProcessor` only exposes the interface needed for event processing and final assembly.
- Any obsolete shallow helpers are deleted or explicitly retained with compatibility rationale.
- Fresh closeout gates pass or blocked workspace-wide gates are recorded with a concrete reason.
- `WORKSTREAM.json`, `TODO.md`, `HANDOFF.md`, and `EVIDENCE_AND_GATES.md` agree on final state.
