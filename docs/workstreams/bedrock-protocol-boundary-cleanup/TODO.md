# Bedrock Protocol Boundary Cleanup - TODO

Status: Closed
Last updated: 2026-05-17

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## M0 - Scope And Evidence Freeze

- [x] BPC-010 [owner=planner] [deps=none] [scope=docs/workstreams/bedrock-protocol-boundary-cleanup]
  Goal: Freeze problem, target state, non-goals, and evidence anchors.
  Validation: Workstream docs exist and agree.
  Evidence: `DESIGN.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`
  Handoff: First implementation slice is Bedrock chat test ownership isolation.

## M1 - Bedrock Chat Test Ownership

- [x] BPC-020 [owner=codex] [deps=BPC-010] [scope=siumai-provider-amazon-bedrock/src/standards/bedrock/chat.rs,siumai-provider-amazon-bedrock/src/standards/bedrock/chat/tests.rs]
  Goal: Move the Bedrock chat standard contract suite out of the production protocol shell while
  preserving private access and behavior.
  Validation: `cargo fmt -p siumai-provider-amazon-bedrock`; `cargo nextest run -p siumai-provider-amazon-bedrock --all-features --no-fail-fast bedrock`
  Evidence: `siumai-provider-amazon-bedrock/src/standards/bedrock/chat/tests.rs`
  Handoff: Test ownership is isolated. Next step is to audit production request/response/stream
  extraction and stop if splitting would only move code by size.

## M2 - Production Boundary Audit

- [x] BPC-030 [owner=codex] [deps=BPC-020] [scope=siumai-provider-amazon-bedrock/src/standards/bedrock/chat.rs,siumai-provider-amazon-bedrock/src/standards/bedrock/chat/streaming.rs,siumai-provider-amazon-bedrock/src/standards/bedrock/chat/tests.rs]
  Goal: Reassess request planning, response metadata, and stream conversion after test isolation.
  Validation: `cargo fmt -p siumai-provider-amazon-bedrock`; `cargo nextest run -p siumai-provider-amazon-bedrock --all-features --no-fail-fast bedrock`
  Evidence: `siumai-provider-amazon-bedrock/src/standards/bedrock/chat/streaming.rs`, `HANDOFF.md`
  Handoff: Stream conversion was extracted because it owns Bedrock stream DTOs, accumulator state,
  and JSON event conversion. Request planning and response shaping remain in `chat.rs` because the
  remaining helpers are shared Bedrock protocol mapping code; splitting them now would mostly create
  cross-module private glue.

## M3 - Closeout

- [x] BPC-040 [owner=codex] [deps=BPC-030] [scope=docs/workstreams/bedrock-protocol-boundary-cleanup]
  Goal: Close the lane or split a narrower follow-on.
  Validation: `git diff --check`; final focused gates recorded.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`
  Handoff: Lane closed. No immediate follow-on was split; future Bedrock work should start from a
  concrete behavior issue rather than a file-size cleanup.
