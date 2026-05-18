# Provider Surface Second Pass - TODO

Status: Closed
Last updated: 2026-05-18

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## M0 - Scope And Inventory

- [x] PSS-010 [owner=planner] [deps=none] [scope=docs/workstreams/provider-surface-second-pass]
  Goal: Freeze the second-pass scope and identify the residual provider-surface candidates.
  Validation: workstream docs exist and agree, and the residual lane list is explicit.
  Evidence: `docs/workstreams/INDEX.md`
  Handoff: Residual candidates were classified without reopening the closed AIPC parent.

## M1 - Workstream Hygiene

- [x] PSS-020 [owner=codex] [deps=PSS-010] [scope=docs/workstreams]
  Goal: Normalize the index and status metadata for the residual lane set.
  Validation: `docs/workstreams/INDEX.md` matches the filesystem and machine-readable status files.
  Evidence: `docs/workstreams/INDEX.md`
  Handoff: Historical unknowns are now machine-readable: empty fearless placeholders are
  superseded, Google Vertex typed options are closed, and stream metadata hardening is closed with
  future-trigger guidance.

## M2 - Residual Candidate Audit

- [x] PSS-030 [owner=codex] [deps=PSS-020] [scope=docs/workstreams/stream-metadata-parity-hardening]
  Goal: Decide whether `stream-metadata-parity-hardening` is the next concrete follow-on or should
  be split/closed.
  Validation: the lane has machine-readable status and its open items are either addressed or split.
  Evidence: `docs/workstreams/stream-metadata-parity-hardening/*`
  Handoff: Closed as historical completed work. Its open items are future triggers only: resume a
  narrower serializer or metadata-helper lane when a concrete provider drift appears.

- [x] PSS-040 [owner=codex] [deps=PSS-020] [scope=docs/workstreams/minimaxi-unified-provider-surface,docs/workstreams/ollama-unified-provider-surface]
  Goal: Reconfirm the deferred MiniMaxi and Ollama provider-surface decisions against the current
  repository state.
  Validation: the docs still describe the intentional deferral and do not imply AI SDK parity.
  Evidence: `docs/workstreams/minimaxi-unified-provider-surface/*`
  Evidence: `docs/workstreams/ollama-unified-provider-surface/*`
  Handoff: Both remain intentional deferred Siumai-native provider surfaces. No code work is split.

## M3 - Closeout

- [x] PSS-050 [owner=planner] [deps=PSS-020,PSS-030,PSS-040] [scope=docs/workstreams/provider-surface-second-pass]
  Goal: Close the lane or split the remaining follow-on work into narrower child lanes.
  Validation: `WORKSTREAM.json`, `HANDOFF.md`, and `EVIDENCE_AND_GATES.md` agree.
  Evidence: `docs/workstreams/provider-surface-second-pass/WORKSTREAM.json`
  Handoff: Closed. No concrete provider drift was found; future work should open a narrow lane from
  a concrete provider behavior, metadata helper, serializer, or public contract gap.
