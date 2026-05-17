# Workstream Status Hygiene - TODO

Status: Closed
Last updated: 2026-05-17

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done

## M0 - Scope And Evidence Freeze

- [x] WSH-010 [owner=codex] [deps=none] [scope=docs/workstreams/workstream-status-hygiene]
  Goal: Freeze the documentation hygiene problem, non-goals, and validation gate.
  Validation: Workstream docs exist and agree.
  Evidence: `DESIGN.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`
  Handoff: First proof slice is a generated workstream status index.

## M1 - Status Inventory

- [x] WSH-020 [owner=codex] [deps=WSH-010] [scope=docs/workstreams/INDEX.md]
  Goal: Add a single status inventory for every directory under `docs/workstreams`.
  Validation: Inventory counts match filesystem scan.
  Evidence: `docs/workstreams/INDEX.md`
  Handoff: Unknown legacy lanes are inventory gaps, not implicit active work.

## M2 - Closeout

- [x] WSH-030 [owner=codex] [deps=WSH-020] [scope=docs/workstreams/workstream-status-hygiene]
  Goal: Close the lane after validation.
  Validation: `git diff --check`
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`
  Handoff: Lane closed. Split a follow-on only for batch-normalizing legacy unknown statuses.
