# Workstream Status Hygiene

Status: Closed
Last updated: 2026-05-17

## Why This Lane Exists

The repository now has many durable workstream directories, but most older lanes do not expose a
machine-readable status file. That makes it hard for future agents to distinguish closed history,
active work, deferred follow-ons, and unknown legacy notes before starting a new refactor.

## Relevant Authority

- Existing docs:
  - `docs/workstreams/fearless-refactor-v4/follow-ons.md`
  - `docs/workstreams/bedrock-protocol-boundary-cleanup/WORKSTREAM.json`
  - `docs/workstreams/bedrock-protocol-boundary-cleanup/HANDOFF.md`
- Workflow docs:
  - `AGENTS.md`

## Problem

`docs/workstreams/` contains 64 directories. Only a small subset has `WORKSTREAM.json` or
`workstream.json`; many legacy directories have no explicit status at all. Without an index, the
safe default is to reread large historical docs every time, or worse, to accidentally reopen stale
lanes.

## Target State

- `docs/workstreams/INDEX.md` gives a single human-readable status inventory.
- The inventory records which lanes have machine-readable status and which are legacy unknown.
- The governance rule is explicit: unknown status is not active by default and must be normalized
  before reuse.
- This lane closes after the first inventory is generated and validated.

## In Scope

- `docs/workstreams/INDEX.md`
- `docs/workstreams/workstream-status-hygiene/*`
- Documentation-only status hygiene rules.

## Out Of Scope

- Rewriting historical workstream bodies.
- Marking legacy unknown lanes closed without reading their full history.
- Changing provider, protocol, or runtime code.
- Editing ADRs.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| A status index is safer than changing all legacy statuses in one pass. | High | 54 legacy directories currently have unknown machine-readable status. | Split a normalization follow-on for a small batch. |
| Unknown historical workstreams should not be treated as active by default. | High | Recent closeout rules require explicit `WORKSTREAM.json` / handoff state. | Future agents may reopen stale lanes. |
| This is a documentation hygiene slice, not production architecture work. | High | Scope only touches `docs/workstreams`. | If code changes are needed, open a separate workstream. |

## Architecture Direction

Treat the workstream index as the shallow public interface for navigating historical lanes. The
implementation remains the individual workstream directories. The index should not duplicate full
handoffs; it should expose enough state to decide whether to resume, close, or open a new lane.

## Closeout Condition

This lane can close when:

- the index exists,
- the status source and unknown-status rule are documented,
- the initial inventory counts are recorded,
- and `git diff --check` passes.

## Closeout Summary

Closed on 2026-05-17. `docs/workstreams/INDEX.md` now lists all 65 workstream directories, records
9 machine-readable status files, and identifies 54 unknown legacy lanes. Unknown lanes are explicitly
treated as normalization gaps, not implicit active work.
