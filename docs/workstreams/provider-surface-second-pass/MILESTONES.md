# Provider Surface Second Pass - Milestones

Status: Closed
Last updated: 2026-05-18

## M0 - Scope And Inventory

Exit criteria:

- the second-pass scope is explicit
- the residual candidate set is named
- the lane does not reopen the closed AIPC parent by accident

Primary evidence:

- `docs/workstreams/provider-surface-second-pass/DESIGN.md`
- `docs/workstreams/provider-surface-second-pass/TODO.md`

Status: completed

## M1 - Workstream Hygiene

Exit criteria:

- `docs/workstreams/INDEX.md` matches the actual directory and status layout
- the remaining historical unknowns are explicitly justified

Primary evidence:

- `docs/workstreams/INDEX.md`
- residual lane `WORKSTREAM.json` files

Status: completed

## M2 - Residual Candidate Audit

Exit criteria:

- `stream-metadata-parity-hardening` is either resumed as the next code lane or split/closed
- `minimaxi` and `ollama` remain documented as intentional deferrals unless evidence says otherwise
- any concrete drift is isolated into a bounded child lane

Primary evidence:

- residual workstream docs and handoffs
- any follow-on child workstream docs opened from this lane

Status: completed

Progress:

- `stream-metadata-parity-hardening` is closed as completed historical hardening work with
  future-trigger guidance.
- `minimaxi` and `ollama` remain intentional deferred Siumai-native surfaces.
- `google-vertex-typed-option-surface-alignment` is closed with only authenticated GCS
  materialization as a future trigger.

## M3 - Closeout

Exit criteria:

- the lane can hand off a stable residual candidate set
- the docs teach future agents what is historical, deferred, and still actionable

Primary evidence:

- `docs/workstreams/provider-surface-second-pass/WORKSTREAM.json`
- `docs/workstreams/provider-surface-second-pass/HANDOFF.md`

Status: completed
