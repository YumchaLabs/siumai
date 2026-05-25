# Registry Typed Builder Isolation — Milestones

Status: Closed
Last updated: 2026-05-25

## M0 — Scope And Evidence Freeze

Exit criteria:

- Workstream docs exist and agree on active task `RTBI-010`.
- `docs/workstreams/INDEX.md` includes the active lane.
- The lane explicitly keeps deprecated generic-client helpers out of scope.

Result: Complete.

## M1 — Source Guard

Exit criteria:

- A source guard fails while production factories call typed helpers through `registry::factory`.
- The guard passes after production calls move to the internal typed builder module.

Result: Complete.

## M2 — Typed Builder Isolation

Exit criteria:

- Typed helper implementation lives behind an internal registry module.
- Production provider factories call the internal module.
- `registry::factory` remains focused on public compatibility wrappers.

Result: Complete.

## M3 — Docs And Closeout

Exit criteria:

- Changelog or architecture docs describe the module ownership clarification.
- Focused nextest, all-features compile, fmt, and diff hygiene gates pass.
- The lane is closed or follow-on work is explicitly split.

Result: Complete.
