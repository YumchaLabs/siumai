# Builder Dead Capability API Removal — Milestones

Status: Closed
Last updated: 2026-05-25

## M0 — Scope And Evidence Freeze

Exit criteria:

- Workstream docs exist and agree on active task `BDCA-010`.
- `docs/workstreams/INDEX.md` includes the active lane.
- The starting inventory records where builder capability flags are written and not read.

Result: Complete.

## M1 — Source Guard

Exit criteria:

- A focused source guard fails while `SiumaiBuilder` still exposes the no-op capability field or
  helper methods.
- The guard passes only after the field and helper methods are removed.

Result: Complete.

## M2 — API Removal

Exit criteria:

- `SiumaiBuilder` no longer stores builder-authored capability strings.
- `with_capability()`, `with_audio()`, `with_embedding()`, and `with_image_generation()` are absent
  from `SiumaiBuilder`.
- Registry/provider capability fluent APIs remain untouched.

Result: Complete.

## M3 — Migration Notes And Closeout

Exit criteria:

- Changelog or migration notes mention the removal and supported alternatives.
- Focused nextest, compile, fmt, and diff hygiene gates pass.
- The lane is closed or follow-on work is explicitly split.

Result: Complete.
