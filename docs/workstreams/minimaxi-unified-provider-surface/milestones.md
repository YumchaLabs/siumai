# MiniMaxi Unified Provider Surface - Milestones

Last updated: 2026-04-10

## MMP-M0 - Scope locked

Acceptance criteria:

- the remaining MiniMaxi package-shape drift is described explicitly
- the chosen fix keeps the provider-owned wrapper instead of reopening a larger architecture split

Status: completed

## MMP-M1 - Curated model source exists

Acceptance criteria:

- `siumai-provider-minimaxi` owns a curated `models.rs`
- the curated surface groups public families by capability rather than raw legacy constants only

Status: completed

## MMP-M2 - Facade and catalog share one source

Acceptance criteria:

- `provider_ext::minimaxi` exposes the curated model families
- provider catalog output reuses the same source instead of handwritten arrays
- image-family defaults are no longer omitted from the catalog

Status: completed

## MMP-M3 - Stream metadata boundary is normalized

Acceptance criteria:

- MiniMaxi stream finish metadata no longer leaks the borrowed `anthropic` provider key
- typed stream parts and final `StreamEnd` both use the public `minimaxi` root

Status: completed

## MMP-M4 - Docs and changelog handoff complete

Acceptance criteria:

- a dedicated workstream folder exists
- fearless-refactor docs mention the new single-source model/catalog shape
- unreleased changelogs describe the user-visible implications

Status: completed
