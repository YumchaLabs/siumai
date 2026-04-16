# Ollama Unified Provider Surface - Milestones

Last updated: 2026-04-10

## OLP-M0 - Scope locked

Acceptance criteria:

- the Ollama package-shape collision around `models` is documented
- the chosen refactor keeps behavior but restores a consistent naming contract

Status: completed

## OLP-M1 - Runtime/facade naming conflict is removed

Acceptance criteria:

- runtime model-listing no longer occupies `providers/ollama/models.rs`
- the renamed runtime module still builds and tests cleanly

Status: completed

## OLP-M2 - Curated model source exists

Acceptance criteria:

- provider-owned curated `models.rs` exists for Ollama
- the public families explicitly cover chat plus embeddings

Status: completed

## OLP-M3 - Catalog/defaults/facade share one source

Acceptance criteria:

- `provider_ext::ollama` exposes the curated model families
- provider catalog output reuses the curated source
- `get_default_models()` reuses the curated source

Status: completed

## OLP-M4 - Docs and changelog handoff complete

Acceptance criteria:

- a dedicated workstream folder exists
- fearless-refactor docs mention the new single-source model shape
- unreleased changelogs describe the user-visible implications

Status: completed
