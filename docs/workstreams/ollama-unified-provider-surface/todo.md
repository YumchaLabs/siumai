# Ollama Unified Provider Surface - TODO

Last updated: 2026-04-10

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Naming cleanup

- [x] Move the runtime model-listing implementation out of `models.rs`.
- [x] Keep the runtime model-listing behavior intact after the rename.

## Track B - Curated model surface

- [x] Add provider-owned curated Ollama `chat` / `embedding` model constants.
- [x] Expose grouped defaults plus `all_models()` for catalog/default reuse.

## Track C - Public alignment

- [x] Expose curated model constants on `provider_ext::ollama`.
- [x] Switch provider catalog output to the curated provider-owned source.
- [x] Switch `get_default_models()` to the curated provider-owned source.

## Follow-up

- [ ] Revisit the curated Ollama subset when the public default list changes intentionally.
- [-] Delete the broader legacy `model_constants.rs` surface immediately.
  - deferred because downstream callers may still rely on the wider alias coverage
