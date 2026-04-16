# MiniMaxi Unified Provider Surface - TODO

Last updated: 2026-04-10

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Curated model source

- [x] Add provider-owned curated model constants for MiniMaxi public families.
- [x] Keep legacy `model_constants.rs` available as the broader compatibility layer.

## Track B - Facade and registry alignment

- [x] Expose the curated model families on `provider_ext::minimaxi`.
- [x] Switch provider catalog output to the provider-owned curated source.
- [x] Include image-family defaults in the catalog output.

## Track C - Stream metadata normalization

- [x] Normalize MiniMaxi finish-part provider metadata on the typed stream-part lane.
- [x] Keep `StreamEnd` metadata normalized under the same `minimaxi` root.

## Follow-up

- [ ] Revisit the curated model subset when MiniMaxi promotes new first-class family ids.
- [-] Remove the broader legacy `model_constants.rs` surface immediately.
  - deferred because internal callers may still rely on its wider alias coverage
