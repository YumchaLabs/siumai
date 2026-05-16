# Fearless Vision Compatibility Removal - Handoff

Last updated: 2026-05-16

## Current State

The workstream is closed. `VCR-010` through `VCR-050` are complete.

The dedicated vision compatibility surface has been removed from production core, registry, and spec
sources, including the deprecated `SiumaiBuilder::with_vision()` capability-tag hint. The beta.7
migration guide now points users to multimodal chat for image understanding and image-family APIs
for image creation.

## Continuation Notes

- The removal guard intentionally allows historical docs and migration text to keep naming the
  removed symbols.
- No follow-up is split from this lane. Reopen only if downstream evidence shows a missing
  migration replacement.
