# Fearless Vision Compatibility Removal - TODO

Last updated: 2026-05-16

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[d]` deferred with rationale

## VCR-010 - Workstream Framing

- [x] Create design, TODO, milestones, evidence, handoff, and machine-readable workstream docs.
- [x] Link the workstream from `docs/README.md`.

Validation:

- Documentation review and `git diff --check`.

## VCR-020 - Removal Guard

- [x] Add or update source guard coverage proving the dedicated vision compatibility family is
      removed from production public surfaces.
- [x] Keep the guard narrow enough to allow historical migration notes and audit text.

Validation:

- `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --features openai,google --no-default-features --no-fail-fast`

## VCR-030 - Remove Vision Compatibility Code

- [x] Remove `VisionCapability` from core traits and `LlmClient` downcast methods.
- [x] Remove `VisionCapabilityProxy` and `Siumai::vision_capability()`.
- [x] Remove the deprecated `SiumaiBuilder::with_vision()` capability-tag hint.
- [x] Remove provider/factory forwarding impls that only delegate the deprecated vision downcast.
- [x] Remove deprecated request/response type aliases that only existed for the vision trait.

Validation:

- `cargo check -p siumai-core --no-default-features`
- `cargo check -p siumai-registry --tests --features openai,google --no-default-features`
- `cargo check -p siumai --tests --features openai,google --no-default-features`

## VCR-040 - Migration And Public Surface Docs

- [x] Update the beta.7 migration guide with canonical replacements.
- [x] Update compatibility audit notes from planned removal to removed.
- [x] Update public surface/import coverage that intentionally pinned the old deprecated names.

Validation:

- `cargo nextest run -p siumai --test public_surface_imports_test --features openai,google --no-default-features --no-fail-fast`
- `git diff --check`

## VCR-050 - Closeout

- [x] Run final focused gates.
- [x] Update evidence, milestone, handoff, and journal notes.
- [x] Close or split any residual follow-up.

Validation:

- Full focused gate list from `evidence-and-gates.md`.

Notes:

- `siumai-registry::factory_architecture_boundary_test::dedicated_vision_compatibility_surface_is_removed`
  now guards the removal.
- Historical docs and migration notes may still mention the removed names, but production core,
  registry, and spec sources cannot re-expose them.
- No residual follow-up is split from this workstream.
