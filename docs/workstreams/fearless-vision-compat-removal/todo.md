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

- [ ] Add or update source guard coverage proving the dedicated vision compatibility family is
      removed from production public surfaces.
- [ ] Keep the guard narrow enough to allow historical migration notes and audit text.

Validation:

- `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --features openai,google --no-default-features --no-fail-fast`

## VCR-030 - Remove Vision Compatibility Code

- [ ] Remove `VisionCapability` from core traits and `LlmClient` downcast methods.
- [ ] Remove `VisionCapabilityProxy` and `Siumai::vision_capability()`.
- [ ] Remove provider/factory forwarding impls that only delegate the deprecated vision downcast.
- [ ] Remove deprecated request/response type aliases that only existed for the vision trait.

Validation:

- `cargo check -p siumai-core --no-default-features`
- `cargo check -p siumai-registry --tests --features openai,google --no-default-features`
- `cargo check -p siumai --tests --features openai,google --no-default-features`

## VCR-040 - Migration And Public Surface Docs

- [ ] Update the beta.7 migration guide with canonical replacements.
- [ ] Update compatibility audit notes from planned removal to removed.
- [ ] Update public surface/import coverage that intentionally pinned the old deprecated names.

Validation:

- `cargo nextest run -p siumai --test public_surface_imports_test --features openai,google --no-default-features --no-fail-fast`
- `git diff --check`

## VCR-050 - Closeout

- [ ] Run final focused gates.
- [ ] Update evidence, milestone, handoff, and journal notes.
- [ ] Close or split any residual follow-up.

Validation:

- Full focused gate list from `evidence-and-gates.md`.
