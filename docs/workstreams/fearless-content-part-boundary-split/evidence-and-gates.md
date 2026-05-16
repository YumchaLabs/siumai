# Fearless ContentPart Boundary Split - Evidence And Gates

Last updated: 2026-05-16

## Evidence Anchors

- Existing direct construction audit:
  `docs/workstreams/fearless-spec-core-boundary-convergence/content-part-construction-audit.md`
- Spec request/response projection guard:
  `siumai-spec/tests/content_projection_boundary_test.rs`
- Facade audit guard:
  `siumai/tests/facade_architecture_boundary_test.rs`
- Core provider-map guard:
  `siumai-core/tests/core_provider_boundary_test.rs`

## Required Gates

- `cargo fmt --package siumai-spec --package siumai-core --package siumai-bridge --package siumai --check`
- `cargo nextest run -p siumai-spec --test content_projection_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `git diff --check`

## Validation Log

- Pending.
