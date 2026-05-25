# Registry Typed Builder Isolation — Handoff

Status: Closed
Last updated: 2026-05-25

## Current State

The lane is closed. Production provider factories now call typed helper implementation through the
internal `registry::typed_builders` module. The legacy public `registry::factory` module retains
compatibility wrappers for the typed helper paths and deprecated generic-client helpers.

## Next Action

Do not reopen this lane for unrelated generic-client retirement. A future lane should start from a
separate ADR-0007 retirement gate or a concrete provider/family boundary.

## Validation To Re-run

```powershell
cargo nextest run -p siumai-registry --test factory_architecture_boundary_test production_factories_use_internal_typed_builders_not_legacy_factory_module --no-default-features --features openai,google,google-vertex,togetherai,deepinfra --no-fail-fast
cargo check -p siumai-registry --tests --all-features
```

## Constraints

- Do not delete deprecated public `registry::factory::build_*_client(...)` helpers in this lane.
- Keep compatibility wrappers until a separate breaking/public-surface decision removes them.
- Keep all-features validation for future edits because typed helper users are feature-gated.
