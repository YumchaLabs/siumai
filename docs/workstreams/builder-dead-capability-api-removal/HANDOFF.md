# Builder Dead Capability API Removal — Handoff

Status: Closed
Last updated: 2026-05-25

## Current State

The lane is closed. `SiumaiBuilder` no longer stores capability strings and no longer exposes the
write-only capability helper methods.

## Next Action

Do not reopen this lane for unrelated compatibility cleanup. A future lane should start from a
separate public contract or provider/family boundary.

## Validation To Re-run

```powershell
cargo nextest run -p siumai-registry --test builder_architecture_boundary_test builder_does_not_expose_noop_capability_flags --no-default-features --features openai --no-fail-fast
cargo check -p siumai-registry --tests --no-default-features --features openai
```

## Constraints

- Remove only builder capability flags. Do not change `ProviderCapabilities` or provider metadata.
- Keep future validation feature sets narrow unless a change touches provider-specific gates.
