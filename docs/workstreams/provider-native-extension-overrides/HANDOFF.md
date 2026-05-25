# Provider Native Extension Overrides — Handoff

Status: Active
Last updated: 2026-05-25

## Current State

The lane is being opened to remove generic-client adapter fallback from provider-native file,
skills, and music extension hooks.

## Next Action

Run PNEO-020:

1. Add `provider_native_extension_hooks_bypass_generic_client_adapters` to
   `siumai-registry/tests/factory_architecture_boundary_test.rs`.
2. Make it name the selected factory methods and reject `compat_language_client_with_ctx(...)`,
   `as_*_capability()`, and `ClientBacked*` adapter markers.
3. Implement PNEO-030 by returning typed provider clients as extension trait objects.

## Validation To Re-run

```powershell
cargo nextest run -p siumai-registry --test factory_architecture_boundary_test provider_native_extension_hooks_bypass_generic_client_adapters --no-default-features --features openai,azure,anthropic,google,xai,minimaxi --no-fail-fast
cargo check -p siumai-registry --tests --no-default-features --features openai,azure,anthropic,google,xai,minimaxi
```

## Notes

- Do not reopen `docs/workstreams/native-extension-and-compat-retirement/`; it is closed and points
  here for additional provider-native extension overrides.
- Keep speech/transcription extras out unless a provider-owned native object is proven.
