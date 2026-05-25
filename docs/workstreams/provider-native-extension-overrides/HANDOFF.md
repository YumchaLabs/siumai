# Provider Native Extension Overrides — Handoff

Status: Closed
Last updated: 2026-05-25

## Current State

The lane is closed. Provider-native file, skills, and music extension hooks were implemented and
guarded for the selected built-in providers.

## Next Action

Do not continue this lane for unrelated extension cleanup. Open a new narrow workstream if
speech/transcription extras gain provider-owned native extension clients or if ADR-0007 deletion
preconditions are ready.

## Validation To Re-run

```powershell
cargo nextest run -p siumai-registry --test factory_architecture_boundary_test provider_native_extension_hooks_bypass_generic_client_adapters --no-default-features --features openai,azure,anthropic,google,xai,minimaxi --no-fail-fast
cargo check -p siumai-registry --tests --no-default-features --features openai,azure,anthropic,google,xai,minimaxi
```

## Notes

- Do not reopen `docs/workstreams/native-extension-and-compat-retirement/`; it is closed and points
  here for additional provider-native extension overrides.
- Keep speech/transcription extras out unless a provider-owned native object is proven.
