# Fearless Registry Facade Construction Boundary - TODO

Last updated: 2026-05-14

## Current Slice

- [x] Define the registry/facade construction boundary in a dedicated workstream.
- [x] Add a registry-owned helper for built-in provider factory selection.
- [x] Reuse the helper from default registry creation.
- [x] Reuse the helper from compatibility `SiumaiBuilder` construction.
- [x] Update public facade tests to use the helper instead of concrete built-in factory structs.
- [x] Add source guard coverage for the new boundary.
- [x] Run focused formatting, check, and nextest validation.

## Follow-up Candidates

- [ ] Decide whether `siumai::registry::factories` should become experimental-only in a future
      breaking cleanup.
- [ ] Audit `provider_public_path_parity_test.rs` for repeated local registry builder snippets and
      extract shared public-test helpers when the file is already being touched for that provider.
- [ ] Move provider-specific default model selection out of compatibility `SiumaiBuilder` and into
      registry/provider metadata once the construction helper is stable.
- [ ] Revisit `ProviderBuildOverrides` ergonomics for common test/custom-transport setup.

## Done Criteria

- Built-in factory selection has one registry-owned implementation.
- Facade tests that only need built-in providers do not import concrete factory types.
- Custom registry support remains intact.
- Focused checks pass.
