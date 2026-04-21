# Prompt Call Settings Alignment - TODO

Last updated: 2026-04-21

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Shared compatibility surface

- [x] Audit `repo-ref/ai/packages/ai/src/prompt/index.ts` for the remaining small compatibility
  exports after the earlier request/call-option work.
- [x] Add deprecated `CallSettings` on the shared Rust facade.
- [x] Keep `CallSettings` limited to `LanguageModelCallOptions + Omit<RequestOptions, 'timeout'>`.
- [x] Add free timeout helper functions over `TimeoutConfiguration`.

## Track B - Facade and tests

- [x] Re-export `CallSettings` and the timeout helper functions from the stable public facade.
- [x] Add compile-guard coverage in `siumai/tests/public_surface_imports_test.rs`.
- [x] Add focused unit coverage in `siumai-spec/src/types/ai_sdk.rs`.

## Track C - Docs and changelog

- [x] Create a dedicated `docs/workstreams/prompt-call-settings-alignment/` folder.
- [x] Record this slice in `CHANGELOG.md` `Unreleased`.

## Track D - Intentional deferrals

- [-] Do not treat `CallSettings` as the new primary request contract.
- [-] Do not fold the larger prompt-message/content shared-contract audit into this small helper
  workstream.
