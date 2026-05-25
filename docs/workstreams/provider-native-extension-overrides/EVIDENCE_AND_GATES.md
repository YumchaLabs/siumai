# Provider Native Extension Overrides — Evidence And Gates

Status: Closed
Last updated: 2026-05-25

## Smallest Current Repro

```powershell
rg -n "file_management_capability_with_ctx|skills_capability_with_ctx|music_generation_capability_with_ctx|compat_language_client_with_ctx|ClientBacked(FileManagementCapability|SkillsCapability|MusicGenerationCapability)|as_(file_management_capability|skills_capability|music_generation_capability)" siumai-registry/src/registry
```

This shows whether selected provider factories return native extension trait objects or inherit the
default generic-client adapter path.

## Gate Set

### Registry Boundary Gate

```powershell
cargo nextest run -p siumai-registry --test factory_architecture_boundary_test provider_native_extension_hooks_bypass_generic_client_adapters --no-default-features --features openai,azure,anthropic,google,xai,minimaxi --no-fail-fast
```

### Registry Compile Gate

```powershell
cargo check -p siumai-registry --tests --no-default-features --features openai,azure,anthropic,google,xai,minimaxi
```

### Formatting / Diff Gate

```powershell
cargo fmt --package siumai-registry
git diff --check -- docs/workstreams/provider-native-extension-overrides docs/workstreams/INDEX.md siumai-registry/src/registry/factories siumai-registry/tests/factory_architecture_boundary_test.rs
```

## Candidate Inventory

Selected native extension hooks:

- Azure OpenAI: file management.
- OpenAI: file management and skills.
- Anthropic: file management and skills.
- Gemini: file management.
- xAI: file management.
- MiniMaxi: file management and music generation.

Deferred extension hooks:

- Speech extras and transcription extras remain outside this lane because current provider
  implementations often delegate through OpenAI-compatible generic-client extras rather than a
  dedicated provider-owned native extension object.
- Providers without a typed client implementation of the selected trait remain on the default
  compatibility fallback until a provider-specific native hook is proven.

## Evidence Log

| Date | Task | Command / Evidence | Result | What it proves |
| --- | --- | --- | --- | --- |
| 2026-05-25 | PNEO-010 | Workstream docs opened and `docs/workstreams/INDEX.md` corrected to include previously missing machine-readable lanes plus this active lane. | Pass | Establishes the follow-on lane and restores workstream navigation to the actual directory inventory. |
| 2026-05-25 | PNEO-020 | `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test provider_native_extension_hooks_bypass_generic_client_adapters --no-default-features --features openai,azure,anthropic,google,xai,minimaxi --no-fail-fast` before implementation. | Expected fail: missing `file_management_capability_with_ctx` in `azure.rs`. | Proves the guard exposes inherited default extension fallback before the native overrides are added. |
| 2026-05-25 | PNEO-030 | Azure OpenAI, OpenAI, Anthropic, Gemini, xAI, and MiniMaxi registry factories now project typed provider clients as `Arc<dyn FileManagementCapability>`, `Arc<dyn SkillsCapability>`, or `Arc<dyn MusicGenerationCapability>` for selected hooks. | Pass | Removes the selected extension paths from generic-client adapter fallback without changing provider runtime request behavior. |
| 2026-05-25 | PNEO-020/PNEO-030 | `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test provider_native_extension_hooks_bypass_generic_client_adapters --no-default-features --features openai,azure,anthropic,google,xai,minimaxi --no-fail-fast`. | Pass: 1 test run, 1 passed. | Guards that selected provider-native extension hooks do not call `compat_language_client_with_ctx(...)`, `ClientBacked*` adapters, or `as_*_capability()` downcasts. |
| 2026-05-25 | PNEO-030 | `cargo fmt --package siumai-registry`. | Pass | Formats the touched registry factory sources and boundary test. |
| 2026-05-25 | PNEO-030 | `cargo check -p siumai-registry --tests --no-default-features --features openai,azure,anthropic,google,xai,minimaxi`. | Pass | Proves the selected provider-native extension projections compile under the touched provider feature set. |
| 2026-05-25 | PNEO-040 | `git diff --check -- docs/workstreams/provider-native-extension-overrides docs/workstreams/INDEX.md` and `git diff --check -- siumai-registry/src/registry/factories siumai-registry/tests/factory_architecture_boundary_test.rs`. | Pass; Git reported the expected LF-to-CRLF working-copy warnings. | Proves docs and code diffs have no whitespace-error diff. |

## Residual Risks

- Default extension fallback still exists by design for custom providers and built-ins without a
  native extension hook.
- The selected factory overrides construct a provider client for each extension handle request,
  matching existing registry family construction behavior but not sharing handles.
- Broader deletion of generic-client compatibility remains governed by ADR-0007.

## Closeout Gate Matrix

| Area | Evidence | Result |
| --- | --- | --- |
| Source guard | `provider_native_extension_hooks_bypass_generic_client_adapters` under `openai,azure,anthropic,google,xai,minimaxi`. | Pass |
| Compile gate | `cargo check -p siumai-registry --tests --no-default-features --features openai,azure,anthropic,google,xai,minimaxi`. | Pass |
| Diff hygiene | `git diff --check` for touched docs, registry factories, and boundary test. | Pass |
