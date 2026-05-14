# Fearless Core Provider Alias Extraction - TODO

Last updated: 2026-05-14

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Planning

- [x] Create the workstream design document.
- [x] Create the workstream TODO document.
- [x] Create the workstream milestone document.
- [x] Link this workstream from the docs index.

## Track B - Alias Ownership

- [x] Move provider-specific model alias normalization from `siumai-core` to registry/provider-owned
  code.
  - OpenAI-compatible vendor alias logic now lives in
    `siumai-provider-openai-compatible::providers::openai_compatible::model_alias`.
  - `siumai-registry::provider::resolver::normalize_model_id` delegates to the provider-owned
    helper when OpenAI-compatible provider features are enabled.
- [x] Update `SiumaiBuilder` compatibility construction to call the registry-owned alias resolver.
- [x] Update OpenAI-compatible registry factory construction to call the registry-owned alias
  resolver.
- [x] Preserve current alias behavior for DeepSeek, SiliconFlow, Together, Fireworks, and
  OpenRouter compatibility paths.
- [x] Delete `siumai-core::utils::model_alias`.
- [x] Delete `siumai_core::utils::builder_helpers::normalize_model_id`.

## Track C - Core Validator Cleanup

- [x] Remove provider-specific model support predicates from `siumai-core::params::validator`.
- [x] Remove provider-specific recommended model strings from `siumai-core::params::validator`.
- [x] Keep `EnhancedParameterValidator` focused on generic numeric sanity checks.
- [x] Keep provider-specific validation responsibility in registry/provider-owned layers.

## Track D - Boundary Guards

- [x] Add core boundary tests that reject provider model alias tables in core utilities.
- [x] Add core boundary tests that reject provider-specific model recommendation strings in the
  core validator.
- [x] Add provider-owned and registry resolver tests for the moved alias resolver.

## Track E - Validation

- [x] Run package-scoped formatting checks for touched crates.
- [x] Run focused `cargo check` for `siumai-core`.
- [x] Run focused `cargo check` for `siumai-registry` and facade tests with affected provider
  features.
- [x] Run focused `cargo nextest` boundary and resolver tests.
- [x] Run `git diff --check`.
