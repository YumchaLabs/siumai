# Fearless Core Provider Alias Extraction - Milestones

Last updated: 2026-05-14

## FCPA-M0 - Workstream Locked

Acceptance criteria:

- Design, TODO, and milestone documents exist.
- Docs index links to this workstream.
- Scope is explicit: provider model aliases leave `siumai-core`.

Status: done

Notes:

- Workstream docs were added under `docs/workstreams/fearless-core-provider-alias-extraction/`.
- `docs/README.md` now links this workstream.

## FCPA-M1 - Registry Owns Model Alias Resolution

Acceptance criteria:

- Alias normalization behavior lives under `siumai-registry`.
- `SiumaiBuilder` compatibility construction uses the registry-owned resolver.
- OpenAI-compatible registry factory construction uses the registry-owned resolver.
- Existing behavior is covered by registry tests.

Status: done

Notes:

- OpenAI-compatible provider alias behavior moved to provider-owned
  `siumai-provider-openai-compatible::providers::openai_compatible::model_alias`.
- `siumai-registry::provider::resolver::normalize_model_id` delegates to the provider-owned helper
  for OpenAI-compatible feature sets.
- `SiumaiBuilder` compatibility construction now calls the registry resolver instead of
  `siumai-core`.
- OpenAI-compatible registry factory construction now calls the registry resolver instead of
  `siumai-core`.
- Alias tests cover DeepSeek, SiliconFlow, Together, Fireworks, and OpenRouter behavior.

## FCPA-M2 - Core Alias Helper Removed

Acceptance criteria:

- `siumai-core/src/utils/model_alias.rs` is deleted.
- `siumai-core::utils` no longer exports model alias helpers.
- `builder_helpers` no longer exposes provider-specific model normalization.
- Core source guards prevent the helper from returning.

Status: done

Notes:

- `siumai-core/src/utils/model_alias.rs` was deleted.
- `siumai-core::utils` no longer declares or exports `model_alias`.
- `siumai_core::utils::builder_helpers::normalize_model_id` was removed.
- Core URL/helper examples were adjusted away from provider-specific model ids where they were only
  generic examples.

## FCPA-M3 - Core Validator Is Provider-Agnostic

Acceptance criteria:

- Core parameter validation no longer contains provider model support tables.
- Core parameter validation no longer emits provider-specific recommended model strings.
- Provider-specific model validation remains a provider/registry concern.
- Core validator tests still cover generic numeric validation.

Status: done

Notes:

- Provider-specific model support predicates and recommended model strings were removed from
  `siumai-core::params::validator`.
- `EnhancedParameterValidator` now validates generic numeric invariants plus empty stop sequence
  sanity only.
- Provider-specific model validation remains with provider/config layers.

## FCPA-M4 - Validation Complete

Acceptance criteria:

- Focused formatting checks pass.
- Focused `cargo check` passes for affected crates.
- Focused `cargo nextest` runs pass for boundary and resolver tests.
- `todo.md` has no incomplete tasks except explicitly deferred items.

Status: done

Notes:

- `git diff --check` reported only CRLF conversion warnings from the Windows worktree, with no
  whitespace errors.
- Verified commands:
  - `cargo fmt -p siumai-core -p siumai-provider-openai-compatible -p siumai-registry -p siumai --check`
  - `cargo check -p siumai-core --no-default-features`
  - `cargo check -p siumai-core --tests --no-default-features`
  - `cargo check -p siumai-provider-openai-compatible --features openai-standard --no-default-features`
  - `cargo check -p siumai-registry --tests --features openai,deepseek,deepinfra,togetherai,google-vertex --no-default-features`
  - `cargo check -p siumai --tests --features openai,deepseek,deepinfra,togetherai,google-vertex --no-default-features`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-provider-openai-compatible providers::openai_compatible::model_alias --features openai-standard --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-registry provider::resolver --features openai,deepseek,deepinfra,togetherai,google-vertex --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test public_surface_imports_test --features openai,deepseek,deepinfra,togetherai,google-vertex --no-default-features --no-fail-fast`
  - `git diff --check`
