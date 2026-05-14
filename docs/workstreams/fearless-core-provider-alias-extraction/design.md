# Fearless Core Provider Alias Extraction - Design

Last updated: 2026-05-14

## Context

`fearless-boundary-hardening` removed the largest protocol/provider leftovers from
`siumai-core`, narrowed broad re-exports, and made registry handles family-first.

One high-value coupling point remains: `siumai-core` still owns provider-specific model alias
normalization and model recommendation logic. That means a provider-agnostic crate still knows
about OpenRouter, DeepSeek, SiliconFlow, Fireworks, Together, model families, and provider-owned
model ids.

This workstream extracts that knowledge into the registry/provider boundary. During the fearless
refactor phase, removing redundant compatibility helpers is preferred over preserving them as
public utilities without an architectural home.

## Decision

`siumai-core` must not own provider-specific model aliases, model support tables, recommended
models, or OpenAI-compatible vendor routing rules.

Provider/model alias resolution belongs to `siumai-registry` and provider-owned metadata because
the registry already owns provider ids, built-in provider selection, provider records, default
models, and compatibility construction.

`siumai-core` may still carry provider-agnostic parameter primitives and report structures, but it
must not validate model names against provider-specific catalogs. Providers and registry factories
remain responsible for provider-specific defaults and validation.

## Goals

- Remove `siumai-core::utils::model_alias`.
- Remove `siumai_core::utils::builder_helpers::normalize_model_id`.
- Move current provider-specific alias behavior to registry-owned code.
- Keep `SiumaiBuilder` and OpenAI-compatible registry construction behavior intact after the move.
- Strip provider-specific model support/recommendation tables out of `siumai-core` parameter
  validation.
- Add boundary guards that prevent provider alias tables from returning to core utilities or core
  validators.
- Update docs so this extraction is explicit and reviewable.

## Non-goals

- Do not build a full model catalog system in this workstream.
- Do not change provider ids, provider catalogs, or public registry model handle syntax.
- Do not remove compatibility builder construction solely because it still exists.
- Do not rewrite provider factories outside the alias/validation boundary unless required by the
  extraction.

## Target Ownership

### `siumai-core`

Allowed:

- generic parameter sanity checks
- provider-agnostic report and error types
- shared builder helpers that do not contain provider/model catalogs
- provider ids as opaque strings when needed for diagnostics

Forbidden:

- provider-specific model aliases
- provider-specific model recommendation strings
- provider-specific model support predicates
- OpenAI-compatible vendor alias tables

### `siumai-registry`

Expected:

- provider id normalization
- provider inference from strongly-owned model prefixes
- provider/model alias normalization for compatibility construction
- source guards that keep registry/provider-owned alias logic out of core

### Provider Crates

Expected:

- provider-owned model constants and defaults
- provider-specific config validation where it is worth maintaining
- typed provider options and metadata

## Migration Policy

This is a breaking cleanup if external users imported `siumai_core::utils::normalize_model_id`.
That helper was never the architectural home for provider aliases. The canonical replacement is
registry/provider construction: pass the provider/model pair through the registry or provider
builder and let that layer normalize aliases.

No compatibility shim should be added back to core for this helper.

## Guardrails

- Keep alias behavior covered in registry tests before deleting the core helper.
- Keep core validator behavior conservative: validate generic numeric invariants only.
- Do not introduce provider crate dependencies into `siumai-core`.
- Do not revert unrelated worktree changes while extracting this boundary.

## Suggested Validation

```text
cargo fmt -p siumai-core -p siumai-registry -p siumai --check
cargo check -p siumai-core --no-default-features
cargo check -p siumai-registry --tests --features openai,deepseek,deepinfra,togetherai,google-vertex --no-default-features
cargo check -p siumai --tests --features openai,deepseek,deepinfra,togetherai,google-vertex --no-default-features
cargo nextest run -p siumai-core --test core_provider_boundary_test --no-fail-fast
cargo nextest run -p siumai-registry provider::resolver --features openai,deepseek,deepinfra,togetherai,google-vertex --no-default-features --no-fail-fast
git diff --check
```
