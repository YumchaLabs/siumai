# Fearless Boundary Hardening - Design

Last updated: 2026-05-13

## Context

The previous `fearless-architecture-convergence` workstream closed the first cleanup pass:

- protocol compatibility modules moved out of `siumai-core`
- `ProviderFactory` became family-first in shape and documentation
- registry responsibilities were split into focused modules
- facade provider-extension bodies moved out of `siumai/src/lib.rs`
- bridge implementation moved into `siumai-bridge`
- deprecated compatibility surfaces were categorized

The next pass should not reopen that completed workstream. This workstream is the follow-up
execution track for hardening the boundaries that now exist and deleting compatibility or redundant
code that no longer carries real migration value.

This is a fearless refactor track. When a compatibility artifact is not part of a documented public
promise, has a canonical replacement, and is covered by tests or migration notes, removal is the
preferred outcome.

## Decision

Siumai is still in a fearless-refactor phase. The default bias for this workstream is:

- remove redundant compatibility code instead of preserving it indefinitely
- delete dead wrappers and aliases once the canonical path is tested
- move provider/protocol-specific logic out of provider-agnostic crates
- tighten public exports so stable paths stay small and intentional
- prefer source guards and focused tests over preserving historical structure

Compatibility remains valuable only when it is documented, time-bounded, and tied to a concrete
migration story.

## Goals

- Harden crate boundaries so new code cannot drift back into old coupling patterns.
- Remove `siumai-core` protocol/provider-specific leftovers that can live in protocol or provider
  crates.
- Reduce broad `pub use siumai_core::{...}` re-export patterns in provider and protocol crates.
- Keep `siumai` as a facade, not a second implementation center.
- Make `LlmClient` and builder-style construction visibly compatibility-only.
- Delete deprecated aliases, duplicate helpers, and compatibility wrappers whose replacements are
  already canonical.
- Keep docs, examples, and tests aligned with the family-first architecture.

## Non-goals

- Do not preserve compatibility for undocumented or unused paths simply because they existed before.
- Do not rewrite provider implementations for cosmetic symmetry alone.
- Do not introduce broad formatting churn as a substitute for architectural cleanup.
- Do not move gateway/server concerns into `siumai-core`.
- Do not add a new abstraction unless it removes coupling, duplication, or an unstable public path.

## Target Boundaries

### `siumai-spec`

Owns serializable provider-agnostic data contracts only.

Forbidden:

- HTTP execution
- retry/runtime policy
- provider construction
- provider-specific wire schemas

### `siumai-core`

Owns provider-agnostic runtime primitives:

- family traits
- shared errors
- HTTP abstractions
- retry and middleware contracts
- generic streaming carriers
- provider-agnostic tool/runtime utilities

Forbidden:

- provider-specific URL normalization
- provider-specific request/response transformers
- provider-specific wire structs
- provider-specific typed options or metadata
- provider registries for OpenAI-compatible vendors

### `siumai-protocol-*`

Owns reusable protocol mapping and wire-format conversion.

Expected:

- request and response transformers
- protocol stream parsing and serialization
- protocol-owned warning and metadata extraction helpers
- shared OpenAI-compatible vendor protocol utilities

### `siumai-provider-*`

Owns provider clients, builders, provider-specific options, metadata, resources, and extensions.

Expected:

- provider-owned typed option and metadata helpers
- provider-specific endpoint clients
- provider-specific capability implementations
- narrow compatibility exports only when documented

### `siumai-registry`

Owns lookup, provider factories, build context, and model-handle caching.

Expected:

- family-first factory construction
- compatibility adapters isolated in explicit compatibility modules
- no stable family handle execution through `LlmClient` downcasts

### `siumai`

Owns the facade:

- stable public modules and preludes
- feature aggregation
- provider extension exports
- explicit compatibility entry points

Expected:

- no heavy protocol conversion logic
- no provider extension implementation bodies in `lib.rs`
- no new stable-prelude exports for compatibility-only names

## Removal Policy

Remove a compatibility or redundant artifact when all of the following are true:

- a canonical replacement exists
- current docs or examples no longer recommend the old path
- tests or source guards cover the replacement boundary
- the old artifact is not part of a documented removal window, or the removal window has been
  reached

Keep a compatibility artifact only when at least one of the following is true:

- it is explicitly documented in a migration guide
- it is needed for a published source-compatibility window
- it supports an extension-only capability that has no family-first replacement yet
- removing it would silently break a stable public promise

## Guardrails

- Add source guards before large removals when the old pattern is easy to reintroduce.
- Prefer small, reviewable deletion slices.
- Keep migration notes close to public API removals.
- Do not revert unrelated user changes while cleaning up.
- Validate affected crates with focused `cargo nextest` runs.

## Suggested Validation

Use focused validation as slices land:

```text
cargo nextest run -p siumai-core --no-fail-fast
cargo nextest run -p siumai-protocol-openai --features openai-standard --no-fail-fast
cargo nextest run -p siumai-registry --no-default-features --no-fail-fast
cargo nextest run -p siumai --no-default-features --features openai --no-fail-fast
cargo nextest run -p siumai-bridge --no-default-features --features "openai,anthropic,google" --no-fail-fast
```

For provider/protocol boundary changes, also run the affected provider crates:

```text
cargo nextest run -p siumai-provider-openai -p siumai-provider-openai-compatible --features openai-standard --no-fail-fast
cargo nextest run -p siumai-provider-anthropic -p siumai-protocol-anthropic --features anthropic-standard --no-fail-fast
```
