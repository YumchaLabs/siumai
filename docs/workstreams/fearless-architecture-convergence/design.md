# Fearless Architecture Convergence - Design

Last updated: 2026-05-13

## Context

Siumai already has the right public direction:

- application code should prefer registry-first construction
- provider-specific code should prefer config-first construction
- high-level calls should target model-family APIs
- provider-specific features should live in provider crates and `provider_ext`

The remaining problem is not lack of direction. The problem is that several old architectural
centers still exist in code:

- `LlmClient` is still a broad compatibility umbrella with many optional capability downcasts
- `ProviderFactory` still keeps generic-client construction beside family-model construction
- OpenAI-compatible protocol ownership has moved out of `siumai-core`; the remaining work is to
  prevent new reverse imports and keep provider/facade compatibility aliases explicit
- `siumai-registry` still mixes registry lookup, build-context resolution, caching, and built-in
  provider construction in the same large modules
- the `siumai` facade still carries compatibility entry points, but heavy provider-extension and
  bridge implementation logic has moved out of the crate root

This workstream is the cleanup pass after V4: remove redundant compatibility code, collapse parallel
execution paths, and enforce the target layering already documented by the ADRs.

## Goals

- Make model-family traits the only architectural execution center.
- Move protocol-owned OpenAI-compatible modules out of `siumai-core`.
- Reduce `LlmClient` to a compatibility bridge instead of a primary construction target.
- Split oversized registry/facade modules along stable ownership boundaries.
- Remove stale aliases, stale comments, dead compatibility wrappers, and duplicate helpers when
  tests prove the new path is canonical.
- Keep public guidance and tests honest: no new feature may land only in a compatibility path.

## Non-goals

- Do not rewrite provider implementations merely for naming symmetry.
- Do not remove public compatibility APIs without an explicit migration note or documented removal
  version.
- Do not fabricate TypeScript-style callable provider objects where Rust already has a better
  config-first construction shape.
- Do not move gateway/server concerns into `siumai-core`.

## Target Architecture

The target dependency direction is:

```text
siumai facade
  -> siumai-registry / siumai-bridge
  -> siumai-provider-* / siumai-protocol-*
  -> siumai-core
  -> siumai-spec
```

The target execution direction is:

```text
high-level family helper
  -> family model trait
  -> provider-owned implementation
  -> protocol adapter / HTTP execution
```

The legacy execution direction to remove is:

```text
high-level helper
  -> LlmClient
  -> optional capability downcast
  -> provider implementation
```

`LlmClient` can remain during migration, but only as:

- compatibility surface
- legacy adapter source
- downcast bridge for extension-only APIs not yet modeled as families

## Ownership Rules

### `siumai-spec`

Owns serializable provider-agnostic data contracts.

Must not own:

- HTTP execution
- provider-specific protocol mapping
- provider construction

### `siumai-core`

Owns provider-agnostic runtime primitives:

- family traits
- shared error and HTTP abstractions
- streaming carriers and generic stream processing
- retry and middleware contracts
- provider-agnostic utilities

Must not own:

- OpenAI-compatible provider registries
- provider-specific URL normalization
- provider-specific request/response transformers
- typed provider option or metadata structs

### `siumai-protocol-*`

Owns wire-format mapping and reusable protocol adapters.

OpenAI-compatible protocol modules should converge in `siumai-protocol-openai`, not in
`siumai-core`.

### `siumai-registry`

Owns model lookup, provider factories, build-context propagation, and model-handle caching.

Long-term registry handles should implement family model traits directly and should not need
generic-client downcasts for stable model families.

### `siumai`

Owns the stable facade:

- public modules and preludes
- feature aggregation
- provider extension exports
- compatibility entry points

It should not own heavy protocol conversion logic. Gateway/protocol bridge code should move toward
`siumai-extras` or a dedicated bridge crate if it remains important.

## Refactor Strategy

### 1. Convert family factories from parallel path to primary path

Current `ProviderFactory` has both generic-client and family-returning methods. The family methods
must become the primary required contract. Generic-client builders become default adapters only
where compatibility still needs them.

### 2. Move OpenAI-compatible protocol modules out of core

The provider-specific core island was:

```text
siumai-core/src/standards/openai/compat/*
```

It has moved into:

```text
siumai-protocol-openai/src/standards/openai/compat/*
```

`siumai-core` no longer exposes `standards::openai::compat`. Any temporary compatibility exports
must live at facade or provider boundaries, where their migration cost is visible.

### 3. Split registry responsibilities

Large files such as `registry/entry.rs` should be split by responsibility:

- factory trait and build context
- provider registry handle
- family model handles
- compatibility `LlmClient` adapters
- caches and cache keys
- tests by family

The goal is not smaller files for aesthetics. The goal is to make old compatibility paths visible
and removable.

### 4. Slim the facade

Move provider-extension module bodies out of `siumai/src/lib.rs`. The crate root should mostly
declare modules and re-export stable surfaces.

`experimental_bridge` is gateway/protocol integration code, not facade glue. Its implementation now
lives in `siumai-bridge`. The `siumai` facade keeps the historical
`siumai::experimental::bridge::*` path as a compatibility re-export, while `siumai-extras` consumes
`siumai-bridge` directly. This keeps `siumai-core` provider-agnostic and avoids a facade/extras
dependency cycle.

### 5. Delete redundant compatibility artifacts

Remove artifacts only after the canonical replacement exists and is tested:

- stale comments that refer to already-deleted types
- deprecated aliases that are not part of the documented compatibility promise
- compatibility wrappers with no semantic difference from shared types
- duplicate helper functions split between core and protocol/provider crates

## Guardrails

- Every deletion should have one of:
  - direct compile coverage
  - no-network contract coverage
  - explicit migration documentation
- Prefer small deletion slices over huge mechanical moves.
- Do not remove user-facing compatibility APIs silently.
- Do not use broad formatting churn as a substitute for architectural cleanup.

## Validation

Use focused commands as slices land:

```text
cargo nextest run -p siumai-core
cargo nextest run -p siumai-protocol-openai
cargo nextest run -p siumai-registry
cargo nextest run -p siumai
cargo nextest run -p siumai-bridge
cargo fmt --check
```

For broad module moves, run the affected provider crates too:

```text
cargo nextest run -p siumai-provider-openai-compatible -p siumai-provider-openai
cargo nextest run -p siumai-provider-groq -p siumai-provider-xai -p siumai-provider-deepseek
```
