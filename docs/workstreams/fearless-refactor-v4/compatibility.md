# Fearless Refactor V4 - Compatibility Policy

Last updated: 2026-03-06

## Purpose

This document defines the compatibility contract for the V4 refactor.
Its goal is to keep the public story coherent while allowing large internal changes.

For the public-facing surface ladder (`registry-first -> config-first -> builder convenience`) and where
`provider_ext` / `compat` belong, see `public-api-story.md`.

## Compatibility layers

The public and semi-public API surface is divided into four layers.

### 1. Stable

Stable APIs are the recommended long-term entry points.

Properties:

- documented in README and architecture docs
- intended for broad downstream usage
- breaking changes require migration notes
- naming should remain Rust-first and consistent

Examples:

- model-family modules such as `siumai::text` and `siumai::embedding`
- registry-first construction APIs
- provider-agnostic request/response/message/tool types in `siumai-spec`

### 2. Extension

Extension APIs are public and supported, but intentionally provider-specific or feature-specific.

Properties:

- long-lived public surface
- not guaranteed to be provider-agnostic
- may vary by provider capabilities
- should not be promoted as the default path for general use

Examples:

- `siumai::provider_ext::<provider>`
- provider-specific metadata helpers
- provider-specific hosted tools, hosted-search, and resource modules

### 3. Compatibility

Compatibility APIs exist to preserve migration continuity.

Properties:

- still public for a transition period
- not recommended for new code
- may remain temporarily even after internal rewrites
- new features should not land here first

Examples:

- legacy builder-centric wrappers
- `siumai::compat`
- generic-client-oriented convenience shims

### 4. Experimental

Experimental APIs are explicitly low-stability.

Properties:

- suitable for advanced users only
- may change without long deprecation windows
- should not become the dependency anchor for mainstream examples

Examples:

- low-level execution internals
- special transport hooks under experimental modules
- prototype provider-facing building blocks

## Compatibility rules

## Documentation ordering rule

When the docs show multiple construction paths, they should be ordered as:

1. registry-first
2. config-first
3. builder convenience

This keeps ergonomic examples available without implying that builders are the architectural center.

## Rule C1 - Stable APIs must have one recommended story

Stable docs must not present multiple competing primary entry points.

V4 default story:

- app-level code: registry-first
- provider-specific code: config-first
- quick setup: builder convenience

## Rule C2 - Builders remain public, but are compatibility-plus-convenience

Builders are not deprecated by default in V4.
However, they are not the architectural center.

Policy:

- builder APIs may remain public
- builder APIs must map to the same canonical config-first path
- builder-only features are not allowed
- examples should not over-emphasize builders as the default architecture

## Rule C3 - `LlmClient` is not the default public execution abstraction

If `LlmClient` remains public during migration, it must be documented as low-level or transitional.

Policy:

- public family APIs should not require users to understand `LlmClient`
- registry handles should be valid family model objects
- provider migration may use `LlmClient` internally during transition

## Rule C4 - Stable spec types should not be renamed casually

We do not rename spec types solely to mirror AI SDK naming.

Policy:

- retain existing well-shaped names such as `ChatRequest`, `ChatMessage`, `ContentPart`, `TtsRequest`, and `SttRequest`
- only rename when there is a strong semantic mismatch or long-term maintenance cost

## Rule C5 - New features must land on the target architecture first

If a feature is added during migration, prefer:

1. stable family-model path
2. extension path if provider-specific
3. compatibility shim only if necessary

Avoid:

- adding features only to builder APIs
- adding features only to legacy capability traits
- adding features only to generic client shims

## Deprecation expectations

This workstream does not require immediate removal of compatibility APIs.
It does require a clear ranking of surfaces.

Recommended wording:

- Stable: recommended for new code
- Extension: recommended for provider-specific work
- Compatibility: supported for migration, not recommended for new code
- Experimental: unstable, advanced use only

## Documentation requirements by layer

### Stable

- must appear in top-level docs
- must have examples
- must have migration guidance when behavior changes

### Extension

- must have scoped documentation under provider-specific sections
- should explain when to choose extension APIs over stable ones

### Compatibility

- must be labeled clearly
- must point to the preferred replacement path

### Experimental

- must carry a stability warning
- should avoid top-level beginner documentation

## Release-gating policy

Before V4 architecture is announced as the default:

- stable family-model path must be complete for major providers
- builders must be converged onto config-first internals
- the docs must rank API surfaces consistently

## Success criteria

This policy is working if:

1. users can tell which APIs are preferred without reading source code
2. builders remain ergonomic without trapping the architecture
3. provider-specific growth continues under extension modules
4. compatibility APIs no longer distort the default mental model
