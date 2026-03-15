# ADR 0006 - Family-model-first trait policy

- Status: accepted
- Date: 2026-03-06

## Context

The V4 refactor needs a stable architectural center.
The old generic-client-centered layering helped us bootstrap quickly, but it keeps the execution model too broad.
It also makes it harder to align construction, registry handles, middleware boundaries, and provider-specific implementations.

We want to learn from the Vercel AI SDK implementation strategy:

- stable family-oriented contracts at the top
- provider-specific complexity below those contracts
- shared construction/context flow across providers

At the same time, `siumai` should remain Rust-first in naming and public API shape.
We should not mirror AI SDK naming mechanically.

## Decision

We adopt a family-model-first trait policy for V4.

That means:

1. Family model traits become the primary execution contracts.
2. `text`, `embedding`, `image`, `rerank`, `speech`, and `transcription` are the preferred public families.
3. Shared model identity belongs in a lightweight metadata trait rather than being repeated across every family trait.
4. Legacy capability traits remain as migration shims, not as the long-term architectural center.
5. New provider work should target family-model-native implementations first.

The current minimum shared metadata contract is:

- `ModelMetadata`
- `ModelSpecVersion`

The current first landed family contract is:

- `LanguageModel`

## Consequences

### Positive

- Gives the refactor a clear center of gravity.
- Keeps provider-specific behavior below the stable trait boundary.
- Makes registry handles easier to evolve into first-class model objects.
- Reduces the chance that new features only land in compatibility-era abstractions.
- Preserves freedom to keep Rust-idiomatic naming.

### Negative

- Migration is incremental and temporarily duplicates some execution paths.
- Some providers will still bridge through legacy capabilities until their native family paths land.
- Documentation and tests must explicitly describe what is final versus transitional.

## Policy details

### What we preserve

- Rust-first naming
- provider extension traits where provider-specific extras matter
- builder ergonomics as convenience wrappers
- spec-level types where current names are still appropriate

### What we avoid

- making family traits inherit from legacy capability traits
- introducing new features only through `LlmClient`
- renaming public types just to match another SDK verbatim

## Implementation guidance

1. Add minimal family traits first, then grow them carefully.
2. Use adapters from legacy capability traits during migration.
3. Prefer native provider family objects once the provider surface is stable enough.
4. Keep no-network contract tests for every migrated family path.
5. Update architecture docs whenever a provider crosses from bridge mode to native mode.
