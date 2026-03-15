# ADR 0007 - LlmClient demotion policy

- Status: accepted
- Date: 2026-03-06

## Context

`LlmClient` has been a useful compatibility umbrella.
It allowed one object to expose many optional capabilities while the library was still shaping its provider story.

However, as the project moves toward family-model-first execution, treating `LlmClient` as the primary public abstraction creates long-term problems:

- it encourages broad optional capability surfaces
- it blurs which family contract is actually stable
- it makes registry and provider migration harder to reason about
- it encourages new features to land in the compatibility layer instead of the family layer

## Decision

`LlmClient` is demoted from architectural center to compatibility abstraction.

This means:

1. New architecture work should target family model traits first.
2. `ProviderFactory` should evolve toward returning family model objects directly.
3. Registry handles should evolve into family model objects directly.
4. `LlmClient` remains available during migration for backward compatibility and bridge paths.
5. New features must not be added only to `LlmClient` if a family-level home exists.

## Consequences

### Positive

- Clarifies what the real stable execution contracts are.
- Makes provider migration sequencing easier to understand.
- Reduces compatibility-surface sprawl.
- Encourages provider-native implementations to converge on family contracts.

### Negative

- Some duplicate glue exists during the transition.
- We must keep bridge adapters and parity tests for a while.
- A few old call sites will remain generic-client-centered until handle migration is finished.

## Allowed uses of `LlmClient`

`LlmClient` is still acceptable for:

- backward compatibility
- bridge adapters during migration
- registry internals that have not yet moved to family-native execution
- capability discovery in compatibility-focused surfaces

## Disallowed direction

We should avoid:

- adding new provider features only behind `LlmClient`
- making `LlmClient` the recommended public example surface in new docs
- blocking family-trait evolution because of compatibility-only constraints

## Implementation guidance

1. Keep `LlmClient` bridge paths working while providers migrate.
2. Add native family-returning factory methods before deleting generic paths.
3. Use contract tests to compare old and new construction paths during migration.
4. Move registry handle execution off the generic client path incrementally.
5. Reassess removal or stronger deprecation only after major providers and handles are migrated.
