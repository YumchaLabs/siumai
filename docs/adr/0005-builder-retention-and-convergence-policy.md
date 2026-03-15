---
status: Proposed
date: 2026-03-06
---

# ADR 0005: Builder retention and convergence policy

## Context

The workspace is moving toward a model-family-centered architecture with:

- registry-first application construction
- config-first provider construction
- stable family-oriented function APIs

At the same time, the project already has builder-based entry points such as:

- `Siumai::builder()`
- `Provider::<provider>()`
- provider-specific builder types

Builders are valuable for ergonomics, but they currently create an architectural risk when they:

- encourage a separate mental model from registry/config-first flows
- hide provider construction logic behind builder-only code paths
- make it easy for features to land on builders first and nowhere else

The V4 refactor needs a clear policy so the project can preserve convenience without keeping builders
as the architectural center.

## Options considered

1) **Remove builders entirely**
- Pros: simplest architecture story; fewer entry points
- Cons: worse ergonomics; more verbose examples and tests; unnecessary migration cost

2) **Keep builders as a first-class primary architecture path**
- Pros: good discoverability and convenience
- Cons: preserves dual-path architecture; encourages builder-only behavior; slows convergence

3) **Keep builders, but make them thin wrappers over canonical config-first construction (chosen)**
- Pros: preserves ergonomics, keeps one internal construction path, reduces maintenance burden
- Cons: requires migration work and parity tests to ensure both paths behave the same

## Decision

We keep builders, but redefine their role.

Builders are retained as:

- ergonomic setup helpers
- quick-start and test-friendly construction APIs
- compatibility-friendly migration tools

Builders are **not** retained as:

- the primary architectural center
- a separate execution path
- a privileged place for new features

## Policy

### P1 - Config is canonical

Provider config structs are the canonical construction contract.

The internal construction chain should be:

```text
Builder -> Config -> Provider constructor -> Family model / provider object
```

Not:

```text
Builder -> private builder-only graph -> compatibility wrappers -> execution
```

### P2 - No builder-only features

New functionality must not be available only from builders.

If a setting or capability is important enough to expose publicly, it should be representable through:

- canonical config types, or
- request-level provider options, or
- stable provider extension APIs

### P3 - Public guidance is ranked

The recommended construction order for docs and examples is:

1. registry-first for application-level code
2. config-first for provider-specific code
3. builder-first for convenience and incremental setup

### P4 - Builders remain public during the refactor

Builders are not removed as part of the V4 architectural pivot.

However:

- they should not be described as the preferred architectural path
- they should be documented as convenience-oriented
- compatibility docs should point users to registry-first or config-first alternatives

## Consequences

- Builder APIs stay available for users who prefer progressive configuration.
- The internal architecture becomes simpler because construction converges on config-first paths.
- Tests must verify builder/config parity on major providers.
- Documentation must consistently rank construction styles and avoid presenting builder usage as the default story.

## Migration guidance

- Application examples should prefer registry-first construction.
- Provider-specific examples should prefer config-first construction.
- Keep a small number of builder examples for convenience and discoverability.
- Existing builder-heavy tests may remain temporarily, but new architecture work should not depend on builder-only behavior.

## Acceptance criteria

This decision is considered implemented when:

1. major providers support canonical config-first construction
2. builders compile down to the same underlying construction path
3. builder/config parity tests exist for major providers
4. docs clearly rank registry-first, config-first, and builder-first usage

