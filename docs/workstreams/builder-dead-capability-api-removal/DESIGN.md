# Builder Dead Capability API Removal

Status: Closed
Last updated: 2026-05-25

## Why This Lane Exists

`SiumaiBuilder` still carries a historical `capabilities: Vec<String>` field and public helpers such
as `with_capability()`, `with_audio()`, `with_embedding()`, and `with_image_generation()`. These
methods look like they influence the provider family or extension selected by `build()`, but the
current registry-owned construction path does not read them.

That mismatch is worse than ordinary legacy surface area: it teaches callers that provider
capabilities are opt-in builder flags, while the architecture now treats capabilities as facts owned
by provider factories and registry metadata.

## Relevant Authority

- `docs/adr/0007-llmclient-demotion-policy.md`
- `docs/adr/0006-family-model-first-trait-policy.md`
- `docs/architecture/public-surface.md`
- `docs/workstreams/provider-native-extension-overrides/`

## Problem

The builder field and helpers are write-only:

- `SiumaiBuilder::new()` initializes `capabilities` to an empty vector.
- `with_capability()` and the named helpers only append strings to that vector.
- `build()` chooses the provider and default family from the provider factory's
  `ProviderCapabilities`, not from builder-supplied capability strings.
- the only remaining read is a `Debug` count, which exposes storage without behavior.

The result is a silent no-op compatibility layer. Removing it makes the public construction story
more honest: select a provider/family through explicit providers, registry handles, or extension
surfaces instead of annotating a generic builder with unused strings.

## Target State

- `SiumaiBuilder` no longer stores caller-supplied capability strings.
- `with_capability()`, `with_audio()`, `with_embedding()`, and `with_image_generation()` are removed
  from the public builder API.
- A source-level guard rejects reintroduction of these no-op methods or storage.
- Migration notes point users to provider factory metadata, stable registry handles, or explicit
  extension APIs instead of builder capability flags.

## In Scope

- `siumai-registry/src/provider/siumai_builder.rs`.
- Focused public-surface or source-guard tests for the removed no-op API.
- Changelog and workstream documentation for the breaking cleanup.
- Workstream index updates.

## Out Of Scope

- Changing provider factory capability semantics.
- Changing stable family handles such as `language_model()`, `embedding_model()`, or
  `image_model()`.
- Removing `ProviderCapabilities` or its fluent `with_*` methods.
- Removing compatibility layers unrelated to builder capability flags.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Builder capability strings are not consumed during provider construction. | High | `build()` reads factory capabilities and never reads `builder.capabilities`. | If a hidden path is found, replace removal with a typed routing design or split the consumer into the lane. |
| These helpers were already deprecated or documented as historical compatibility surface. | Medium | `CHANGELOG.md` references deprecated `with_capability()`, `with_audio()`, and `with_embedding()` methods. | Add a migration note before deletion and keep the source guard focused on no-op behavior. |
| Removing no-op builder flags is acceptable under the current fearless refactor policy. | High | The repository is converging toward provider/family-owned construction and away from generic builder knobs. | If release compatibility becomes the priority, convert methods into compile-time deprecations first. |

## Architecture Direction

Capabilities should be provider-owned facts, not caller-authored strings. Provider metadata answers
what a provider supports, registry handles select a family, and extension modules expose
non-unified capabilities. Keeping unused builder capability flags creates a second, false control
plane.

This lane removes that false control plane and guards the boundary so future capability work lands
in the registry or provider factory layer where it can be validated.

## Closeout Condition

This lane can close when:

- the no-op field and public methods are removed;
- the guard test and focused compile gates pass;
- migration notes describe the supported alternatives; and
- evidence is recorded in `EVIDENCE_AND_GATES.md`.

## Closeout Summary

Closed on 2026-05-25.

`SiumaiBuilder` no longer stores caller-authored capability strings and no longer exposes
`with_capability()`, `with_audio()`, `with_embedding()`, or `with_image_generation()`. The source
guard `builder_does_not_expose_noop_capability_flags` prevents reintroducing the removed no-op
builder control plane while leaving `ProviderCapabilities` provider metadata untouched.

Migration guidance now points callers to provider builder helpers, registry family handles, and
explicit extension modules.
