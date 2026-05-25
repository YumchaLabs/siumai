# Registry Typed Builder Isolation

Status: Closed
Last updated: 2026-05-25

## Why This Lane Exists

`siumai-registry/src/registry/factory.rs` is documented as a legacy compatibility module for
deprecated `build_*_client(...)` helpers that return generic `LlmClient` objects. That public module
still also owns typed builder helpers used by production provider factories:

- `build_openai_compatible_typed_client(...)`
- `build_gemini_typed_client(...)`
- `build_anthropic_vertex_typed_client(...)`
- `build_google_vertex_typed_client(...)`
- `OpenAiChatApiMode`

That mixes two different interfaces in one module. The compatibility interface is a public migration
surface; the typed builder interface is production wiring for native provider factories.

## Relevant Authority

- `docs/adr/0007-llmclient-demotion-policy.md`
- `docs/adr/0006-family-model-first-trait-policy.md`
- `docs/architecture/public-surface.md`
- `docs/architecture/registry-without-builtins.md`
- `docs/workstreams/fearless-module-deepening/`

## Problem

The current module shape makes `registry::factory` look like the owner of both legacy generic-client
construction and current provider-native typed construction. Source guards already prevent
production provider factories from calling deprecated broad `build_*_client(...)` helpers, but those
same factories still call non-deprecated typed helpers through the legacy module path.

The result is shallow ownership: the legacy public module has to know too much about production
typed construction because the internal helper seam has not been named.

## Target State

- Production provider factories call an internal typed builder module, not `registry::factory`.
- `registry::factory` remains as a compatibility surface for deprecated public generic-client
  helpers.
- Any public typed helper wrappers left in `registry::factory` are compatibility wrappers around
  the internal typed builders, not the implementation owner.
- A source guard rejects production factory calls to `crate::registry::factory::{typed helpers}`.

## In Scope

- A new internal registry module for typed provider-client builders.
- Production factory call-site migration for the typed helpers listed above.
- Source guards and workstream evidence.
- Changelog and architecture notes if public wrapper semantics are clarified.

## Out Of Scope

- Deleting deprecated public `registry::factory::build_*_client(...)` helpers.
- Deleting `ProviderCompatibilityFactory` or `ProviderFactory::compat_*_client*`.
- Rewriting `SiumaiBuilder::build()` to return family-native models.
- Changing provider runtime request/response behavior.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Production factories only need typed helper implementation, not the public legacy module path. | High | `rg` shows production calls target typed helpers through `crate::registry::factory::*`. | Keep compatibility wrappers but migrate only proven internal call sites. |
| Deprecated public `build_*_client(...)` helpers must remain for now. | High | ADR-0007 and compatibility audits keep generic-client compatibility until larger retirement gates are met. | Do not delete them in this lane. |
| Typed helper extraction can be validated by source guards plus `siumai-registry --all-features` compile. | High | The change is module ownership and call path, not runtime serialization. | Add provider-specific focused tests if compile reveals feature-specific drift. |

## Architecture Direction

The registry should expose clear seams:

- `registry::factory` owns legacy public compatibility wrappers.
- `registry::typed_builders` owns internal provider-native typed construction helpers.
- `registry::factories::*` owns provider-specific `ProviderFactory` implementations.

This improves locality: production provider construction can evolve without adding more behavior to
a public compatibility module, while older direct imports keep a stable wrapper path until the
ADR-0007 retirement gates are ready.

## Closeout Condition

This lane can close when:

- production provider factories no longer call typed helpers through `crate::registry::factory`;
- compatibility wrappers, if retained, delegate to the internal typed builder module;
- focused source guards and compile gates pass; and
- evidence is recorded.

Closeout result: complete. The implementation now keeps typed provider-client construction in
internal `registry::typed_builders`, preserves compatibility wrappers in `registry::factory`, and
guards production factories against returning to the legacy public module path.
