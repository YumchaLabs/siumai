# Fearless Registry Facade Construction Boundary - Design

Last updated: 2026-05-14

## Context

`fearless-boundary-hardening` and `fearless-core-provider-alias-extraction` tightened the largest
crate-boundary leaks: provider/protocol logic moved out of `siumai-core`, registry handles became
family-first, and provider-owned model aliases left core utilities.

The next coupling point is provider construction. Public facade tests and some compatibility
builders still reach directly for concrete built-in factory types such as
`registry::factories::OpenAIProviderFactory`. That makes the facade aware of internal factory
classes instead of treating built-in provider construction as registry-owned routing.

This workstream narrows that surface. Custom providers still need `ProviderFactory`, but built-in
provider factory selection should be centralized behind registry helpers.

## Decision

Built-in provider factory construction belongs to `siumai-registry`.

The facade may expose stable registry handles, registry options, provider build overrides, and the
`ProviderFactory` trait for custom provider integration. It should not require public consumers or
facade tests to instantiate concrete built-in factory structs when the intent is "use the built-in
OpenAI/Gemini/Vertex/etc. provider".

During the fearless refactor phase, duplicated construction tables should be collapsed into a
single registry-owned helper instead of preserving multiple compatibility paths.

## Goals

- Introduce a registry helper that returns a built-in provider factory by provider id.
- Reuse that helper from default registry creation and compatibility `SiumaiBuilder` construction.
- Move public facade tests away from `siumai::registry::factories::*` concrete factory imports when
  they only need a built-in provider.
- Keep custom provider registration and the `ProviderFactory` trait available.
- Add source guards that keep public facade tests from reintroducing concrete built-in factory
  dependencies.
- Update the docs index so this workstream is visible.

## Non-goals

- Do not delete provider factory structs in this workstream; provider contract tests still need
  concrete factories.
- Do not remove `ProviderFactory` from the public registry surface; custom registries depend on it.
- Do not rewrite provider white-box fixture tests that intentionally exercise provider specs or
  transformers.
- Do not replace all registry override call sites across the repository in one large churny patch.

## Target Boundary

### Stable facade / user-facing tests

Allowed:

- `registry::global()`
- `registry::create_registry_with_defaults()`
- `registry::create_provider_registry(...)` for custom providers
- `registry::builtin_provider_factory("provider-id")` for focused tests and advanced integrations
- `registry::openai_compatible_provider_factory("provider-id")` for OpenAI-compatible vendors
  and dynamic compatible provider ids
- `registry::azure_provider_factory_with_options(...)` for Azure URL-mode and metadata-key
  registry setups that cannot be represented by generic built-in factory selection
- `ProviderBuildOverrides`, `RegistryOptions`, and `ProviderFactory`

Discouraged:

- `siumai::registry::factories::<ConcreteBuiltInFactory>` for normal built-in provider use

### `siumai-registry`

Expected:

- Own built-in provider-id-to-factory routing.
- Keep provider-specific factory selection in one helper.
- Keep default registry creation and compatibility builder construction on the same routing helper.

### Provider factory modules

Expected:

- Remain concrete implementation modules.
- Stay directly visible to registry contract tests and advanced low-level integrations until a
  later removal decision explicitly narrows them.

## Migration Policy

No public compatibility shim is added for direct concrete built-in factory construction. The
recommended replacement is:

```rust
let factory = siumai::registry::builtin_provider_factory("openai")?;
```

OpenAI-compatible dynamic provider ids should use:

```rust
let factory = siumai::registry::openai_compatible_provider_factory("openrouter")?;
```

Azure's deployment-based URL mode is the one current built-in exception that needs a
provider-specific registry helper:

```rust
let factory = siumai::registry::azure_provider_factory_with_options(
    "azure",
    AzureUrlConfig::default(),
    "azure",
)?;
```

Custom providers should continue to implement and register `ProviderFactory` directly.

## Guardrails

- Do not move provider-specific construction into `siumai` facade modules.
- Do not add provider crate dependencies to `siumai-core`.
- Keep `siumai-registry` usable without built-ins when the `builtins` feature is disabled.
- Keep factory contract tests free to use concrete factories.
- Avoid broad formatting churn and unrelated test rewrites.

## Suggested Validation

```text
cargo fmt -p siumai-registry -p siumai --check
cargo check -p siumai-registry --tests --features openai,google-vertex,deepseek,deepinfra,togetherai --no-default-features
cargo check -p siumai --tests --features openai,google-vertex,deepseek,deepinfra,togetherai --no-default-features
cargo nextest run -p siumai-registry factory_architecture_boundary_test --features openai,google-vertex,deepseek,deepinfra,togetherai --no-default-features --no-fail-fast
cargo nextest run -p siumai --test public_surface_imports_test --features openai,google-vertex,deepseek,deepinfra,togetherai --no-default-features --no-fail-fast
cargo nextest run -p siumai --test openai_embedding_public_helper_request_parity_test --features openai --no-default-features --no-fail-fast
cargo nextest run -p siumai --test google_vertex_typed_metadata_boundary_test --features google-vertex --no-default-features --no-fail-fast
git diff --check
```

## Closeout State

Closed on 2026-05-16. Built-in factory selection, default-model policy, registry option defaults,
and common provider build override composition are now registry-owned. Migrated public facade
parity modules use `RegistryBuilder` provider-level shortcuts instead of raw `RegistryOptions`
maps or generic provider build override wrappers. Concrete built-in factory exports remain public
for provider contract tests and advanced low-level integrations; narrowing that public surface is a
separate deferred breaking-change decision.
