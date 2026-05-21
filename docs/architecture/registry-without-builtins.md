# Using `siumai-registry` without built-in providers

See also: `docs/architecture/module-split-design.md`.

`siumai-registry` is designed to be an abstraction-first integration point:
you can use it to resolve `"provider:model"` identifiers and apply caching/middleware,
without pulling in any built-in provider implementations.

This is useful when you:

- ship a proprietary provider crate,
- want to wrap an internal gateway service,
- or want to keep your dependency graph small (no built-in HTTP provider code).

## Cargo features

By default, `siumai-registry` has **no** built-in providers enabled (`default = []`).
Avoid enabling features like `openai`, `anthropic`, etc. if you want a pure abstraction build.

```toml
[dependencies]
siumai-registry = { version = "0.11.0-beta.6", default-features = false }
```

## Minimal example (custom `ProviderFactory`)

At a high level:

1. Implement `ProviderFactory` with native family-model methods such as
   `language_model_text_with_ctx`.
2. Register factories into `HashMap<provider_id, Arc<dyn ProviderFactory>>`.
3. Build a registry handle via `create_provider_registry`.
4. Resolve and use models like `"my_provider:my_model"`.

Do not build new custom providers around `registry::factory::build_*_client(...)`; those broad
generic-client helpers are compatibility-only shims for older built-in construction paths. Prefer
provider-owned config objects inside the `*_family_with_ctx(...)` methods.

See the runnable example:

- `siumai-registry/examples/no_builtins_custom_factory.rs`

Run it:

```bash
cargo run -p siumai-registry --example no_builtins_custom_factory
```

## Notes

- `ProviderFactory::*_family_with_ctx` methods receive a `BuildContext` that can carry
  cross-cutting settings (HTTP config, retry, interceptors, auth). Your factory may ignore it or use
  it to build consistent clients.
- Generic-client construction belongs behind explicit `compat_*_client` /
  `compat_*_client_with_ctx` methods. Use those only for migration code or extension-only surfaces
  that do not yet have a first-class family model. New registry execution should resolve family
  handles through methods such as `language_model_text_with_ctx` instead of constructing
  `Arc<dyn LlmClient>`.
- Built-in factories should centralize typed-client `Arc` projection in a small helper such as
  `build_*_arc(...)`. Family and compatibility methods may both reuse that helper, but they should
  not repeat `build_*_with_ctx(...)` + `Arc::new(...)` glue or route same-family construction
  through `compat_language_client_with_ctx(...)`.
- Registry execution is split into narrower facets:
  - `ProviderFamilyFactory` for stable family-model construction;
  - `ProviderCompatibilityFactory` for legacy generic-client construction;
  - `ProviderExtensionFactory` for non-family extension capabilities.
  Custom providers still implement `ProviderFactory`; these facets are blanket-implemented from
  that source-compatible trait so registry handles can depend on the narrower boundary.
- If you _do_ want built-in providers, enable `siumai-registry` features like `openai` / `ollama`
  (these imply `builtins`).
