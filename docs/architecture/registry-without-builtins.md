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
- Generic-client construction belongs behind explicit `compat_*_client` methods. The old
  `language_model` trait method remains only as a deprecated source-compatibility wrapper; see
  `docs/workstreams/fearless-architecture-convergence/compatibility-audit.md` for the current
  boundary.
- If you _do_ want built-in providers, enable `siumai-registry` features like `openai` / `ollama`
  (these imply `builtins`).
