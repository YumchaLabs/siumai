# Using `siumai-registry` without built-in providers

See also: `docs/architecture/architecture-refactor-plan.md`.

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
siumai-registry = { version = "0.11.0-beta.5", default-features = false }
```

## Minimal example (custom `ProviderFactory`)

At a high level:

1. Implement `ProviderFactory` (create `LlmClient` instances).
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

- `ProviderFactory::language_model_with_ctx` receives a `BuildContext` that can carry cross-cutting
  settings (HTTP config, retry, interceptors, auth). Your factory may ignore it or use it to
  build consistent clients.
- If you *do* want built-in providers, enable `siumai-registry` features like `openai` / `ollama`
  (these imply `builtins`).

