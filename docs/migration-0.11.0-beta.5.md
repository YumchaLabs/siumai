# Migration Guide: `<=0.11.0-beta.4` → `0.11.0-beta.5` (Alpha.5 Split-Crate Refactor)

This guide focuses on **breaking changes** introduced in `0.11.0-beta.5` and how to migrate
downstream code quickly.

If you are new to the refactor direction, read `docs/next-steps.md` and `docs/module-split-design.md` first.

## Version scope

- This guide targets upgrading from `0.11.0-beta.4` (and earlier alpha.5 snapshots) to `0.11.0-beta.5`.
- If you upgraded across multiple beta versions, apply the steps in this guide first to restore builds,
  then address any remaining, smaller API adjustments.

## Summary of breaking changes

### 1) Default features: `openai` only

The `siumai` facade crate now defaults to `openai` only.

- If you relied on “everything enabled by default”, you must enable provider features explicitly.

```toml
[dependencies]
siumai = { version = "0.11.0-beta.5", features = ["openai", "anthropic", "google", "ollama", "groq", "xai", "minimaxi"] }
```

To build a minimal binary without OpenAI by default:

```toml
[dependencies]
siumai = { version = "0.11.0-beta.5", default-features = false, features = ["groq"] }
```

### 2) Provider options transport: `ProviderOptions` removed

The legacy closed `ProviderOptions` enum transport is removed.

- Requests no longer have a `provider_options` field.
- All provider-specific parameters must go through the open JSON map:
  - `provider_options_map: ProviderOptionsMap`
  - `request.with_provider_option("<provider_id>", <json>)`

### 3) Typed provider options are provider-owned

Typed option structs live in provider crates and are exposed via stable facade paths:

- `siumai::provider_ext::openai::*`
- `siumai::provider_ext::anthropic::*`
- `siumai::provider_ext::gemini::*`
- `siumai::provider_ext::groq::*`
- `siumai::provider_ext::xai::*`
- `siumai::provider_ext::ollama::*`
- `siumai::provider_ext::minimaxi::*`

If you imported typed options from `siumai::types::*` before, update imports accordingly (the top-level
`siumai::types` module is no longer part of the stable facade surface; see breaking change #8).

### 4) `siumai::providers::*` removed

The historical `siumai::providers::*` module path is removed to prevent cross-layer coupling.

- Unified surface: `use siumai::prelude::unified::*;`
- Provider-specific surface: `use siumai::provider_ext::<provider>::*;`
- Protocol-layer/helpers: `use siumai::experimental::*;` (advanced) or depend on the relevant provider crate directly (e.g. `siumai-provider-openai`).

### 5) Groq/xAI internal chat capability modules removed

Groq/xAI chat code paths were unified to always go through `ProviderSpec + standards::openai`
(OpenAI-like family). If you relied on internal symbols like `GroqChatCapability`/`XaiChatCapability`,
switch to `GroqClient`/`XaiClient` (or `ChatCapability`).

### 6) OpenAI web search options are deprecated (use hosted tools + Responses API)

OpenAI web search is now modeled as a **provider-defined tool** (Vercel-aligned).

- `OpenAiOptions::with_web_search_options(...)` is deprecated.
- Prefer: `ChatRequest::with_tools(...)` + `siumai::hosted_tools::openai::web_search()` + enabling the Responses API.

### 7) Low-level internal modules moved under `siumai::experimental::*`

The facade crate no longer re-exports low-level internals at the top-level:

- Removed: `siumai::{execution,auth,utils,params,retry,defaults,client,observability}::*`
- Use instead: `siumai::experimental::{execution,auth,utils,params,retry,defaults,client,observability}::*`

This keeps the stable surface small and prevents accidental cross-layer imports.

### 8) Facade surface tightened (top-level `types/traits/error/streaming` removed)

The facade now intentionally keeps the stable surface small.

- Removed stable entry points:
  - `siumai::types::*`
  - `siumai::traits::*`
  - `siumai::error::*`
  - `siumai::streaming::*`
- `LlmBuilder` is no longer re-exported from the unified prelude.
  - Prefer `Siumai::builder()` for unified construction.
  - For provider-specific construction, prefer `Provider::<provider>()` / `siumai::provider_ext::<provider>::*`.
- Use instead:
  - Unified surface (recommended): `use siumai::prelude::unified::*;`
  - Non-unified extension capabilities: `use siumai::extensions::*;` + `use siumai::extensions::types::*;`
  - Advanced / low-level building blocks: `use siumai::experimental::*;`

### 9) `siumai::core` is no longer a stable entry point

The facade no longer exposes `siumai::core::*` as a convenience path.

- If you need low-level core building blocks, use `siumai::experimental::core::*`.
- If you need provider internals, depend on the relevant provider crate directly (recommended) or use
  `siumai::experimental::providers::*` (advanced, unstable surface).

## Migration cookbook

### A) Attach provider options (open map)

Use `with_provider_option` directly:

```rust
use siumai::prelude::unified::*;

let req = ChatRequest::new(vec![user!("hi").build()])
    .with_provider_option("openai", serde_json::json!({
        "reasoning_effort": "high"
    }));
```

Or use the per-provider convenience setters (feature-gated):

```rust
use siumai::prelude::unified::*;
use siumai::provider_ext::openai::{OpenAiChatRequestExt, OpenAiOptions, ReasoningEffort};

let req = ChatRequest::new(vec![user!("hi").build()])
    .with_openai_options(
        OpenAiOptions::new().with_reasoning_effort(ReasoningEffort::High),
    );
```

### B) Custom provider options (your own provider id)

Implement `CustomProviderOptions`, convert it to a `(provider_id, json)` pair, then insert it into the map:

```rust
use siumai::prelude::unified::*;
use siumai::experimental::custom_provider::CustomProviderOptions;

#[derive(Debug, Clone)]
struct MyFeature {
    mode: String,
}

impl CustomProviderOptions for MyFeature {
    fn provider_id(&self) -> &str {
        "my-provider"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        Ok(serde_json::json!({ "mode": self.mode }))
    }
}

let feature = MyFeature { mode: "fast".to_string() };
let (provider_id, value) = feature.to_provider_options_map_entry()?;
let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
    .with_provider_option(provider_id, value);
# Ok::<(), LlmError>(())
```

### C) OpenAI-compatible vendors: provider id keyed options merge

For OpenAI-compatible vendors (e.g. `"deepseek"`, `"openrouter"`), provider options are keyed by the
**runtime provider id** and merged into the OpenAI-compatible request body.

```rust
use siumai::prelude::unified::*;

let client = Siumai::builder()
    .deepseek()
    .api_key("...")
    .model("deepseek-chat")
    .build()
    .await?;

let req = ChatRequest::new(vec![user!("hi").build()])
    .with_provider_option("deepseek", serde_json::json!({
        "stream_options": { "include_usage": true }
    }));

let _ = client.chat_request(req).await?;
# Ok::<(), LlmError>(())
```

### D) OpenAI built-in tools (Responses API): web search

Use provider-defined tools (stable, Vercel-aligned) and explicitly enable the Responses API:

```rust
use siumai::prelude::unified::*;
use siumai::hosted_tools::openai as openai_tools;
use siumai::provider_ext::openai::{OpenAiChatRequestExt, OpenAiOptions, ResponsesApiConfig};

let req = ChatRequest::new(vec![user!("What's new in Rust async runtimes?")])
    .with_tools(vec![
        openai_tools::web_search()
            .with_search_context_size("high")
            .build(),
    ])
    .with_openai_options(OpenAiOptions::new().with_responses_api(ResponsesApiConfig::new()));
```

## Quick checklist

- Update `Cargo.toml` features (default is `openai` only).
- Replace legacy `ProviderOptions` usage with `provider_options_map` helpers:
  - `with_provider_option(...)`
  - `with_<provider>_options(...)` where available
- Update imports for typed provider options to `siumai::provider_ext::<provider>::*`.
- If you referenced internal provider modules/capabilities, move to `*Client` + unified traits.
- If you used OpenAI built-in tool options:
  - migrate to `ChatRequest::with_tools(...)` + `siumai::hosted_tools::openai::*`

## Common compiler errors (quick fixes)

### `error[E0609]: no field 'provider_options' on type ...Request`

You were setting the removed legacy field.

- Remove `provider_options: ...` from request construction.
- Use `provider_options_map` helpers instead:
  - `request.with_provider_option("<provider_id>", serde_json::json!({...}))`
  - or `with_<provider>_options(...)` convenience methods when available.

### `error[E0432]: unresolved import ... ProviderOptions`

The closed `ProviderOptions` enum transport is removed.

- Replace any `ProviderOptions::...` usage with `provider_options_map`.
- If you had typed options, import them from `siumai::provider_ext::<provider>::*`.

### `error[E0432]: unresolved import ... GroqChatCapability` / `XaiChatCapability`

Those internal capability structs were removed as part of the OpenAI-like family convergence.

- Prefer the client types: `GroqClient`, `XaiClient`.
- Prefer the unified traits: `ChatCapability` / `LlmClient`.

### `error[E0432]: unresolved import siumai::types::...Options`

Typed provider options are provider-owned now.

Examples:

- `siumai::types::OpenAiOptions` → `siumai::provider_ext::openai::OpenAiOptions`
- `siumai::types::AnthropicOptions` → `siumai::provider_ext::anthropic::AnthropicOptions`
- `siumai::types::GeminiOptions` → `siumai::provider_ext::gemini::GeminiOptions`

### `error[E0432]: unresolved import siumai::types` (or `siumai::traits`, `siumai::error`, `siumai::streaming`)

Those top-level module paths are no longer part of the stable facade surface.

- Prefer: `use siumai::prelude::unified::*;`
- For non-unified extension capabilities:
  - traits: `use siumai::extensions::*;`
  - types: `use siumai::extensions::types::*;`
- For low-level internals: `use siumai::experimental::*;`

### `error: cannot find module provider_ext::<provider>`

You are missing the provider feature at compile time.

Update your `Cargo.toml`:

```toml
[dependencies]
siumai = { version = "0.11.0-beta.5", features = ["<provider>"] }
```

Or if you disabled defaults:

```toml
[dependencies]
siumai = { version = "0.11.0-beta.5", default-features = false, features = ["<provider>"] }
```

### `error[E0432]: unresolved import siumai::execution` (or `siumai::auth`, `siumai::utils`, ...)

Those low-level modules are no longer re-exported at the top-level.

- Replace: `siumai::execution::...` → `siumai::experimental::execution::...`
- Replace: `siumai::auth::...` → `siumai::experimental::auth::...`
- Replace: `siumai::utils::...` → `siumai::experimental::utils::...`

### `error[E0432]: unresolved import siumai::core::...`

The facade no longer exposes `siumai::core::*` as a convenience path.

- If you need low-level core building blocks, use `siumai::experimental::core::*`.
- If you need provider internals, depend on the relevant provider crate directly (recommended) or use
  `siumai::experimental::providers::*` (advanced, unstable surface).
