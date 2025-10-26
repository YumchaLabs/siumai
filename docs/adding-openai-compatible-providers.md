# Adding an OpenAI‑Compatible Provider

This guide shows how to add a provider whose API is compatible with the OpenAI REST surface (chat/completions/responses/embeddings) while keeping Siumai’s unified interface intact.

## Overview

OpenAI‑compatible providers are integrated via the `providers/openai_compatible` layer. You supply a provider id, base URL, and any header/field name differences; Siumai handles parameter mapping, request building, streaming, and error handling.

## Steps

1) Pick a provider id and base URL
- Example: `deepseek`, base URL: `https://api.deepseek.com/v1`

2) Register provider in the registry
- File: `src/providers/openai_compatible/registry.rs`
- Add an entry that specifies:
  - Provider name/id
  - Base URL (default)
  - Header overrides (if any)
  - Field mappings (if request/response fields differ)

3) (Optional) Add default models
- File: `src/providers/openai_compatible/default_models.rs`
- Add frequently used model ids for better DX.

4) Validate request mapping
- Ensure common params (model/temperature/max_tokens/top_p/stop/seeds) map correctly.
- Verify any provider‑specific headers (auth, beta flags) are set via the adapter.

5) Test
- Unit tests: `tests/providers/*` and `tests/parameters/*` for mapping and header correctness.
- Streaming tests: `tests/streaming/*` to verify token/content/tool deltas.
- Integration: add a focused test using `--ignored` with real keys if feasible.

## Example: Adding "deepseek"

- Register in `openai_compatible::registry` with:
  - id: `"deepseek"`
  - base_url: `https://api.deepseek.com/v1`
  - auth header: `Authorization: Bearer <key>` (default)
  - any custom headers required by the provider

Then you can use it through the unified interface:

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .provider_id("deepseek")
        .api_key(std::env::var("DEEPSEEK_API_KEY")?)
        .model("deepseek-chat")
        .build()
        .await?;

    let rsp = client.chat(vec![user!("Hello! Who are you?")]).await?;
    println!("{}", rsp.content_text().unwrap_or(""));
    Ok(())
}
```

Or via the OpenAI‑compatible path using OpenAI config:

```rust
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Treat as an OpenAI‑compatible provider
    let client = Provider::openai()
        .api_key(std::env::var("DEEPSEEK_API_KEY")?)
        .base_url("https://api.deepseek.com/v1")
        .model("deepseek-chat")
        .build()
        .await?;
    let rsp = client.chat(vec![user!("Ping?")]).await?;
    println!("{}", rsp.content_text().unwrap_or(""));
    Ok(())
}
```

## Tips
- Keep mapping minimal: only override what differs from OpenAI.
- Prefer unified interface (`Siumai::builder()`) for portability; fall back to provider‑specific clients when you need advanced features.
- Ensure streaming tool‑call deltas behave correctly by running the streaming tests.
