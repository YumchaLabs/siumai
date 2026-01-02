# Provider Extensions: Tools, Options, and `provider_ext`

This document explains how provider-specific features are represented in Siumai without polluting
the unified capability surface.

## Three extension mechanisms

### A) Provider-hosted tools (`hosted_tools::*`)

These are tools that execute on the provider side (not in your application).
They are represented as `Tool::ProviderDefined`.

Typical use cases:

- Web search
- File search / file stores
- Code execution
- URL context / grounding

### B) Provider options (pass-through `providerOptions`)

Provider options are extra provider-specific request inputs.

Guideline:

- If a feature can be fully encapsulated within the provider implementation, expose it as provider options.
- Do **not** add new fields to unified request types for provider-only features.

Example (OpenAI Responses API toggles via `providerOptions`):

```rust,ignore
use siumai::prelude::unified::*;

let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
    .with_provider_option("openai", serde_json::json!({
        "responsesApi": {
            "enabled": true,
            "previousResponseId": "resp_123"
        }
    }));
```

Client-level defaults (builder) are also supported, and requests override defaults:

In the unified surface, prefer request-level `providerOptions` (`ChatRequest::with_provider_option(...)`)
or typed request extensions (see below). Provider-specific builder defaults are intentionally kept out
of the unified facade.

Notes:

- Provider ids are case-insensitive (normalized to lowercase).
- For OpenAI, both camelCase (Vercel-style) and snake_case keys are accepted.

#### Typed provider options (preferred ergonomics)

When a provider offers typed options, they live in the provider crate and are re-exported via
`siumai::provider_ext::<provider>::*`.

OpenAI example:

```rust,no_run
use siumai::prelude::unified::*;
use siumai::provider_ext::openai::{OpenAiChatRequestExt, OpenAiOptions, ResponsesApiConfig};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let client = Siumai::builder()
	    .openai()
    .api_key("sk-...")
    .model("gpt-4.1-mini")
    .build()
    .await?;

	let req = ChatRequest::new(vec![user!("hi")])
	    .with_openai_options(OpenAiOptions::new().with_responses_api(ResponsesApiConfig::new()));

let _ = client.chat_request(req).await?;
# Ok(()) }
```

#### Typed provider metadata (response-side)

Providers may also expose typed metadata helpers as extension traits:

```rust,ignore
use siumai::provider_ext::openai::{OpenAiChatResponseExt, OpenAiMetadata};

fn extract(resp: &siumai::prelude::unified::ChatResponse) -> Option<OpenAiMetadata> {
    resp.openai_metadata()
}
```

Gemini example:

```rust,ignore
use siumai::provider_ext::gemini::{GeminiChatResponseExt, GeminiMetadata};

fn extract(resp: &siumai::prelude::unified::ChatResponse) -> Option<GeminiMetadata> {
    resp.gemini_metadata()
}
```

Or use the provider-agnostic helper (works for any `Deserialize` metadata type):

```rust,ignore
use siumai::provider_ext::openai::OpenAiMetadata;

fn extract(resp: &siumai::prelude::unified::ChatResponse) -> Option<OpenAiMetadata> {
    resp.provider_metadata_as::<OpenAiMetadata>("openai")
}
```

OpenAI-compatible vendors (SiliconFlow/DeepSeek/OpenRouter/...) are treated as configuration presets.
You can use a vendor preset directly (recommended), or use the explicit `openai().compatible("<vendor>")` form:

```rust,no_run
use siumai::prelude::unified::*;

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = Siumai::builder()
    .siliconflow()
    .api_key("your-api-key")
    .model("deepseek-ai/DeepSeek-V3.1")
    .build()
    .await?;
# Ok(()) }
```

```rust,no_run
use siumai::prelude::unified::*;

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = Siumai::builder()
    .provider_id("siliconflow")
    .api_key("your-api-key")
    .model("deepseek-ai/DeepSeek-V3.1")
    .build()
    .await?;
# Ok(()) }
```

Note: vendor presets like `Siumai::builder().siliconflow()` are OpenAI-compatible configuration presets.

Base URL override semantics:

- `base_url(...)` is treated as the **full API prefix** (Vercel AI SDK style). The library does not infer or append
  path segments like `/v1` or `/openai/v1`.
- For vendors whose default base URL includes a path suffix (e.g. Groq uses `/openai/v1`), include that suffix in your
  custom base URL as well (e.g. `format!("{}/openai/v1", server.uri())` for tests/mocks).

API key env fallback:

- By default we fall back to `{PROVIDER_ID}_API_KEY` when `.api_key(...)` is not provided.
- Some vendor presets can override the env var name when the provider id is not a valid env var identifier
  (e.g. `302ai` uses `AI302_API_KEY`).

### C) Provider extension APIs (`provider_ext::<provider>::*`)

Provider extension modules are for provider-specific endpoints/resources that do not map to the
stable model families (or require raw protocol types).

Examples:

- OpenAI Responses API streaming utilities
- Anthropic “thinking replay” helpers
- Provider-specific model management endpoints

## Provider matrix (high level)

This list is intentionally high-level and may evolve quickly.

- OpenAI
  - Hosted tools: web search (Responses API), file search, computer use (if supported by the provider)
  - Extensions: Responses API utilities, non-unified streaming helpers
- Anthropic
  - Hosted tools: web search / web fetch (versioned tool ids)
  - Extensions: thinking replay and related helpers
- Google / Gemini
  - Hosted tools: Google Search / file search / code execution / URL context / enterprise search
  - Extensions: Vertex-specific auth and resource normalization helpers

## API stability rules

- `siumai::prelude::unified::*` is the most stable surface.
- `hosted_tools::*` and `provider_ext::*` are stable module paths, but the feature set can evolve.
- Provider-hosted tool identifiers are provider-owned and may be versioned by the provider.
