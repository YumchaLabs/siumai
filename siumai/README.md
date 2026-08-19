# `siumai` facade

`siumai` is the curated application facade for the Siumai workspace. It combines typed provider
construction, six provider-neutral model families, optional Registry/runtime integrations, and
provider-owned native APIs without rebuilding a universal client or capability matrix.

This README and the crate rustdoc document the unreleased facade on the repository's `main` branch.
The published `0.11.0-beta.10` package does not contain this typed `Siumai` hub. Until a later beta
is published, use the Git source below and pin a revision for reproducible application builds:

```toml
[dependencies]
siumai = { git = "https://github.com/YumchaLabs/siumai.git", default-features = false, features = ["openai"] }
```

No provider feature is enabled by default. The facade's default features are `registry` and
`runtime`; keep `default-features = false` when an application does not use those integrations.

## 1. Typed direct calls

Start ordinary application code with `Siumai::builder()`, select one provider, build a reusable
typed hub, and bind the model family that the call needs:

```rust,no_run
# #[cfg(feature = "openai")]
use siumai::providers::openai::models::GPT_5_6;
# #[cfg(feature = "openai")]
use siumai::Siumai;

# #[cfg(feature = "openai")]
async fn quickstart() -> Result<(), Box<dyn std::error::Error>> {
    let ai = Siumai::builder()
        .openai()
        .api_key("example-openai-key")
        .build()?;
    let client = ai.language(GPT_5_6)?;
    let response = client.generate("Explain typed provider hubs in one sentence.").await?;

    if let Some(text) = response.output_text() {
        println!("{text}");
    }
    Ok(())
}

# #[cfg(feature = "openai")]
fn main() {
    // `quickstart` is intentionally not polled: the credential is synthetic and
    // this documentation remains a compile-only, network-free contract.
    drop(quickstart());
}
# #[cfg(not(feature = "openai"))]
# fn main() {}
```

`generate` returns the complete `LanguageResponse`. `output_text()` is only a display projection;
content parts, termination, usage, warnings, metadata, reasoning, tools, citations, and media remain
available on the response. Use `project_assistant_history()` rather than text projection when
constructing a continuation.

Changing OpenAI to Anthropic or Gemini changes provider construction and the model identifier, not
the application call:

```text
let client = ai.language(model)?;
let response = client.generate("Hello").await?;
```

The compile-checked `provider_switching` example shows all three typed construction chains with the
same `client.generate(...)` line.

One hub can bind multiple models and families without storing provider-wide default model slots:

```rust,no_run
# #[cfg(feature = "openai")]
# fn multi_family() -> Result<(), Box<dyn std::error::Error>> {
use siumai::providers::openai::embeddings::TEXT_EMBEDDING_3_SMALL;
use siumai::providers::openai::images::GPT_IMAGE_2;
use siumai::providers::openai::models::GPT_5_6;
use siumai::Siumai;

let ai = Siumai::builder()
    .openai()
    .api_key("example-openai-key")
    .build()?;
let language = ai.language(GPT_5_6)?;
let embedding = ai.embedding(TEXT_EMBEDDING_3_SMALL)?;
let image = ai.image(GPT_IMAGE_2)?;

assert!(std::ptr::eq(language.provider(), embedding.provider()));
assert!(std::ptr::eq(language.provider(), image.provider()));
# Ok(())
# }
# fn main() {
#     #[cfg(feature = "openai")]
#     multi_family().unwrap();
# }
```

Family methods use one public vocabulary: language `generate`, `stream`, and `call`; embedding
`embed` and `call`; rerank `rerank` and `call`; image `generate` and `call`; speech `synthesize` and
`call`; transcription `transcribe` and `call`.

## 2. Root family modules for generic and Registry code

The root `language`, `embedding`, `rerank`, `image`, `speech`, and `transcription` modules remain the
canonical execution seam for dependency injection, concrete generic models, family trait objects,
and Registry-resolved models:

```rust,no_run
use siumai::{language, LanguageCallError, LanguageModel, LanguageResponse};

async fn answer<M>(model: &M) -> Result<LanguageResponse, LanguageCallError>
where
    M: LanguageModel + ?Sized,
{
    language::generate(model, "Explain explicit type erasure.").await
}
```

A typed facade client implements the same family trait, so generic code accepts it without a
facade-specific abstraction. Registry remains an explicit, caller-configured, one-way erasure
boundary: resolve a family model, pass it to the same root function, and accept that concrete native
APIs are no longer recoverable after erasure. `Siumai::builder()` does not register providers,
resolve routes, inspect global state, or choose fallback policy.

Use `Siumai::from_provider(provider)` when a provider is already configured or supplied by a
third-party crate. The provider's existing family-provider traits determine which family selectors
are available at compile time.

## 3. Typed provider and model access

Portable calls and provider-specific intent coexist on the same exact target. Advanced calls reuse
the family call builder, while `provider()` exposes provider-wide resources and `model()` exposes
mode- or model-specific operations:

```rust,no_run
# #[cfg(feature = "openai")]
use siumai::providers::openai::models::GPT_5_6;
# #[cfg(feature = "openai")]
use siumai::providers::openai::responses::OpenAiResponsesOptions;
# #[cfg(feature = "openai")]
use siumai::Siumai;

# #[cfg(feature = "openai")]
async fn advanced() -> Result<(), Box<dyn std::error::Error>> {
    let ai = Siumai::builder()
        .openai()
        .api_key("example-openai-key")
        .build()?;
    let client = ai.language(GPT_5_6)?;
    let options = OpenAiResponsesOptions {
        instructions: Some("Keep this provider intent exact.".to_string()),
        ..OpenAiResponsesOptions::default()
    };
    let response = client
        .call("Summarize the facade boundary.")
        .with_provider_options(&options)?
        .generate()
        .await?;

    let _complete_content = response.content();
    let _files = client.provider().files();
    let _responses_model = client.model();
    Ok(())
}

# #[cfg(feature = "openai")]
fn main() {
    // Compile the complete path without making a provider request.
    drop(advanced());
}
# #[cfg(not(feature = "openai"))]
# fn main() {}
```

Native files, batches, catalogs, sessions, hosted tools, media jobs, Realtime, WebSocket sessions,
and native response methods stay on their provider-owned types. The facade intentionally adds no
native resource enum, forwarding layer, `Any`, or downcast escape hatch.

## Explicit language API modes

Canonical language selectors have fixed provider meanings and never inspect model names:

- OpenAI `.language(model)` uses Responses; `.chat_completions(model)` selects Chat Completions.
- Gemini `.language(model)` uses Interactions; `.generate_content(model)` selects Generate Content.
- Other providers document their canonical mode and expose alternatives only through their one
  explicit facade method or provider-owned model constructor.

```rust,no_run
# #[cfg(all(feature = "openai", feature = "google"))]
# fn api_modes() -> Result<(), Box<dyn std::error::Error>> {
use siumai::providers::google::models::GEMINI_3_5_FLASH;
use siumai::providers::openai::models::GPT_5_6;
use siumai::Siumai;

let openai = Siumai::builder()
    .openai()
    .api_key("example-openai-key")
    .build()?;
let _responses = openai.language(GPT_5_6)?;
let _chat = openai.chat_completions(GPT_5_6)?;

let gemini = Siumai::builder()
    .gemini()
    .api_key("example-gemini-key")
    .build()?;
let _interactions = gemini.language(GEMINI_3_5_FLASH)?;
let _generate_content = gemini.generate_content(GEMINI_3_5_FLASH)?;
# Ok(())
# }
# fn main() {
#     #[cfg(all(feature = "openai", feature = "google"))]
#     api_modes().unwrap();
# }
```

Unknown future model identifiers use the provider's protocol baseline. Model names do not enable
reasoning, caching, media, tools, or a different API mode heuristically.

## Provider features and construction

Provider features are additive and activate only their owning provider plus the narrow transport
configuration surface:

| Cargo feature | Typed builder selector |
|---|---|
| `openai` | `.openai()` |
| `anthropic` | `.anthropic()` |
| `google` | `.gemini()` |
| `openai-compatible` | `.openai_compatible()` |
| `alibaba` | `.alibaba()` |
| `moonshotai` | `.moonshot()` |
| `volcengine` | `.volcengine()` |
| `google-vertex-anthropic` | `.vertex_anthropic()` |
| `groq` | `.groq()` |
| `xai` | `.xai()` |
| `minimax` | `.minimax()` |
| `deepseek` | `.deepseek()` |
| `cohere` | `.cohere()` |
| `deepgram` | `.deepgram()` |
| `elevenlabs` | `.elevenlabs()` |

`all-providers` activates the retained branded provider features but not the generic
`openai-compatible` escape hatch or experimental OpenAI Realtime/WebSocket features.

Typed construction stages expose only required inputs plus one optional `configure_provider`
transition over the real provider builder. `.build()` is synchronous and network-free and returns
the provider-owned configuration error. The facade does not discover environment credentials or
retain a second plaintext credential copy. Alibaba additionally requires
`configure_provider(...)` to select a real endpoint before `.build()`; Vertex Anthropic requires
project, location, and a Google credential; OpenAI-compatible and ElevenLabs construction starts
with an explicit provider-owned profile.

See the repository provider support policy for the exact implemented family/API-mode/native slice.
A facade feature is not a claim of complete parity with every product a vendor offers.

## Registry, runtime, and transport

Enable `registry` for immutable local lookup over providers the caller has already configured.
Enable `runtime` for multi-step tool loops, structured output, approvals, budgets, and durable run
behavior. Use `Runtime::{generate, stream}` when runtime defaults or step orchestration are required;
ordinary one-call applications use a family client or root family function.

Every provider feature also enables the curated `siumai::transport` configuration namespace.
`ProviderHttpTransportSettings` owns retry caps, deadlines, bounds, attempt observation, and
explicit trusted CONNECT routing. It does not expose authenticated execution, raw clients, request
interceptors, or environment proxy discovery.

## Compile-checked examples

- `examples/provider_switching.rs` — OpenAI, Anthropic, and Gemini typed construction followed by
  identical portable calls.
- `examples/registry_switching.rs` — the separate generic/Registry path and deliberate one-way type
  erasure.
- `examples/openai_flagship.rs` — typed Responses options and prompt-cache intent, complete portable
  responses, native resources, and model-native Responses access.
- `examples/anthropic_flagship.rs` — typed Messages/cache/file intent, complete portable responses,
  and provider-native Files, Message Batches, and model cache prewarming.
- `examples/trusted_connect_route.rs` — explicit transport-route configuration without credentials
  or network I/O.

All flagship and switching examples use synthetic credentials and remain offline: they construct
or compile futures without polling provider requests.

See the [repository README](https://github.com/YumchaLabs/siumai#readme),
[migration guide](https://github.com/YumchaLabs/siumai/blob/main/docs/migration/siumai-next.md),
[typed facade ADR](https://github.com/YumchaLabs/siumai/blob/main/docs/adr/0020-typed-siumai-provider-hub.md),
and [provider support policy](https://github.com/YumchaLabs/siumai/blob/main/docs/providers/support-policy.md)
for the release boundary and dated support evidence.
