# Fearless Refactor V4 - Migration Examples

Last updated: 2026-03-08

## Purpose

This document shows how the V4 architecture should be explained in concrete migration terms.
It is intentionally illustrative and focuses on call-site shape rather than exact implementation details.

## Construction ranking

Use the migration examples with this priority order in mind:

1. registry-first for application code
2. config-first for provider-specific setup
3. builder convenience only when readability or migration ergonomics matter more than architectural explicitness

## Recommended runnable examples

These repository examples now reflect the preferred V4 teaching path for secondary providers:

- `siumai/examples/04-provider-specific/google/logprobs.rs`: config-first Gemini wrapper + typed request options + typed logprobs metadata extraction, backed by no-network `responseLogprobs` / `logprobs` mapping coverage
- `siumai/examples/04-provider-specific/xai/web-search.rs`: config-first xAI wrapper + typed provider options + typed response metadata, backed by no-network typed request-path normalization coverage
- `siumai/examples/04-provider-specific/groq/structured-output.rs`: registry-first Groq model + Stable `response_format` + provider-agnostic JSON extraction, backed by no-network precedence coverage against raw Groq `response_format` provider options
- `siumai/examples/04-provider-specific/groq/logprobs.rs`: config-first Groq wrapper + typed request options + typed logprobs metadata extraction
- `siumai/examples/04-provider-specific/deepseek/reasoning.rs`: config-first DeepSeek wrapper + typed reasoning options + `ChatResponse::reasoning()` extraction
- `siumai/examples/04-provider-specific/ollama/metadata.rs`: registry-first Ollama model + typed timing metadata via `OllamaChatResponseExt`
- `siumai/examples/04-provider-specific/openai-compatible/openrouter-transforms.rs`: built-in generic OpenAI-compatible client + typed `OpenRouterOptions` plus typed `OpenRouterMetadata`, showing compat vendor request/response helpers without falling back to raw `with_provider_option(...)` or `provider_metadata["openrouter"]` traversal
- `siumai/examples/04-provider-specific/openai-compatible/perplexity-search.rs`: built-in generic OpenAI-compatible client + typed `PerplexityOptions` plus typed `PerplexityMetadata`, showing hosted-search request/response escape hatches without promoting them into the Stable family surface

## Example 1 - Builder to registry-first

### Before

```rust
use siumai::prelude::*;

let client = Siumai::builder()
    .openai()
    .api_key("...")
    .model("gpt-4o-mini")
    .build()
    .await?;

let response = client.chat(vec![user!("Hello")]).await?;
```

### After

```rust
use siumai::prelude::unified::*;

let model = registry::global().language_model("openai:gpt-4o-mini")?;

let request = ChatRequest::new(vec![user!("Hello")]);
let response = siumai::text::generate(&model, request, siumai::text::GenerateOptions::default())
    .await?;
```

### Why this is better

- application code depends on a family model, not a provider client
- the model ID is explicit and registry-routable
- provider switching becomes easier

## Example 2 - Builder to config-first provider construction

### Before

```rust
use siumai::prelude::*;

let client = Provider::openai()
    .api_key("...")
    .model("gpt-4o")
    .build()
    .await?;
```

### After

```rust
use siumai::providers::openai::{OpenAiClient, OpenAiConfig};

let config = OpenAiConfig::new("...")
    .with_model("gpt-4o")
    .with_temperature(0.7);

let client = OpenAiClient::from_config(config)?;
```

### Why this is better

- config becomes the canonical construction contract
- builders can remain, but they become thin wrappers over config
- tests and advanced users can construct providers without builder indirection

## Example 3 - Generic client mindset to family-model mindset

### Before

```rust
let client: Arc<dyn LlmClient> = get_client_somehow().await?;

let chat = client
    .as_chat_capability()
    .ok_or_else(|| anyhow!("chat not supported"))?;

let response = chat.chat_request(request).await?;
```

### After

```rust
let model = registry::global().language_model("openai:gpt-4o-mini")?;
let response = siumai::text::generate(&model, request, siumai::text::GenerateOptions::default())
    .await?;
```

### Why this is better

- the call site expresses intent directly
- users do not need to reason about downcasting
- family traits become the execution center

## Example 4 - Streaming stays family-oriented

### Before

```rust
let client = Siumai::builder()
    .openai()
    .api_key("...")
    .model("gpt-4o-mini")
    .build()
    .await?;

let stream = client.chat_stream_request(request).await?;
```

### After

```rust
let model = registry::global().language_model("openai:gpt-4o-mini")?;

let stream = siumai::text::stream(&model, request, siumai::text::StreamOptions::default())
    .await?;
```

### Why this is better

- stream and non-stream are presented consistently within one family surface
- middleware and registry behavior remain hidden behind the model object

## Example 5 - Provider-specific functionality stays in extensions

### Stable path

```rust
let model = registry::global().language_model("openai:gpt-4o")?;
let response = siumai::text::generate(&model, request, Default::default()).await?;
```

### Extension path

```rust
use siumai::providers::openai::resources::OpenAiFiles;
use siumai::providers::openai::metadata::OpenAiChatResponseExt;
```

### Why this is important

- stable APIs remain provider-agnostic
- provider-native growth remains possible without polluting the stable family contracts

## Example 6 - Audio split should become explicit

### Old broad mindset

```rust
let audio = client.as_audio_capability().unwrap();
let tts = audio.text_to_speech(tts_request).await?;
let stt = audio.speech_to_text(stt_request).await?;
```

### New explicit mindset

```rust
let speech_model = registry::global().speech_model("openai:gpt-4o-mini-tts")?;
let tts = siumai::speech::synthesize(&speech_model, tts_request, Default::default()).await?;

let transcription_model = registry::global().transcription_model("openai:whisper-1")?;
let stt = siumai::transcription::transcribe(
    &transcription_model,
    stt_request,
    Default::default(),
)
.await?;
```

## Migration guidance summary

When updating examples, tests, and docs, prefer these transformations:

1. `Siumai::builder()` in application examples -> registry-first
2. provider-specific builders in provider docs/tests -> config-first first, thin builder second
3. `Arc<dyn LlmClient>` call sites -> family model traits
4. broad audio entry points -> explicit speech/transcription families

If a document shows more than one construction style, present them in this order:

1. registry-first
2. config-first
3. builder convenience

## Provider Package Migration Rule

When migrating provider-specific examples, do not assume every provider name needs a fully separate package story.

Use these rules instead:

- native or wrapper providers with real provider-owned semantics should migrate toward `provider_ext::<provider>::*Config` and `*Client::from_config(...)`
- focused providers should migrate only the capabilities they truly own today
- OpenAI-compatible vendor presets should prefer typed vendor views or built-in compat construction before any new top-level package is introduced
- examples for `openrouter` and `perplexity` should stay framed as typed vendor views over the compat runtime
- examples for `siliconflow`, `together`, and `fireworks` should stay framed as compat preset examples unless promotion criteria are explicitly met

## Reviewer checklist

When reviewing migration patches, ask:

- does this change reduce user dependence on generic client downcasting?
- does this example show the preferred stable path?
- is a builder used for convenience rather than necessity?
- should this API live under stable, extension, or compatibility?
