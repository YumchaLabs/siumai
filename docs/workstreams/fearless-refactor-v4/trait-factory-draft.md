# Fearless Refactor V4 - Trait and Factory Draft

Last updated: 2026-03-06

## Spike status

The following parts of this draft now have working code spikes in the repository:

- shared `ModelMetadata` contract
- minimal `LanguageModel` family trait
- minimal `EmbeddingModel` family trait
- minimal `ImageModel` family trait
- `ProviderFactory` parallel text-family-returning interface
- default bridge from generic client path to family model path
- native OpenAI text-family factory path
- native Anthropic text-family factory path
- native OpenAI-compatible text-family factory path
- native Gemini text-family factory path
- default embedding-family bridge path
- native OpenAI-compatible embedding-family factory path
- native Gemini embedding-family factory path
- default image-family bridge path
- native OpenAI image-family factory path
- native Gemini image-family factory path
- native Groq text-family factory path
- native xAI text-family factory path
- native DeepSeek text-family factory path
The following parts remain design-only:

- final non-text family traits
- speech/transcription/reranking family traits
- provider-wide migration beyond text/embedding/image
- builder/config parity cleanup across all providers

Current implementation note:

- text, embedding, and image registry handles now execute through cached family-model objects
- image editing and variation still use the generic client path because they remain extension-only in V4
- OpenAI, Anthropic, Gemini, OpenAI-compatible, xAI, Groq, DeepSeek, and Ollama builders now emit canonical provider configs via `into_config()` before client construction or wrapper handoff, and xAI / DeepSeek additionally own provider-level config/client entry types over the shared runtime
- OpenAI-compatible config-first construction now also has fluent reasoning/thinking convenience, and Gemini config-first construction now has fluent generation/thinking convenience, reducing builder-only ergonomics drift
- DeepSeek now also keeps dedicated registry registration, default-model catalog metadata, and `SiumaiBuilder::new().deepseek()` available in deepseek-only builds without depending on the OpenAI registry feature
- Builder method availability should follow provider package ownership; dedicated methods such as `deepseek()` compile only with their provider feature instead of piggybacking on generic compat features
- Public `ProviderType` mapping now treats DeepSeek as a first-class provider rather than a custom OpenAI-compatible id, which keeps catalog and retry semantics aligned with the dedicated registry route
- Shared compat runtimes may still power wrapper providers internally, but registry/unified public surfaces should materialize provider-owned clients (for example `GroqClient` / `XaiClient` / `DeepSeekClient`) rather than exposing generic adapter client types
- Shared compat runtimes must preserve full-request semantics: request-level provider options should survive `chat_request` / `chat_stream_request`, and missing request fields should be filled from client defaults rather than degrading to message/tool-only fallbacks
- Provider-owned typed request option helpers should live beside those provider clients (for example `DeepSeekOptions` + `DeepSeekChatRequestExt`) instead of forcing apps onto raw `provider_options_map` plumbing
- Wrapper providers that keep shared runtimes should still own provider-specific request normalization hooks when the wire contract diverges (for example DeepSeek camelCase reasoning options normalized by `DeepSeekSpec`)
- Provider-owned typed metadata helpers should follow the same rule (for example `GroqMetadata`, `XaiMetadata`, and `DeepSeekMetadata` plus their `*ChatResponseExt` helpers) instead of pushing response consumers onto raw nested metadata maps

## Purpose

This document is a coding-oriented draft for the V4 architecture.
It translates the refactor direction into concrete trait and factory design targets.

It is intentionally opinionated, but still a draft.
The goal is to guide implementation sequencing and review discussions.

## Design goals

1. Make family model traits the primary execution contracts.
2. Keep provider-specific complexity below the trait boundary.
3. Let registry handles behave as first-class model objects.
4. Preserve current strengths such as cache, TTL, middleware, and build context injection.
5. Avoid unnecessary renames in `siumai-spec`.

## Non-goals

- Do not mirror AI SDK naming verbatim.
- Do not introduce a large amount of trait-version churn unless there is a clear migration need.
- Do not force all providers to expose identical non-family extension resources.

## Proposed layering

```text
siumai facade functions
    -> family model traits
    -> registry handles / provider-native model objects
    -> provider + protocol implementation details
```

## Shared metadata contract

The family model traits should share a lightweight metadata contract.

Illustrative direction:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSpecVersion {
    V1,
}

pub trait ModelMetadata: Send + Sync {
    fn provider_id(&self) -> &str;
    fn model_id(&self) -> &str;
    fn specification_version(&self) -> ModelSpecVersion {
        ModelSpecVersion::V1
    }
}
```

Rationale:

- keeps provider/model identity uniform
- avoids repeating metadata methods on every family trait
- gives room for future trait versioning without hard-coding Vercel naming

## Family model traits

## 1. Language model

The language model is the family behind `text::generate` and `text::stream`.

Suggested direction:

```rust
#[async_trait]
pub trait LanguageModel: ModelMetadata {
    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse, LlmError>;

    async fn stream(&self, request: ChatRequest) -> Result<TextStreamHandle, LlmError>;
}
```

Notes:

- `ChatRequest` and `ChatResponse` remain acceptable names at the spec layer.
- `text` is the family module name; it does not require the request type to be renamed.
- `TextStreamHandle` is preferable to a raw stream because it preserves first-class cancellation.

## 2. Embedding model

```rust
#[async_trait]
pub trait EmbeddingModel: ModelMetadata {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError>;
}
```

If batching remains a common need, either:

- keep it in `EmbeddingRequest`, or
- expose a separate higher-level function that batches above the model trait

Prefer keeping provider batching complexity outside the core trait if possible.

## 3. Image model

```rust
#[async_trait]
pub trait ImageModel: ModelMetadata {
    async fn generate(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError>;
}
```

Provider-native image editing/variation should remain extension-oriented unless we intentionally promote them.

## 4. Reranking model

```rust
#[async_trait]
pub trait RerankingModel: ModelMetadata {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError>;
}
```

## 5. Speech model

```rust
#[async_trait]
pub trait SpeechModel: ModelMetadata {
    async fn synthesize(&self, request: TtsRequest) -> Result<TtsResponse, LlmError>;
}
```

Streaming TTS should live in an extension trait if we want to keep the stable core minimal.

## 6. Transcription model

```rust
#[async_trait]
pub trait TranscriptionModel: ModelMetadata {
    async fn transcribe(&self, request: SttRequest) -> Result<SttResponse, LlmError>;
}
```

Audio translation and segment-heavy provider extras should remain in extension traits unless explicitly promoted.

## Why family traits should not inherit from legacy capability traits

We should avoid designs like:

```rust
pub trait LanguageModel: ChatCapability {}
```

Reason:

- it preserves the old abstraction hierarchy
- it keeps the family model layer dependent on compatibility-era contracts
- it makes future cleanup harder

Instead, use adapters during migration.

## Adapter strategy during migration

Legacy capability traits can temporarily adapt into the new family traits.

Illustrative direction:

```rust
#[async_trait]
impl<T> LanguageModel for T
where
    T: ChatCapability + ModelMetadata + Send + Sync,
{
    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_request(request).await
    }

    async fn stream(&self, request: ChatRequest) -> Result<TextStreamHandle, LlmError> {
        self.chat_stream_request_with_cancel(request).await
    }
}
```

This is a migration bridge, not the final conceptual architecture.

## Factory contract draft

The provider factory should return family model objects directly.

Suggested direction:

```rust
#[async_trait]
pub trait ProviderFactory: Send + Sync {
    async fn language_model(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LanguageModel>, LlmError>;

    async fn embedding_model(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn EmbeddingModel>, LlmError>;

    async fn image_model(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn ImageModel>, LlmError>;

    async fn speech_model(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn SpeechModel>, LlmError>;

    async fn transcription_model(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn TranscriptionModel>, LlmError>;

    async fn reranking_model(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn RerankingModel>, LlmError>;

    fn provider_id(&self) -> Cow<'static, str>;
    fn capabilities(&self) -> ProviderCapabilities;
}
```

## Build context draft

`BuildContext` should remain shared across all model-family constructors.

Recommended contents:

- HTTP interceptors
- HTTP config
- retry options
- optional provider-id override
- optional common params defaults where still meaningful
- telemetry/tracing hooks if needed

Build context should describe environment and construction policy, not per-request semantics.

Per-request semantics belong in spec request types.

## Registry handle draft

Registry handles should implement family traits directly.

Example direction for a language-model handle:

```rust
pub struct LanguageModelHandle {
    factory: Arc<dyn ProviderFactory>,
    provider_id: String,
    model_id: String,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    cache: Arc<TokioMutex<...>>,
    client_ttl: Option<Duration>,
    build_context: HandleBuildContext,
}

#[async_trait]
impl LanguageModel for LanguageModelHandle {
    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        // apply middleware
        // resolve possibly overridden model id
        // get cached family model object or build one
        // delegate generate
    }

    async fn stream(&self, request: ChatRequest) -> Result<TextStreamHandle, LlmError> {
        // same pattern for stream
    }
}
```

Important point:

- the cache should preferably store family model objects, not generic client objects

That keeps the post-refactor architecture honest.

## Builder convergence draft

Provider builders should emit canonical config structs.

Example direction:

```rust
let config = Provider::openai()
    .api_key("...")
    .model("gpt-4o")
    .into_config()?;

let provider = OpenAiProvider::from_config(config)?;
```

Or, if builders continue to expose `.build()`:

```rust
pub async fn build(self) -> Result<OpenAiProvider, LlmError> {
    let config = self.into_config()?;
    OpenAiProvider::from_config(config)
}
```

The exact API can vary, but the convergence rule must hold.

## Audio cleanup draft

Target ownership:

- `SpeechModel` owns TTS
- `TranscriptionModel` owns STT
- `AudioCapability` remains compatibility-only during migration

Suggested extension split:

- `SpeechExtras` for streaming TTS and voice listing
- `TranscriptionExtras` for streaming STT, translation, language listing

## Recommended migration order

1. lock family trait signatures
2. adapt current handles behind those signatures
3. redesign `ProviderFactory`
4. migrate OpenAI to direct family model return types
5. migrate Anthropic and Gemini
6. migrate OpenAI-compatible base
7. finish secondary providers

## Open questions

These questions should be resolved before locking implementation:

1. Should `LanguageModel::stream` return a cancellable handle by default or a plain stream plus an alternate cancel path?
2. Should embeddings expose a minimal single-call contract only, with batching handled above the trait?
3. Image editing/variation remains extension-only for V4.
4. Should model traits include capability metadata directly, or should capability hints stay separate from execution contracts?

## Recommended decision baseline

If we want the smallest practical pivot, the baseline should be:

- keep current spec type names
- use family traits as the new execution center
- keep adapters temporarily
- redesign factories and handles around family objects
- converge builders onto config-first construction

That is enough to complete the architectural shift without forcing unnecessary public churn.



