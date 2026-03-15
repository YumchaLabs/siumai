# Fearless Refactor V4 - Design

Last updated: 2026-03-06

## Context

The workspace already completed several important refactor steps:

- `siumai-spec` contains provider-agnostic request/response/message/tool types.
- `siumai-core` contains runtime concerns such as HTTP, streaming, retry, middleware, and traits.
- `siumai-registry` contains provider registration, factory routing, and model handles.
- `siumai-provider-*` crates contain provider-specific implementations.

This is a strong foundation, but the current architecture is still in a transitional state.
The public story is already moving toward model-family APIs and registry-first construction.
The internal execution path has now moved to family-native delegation for text, embedding,
and image, while some compatibility and extension surfaces still rely on generic `LlmClient`
bridges and capability downcasts.

The next refactor should finish that transition.

For the public-facing construction ladder and surface ownership rules, see `public-api-story.md`.

## Executive summary

This workstream adopts the **implementation structure** of the Vercel AI SDK, but not a
1:1 copy of its public naming.

We will:

1. Keep Rust-first naming and module layout.
2. Keep `siumai-spec` as the canonical provider-agnostic contract layer.
3. Promote model-family traits to the architectural center.
4. Demote `LlmClient` to an internal compatibility/runtime abstraction.
5. Keep builders as an ergonomic layer, but not as the primary architectural entry.

## Design principles

### P1 - Rust-first public API

We should be inspired by AI SDK structure, not mechanically translated from TypeScript.

That means:

- Module names should remain concise and idiomatic in Rust.
- Trait names should describe capability and lifecycle clearly.
- Request/response structs should keep domain-oriented names where those names are already good.
- Function naming can stay family-oriented and explicit, such as `text::generate` and `text::stream`.

### P2 - One architectural center

The current design has overlapping centers:

- model-family APIs
- registry handles
- provider builders
- generic `LlmClient`
- capability traits

V4 must reduce this to a single center:

- **family model traits are the stable execution center**

Everything else should either construct them, adapt to them, or extend them.

### P3 - Preserve ergonomics without preserving coupling

Convenience is valuable.
Architectural coupling is expensive.

Therefore:

- keep builder ergonomics
- remove builder-only logic
- ensure every builder maps to the same config-first construction path

### P4 - Provider complexity stays below the trait boundary

Provider implementations are messy by nature:

- request shape drift
- tool call variants
- streamed event inconsistencies
- multimodal edge cases
- provider-specific options and response metadata

That complexity should live in provider crates and protocol adapters, not in the core model-family traits.

## Decision 1 - Keep builders, but change their role

## Recommendation

Keep builders, but redefine them as **ergonomic frontends** instead of **architectural anchors**.

This means:

- `Siumai::builder()` may remain for compatibility and discoverability.
- provider builders such as `Provider::openai()` may remain for convenience.
- builders must become thin wrappers over the same config objects used by config-first construction.
- new features must not be builder-only.

## Why keep them

Builders are genuinely useful in Rust for:

- progressive configuration
- optional transport configuration
- tests and examples
- discoverability in IDE autocomplete
- reducing constructor overloads

Removing builders entirely would reduce ergonomics without creating architectural clarity by itself.

## What must change

Builders should no longer be allowed to define a separate execution path.

The internal rule should be:

```text
Builder -> Config -> Provider model/client constructor -> Family model trait object
```

Not:

```text
Builder -> special hidden client graph -> compatibility wrappers -> family API
```

## Public guidance

Recommended order for new code:

1. Registry-first for application code.
2. Config-first provider construction for provider-specific code.
3. Builder-first for convenience, tests, and quick setup.

Illustrative direction:

- application code: `registry::global().language_model("openai:gpt-4o-mini")?`
- provider-specific code: `OpenAiProvider::from_config(config)`
- convenience code: `Provider::openai().api_key(...).model(...).build()`

## Decision 2 - Introduce a strict family-model core

The architecture should follow this conceptual chain:

```text
High-level functions -> Family model traits -> Provider implementations
```

Not:

```text
High-level functions -> Generic client -> capability downcast -> provider implementation
```

## Proposed trait layers

### A. Stable family model traits

These traits become the main execution contracts.

Suggested direction:

- `LanguageModel`
- `EmbeddingModel`
- `ImageModel`
- `RerankingModel`
- `SpeechModel`
- `TranscriptionModel`

Versioned aliases or versioned trait names are still acceptable if we want long-term evolution,
for example `LanguageModelV1`/`LanguageModelV2` or `LanguageModelV4`.
The important point is that these are the primary contracts.

### B. Shared metadata trait

Introduce a lightweight shared metadata trait for all models.

Example direction:

```rust
pub trait ModelMetadata {
    fn provider_id(&self) -> &str;
    fn model_id(&self) -> &str;
    fn specification_version(&self) -> ModelSpecVersion;
}
```

This keeps common metadata separate from execution behavior.

### C. Optional extension traits

Provider-specific or non-unified features should stay in extension traits.

Examples:

- hosted tools extensions
- provider-owned hosted-search extensions when semantics are not yet stable across providers
- response administration APIs
- provider-native moderation resources
- websocket/realtime features
- image extras
- transcription extras

These should not pollute the stable family model traits.

## Decision 3 - Reposition `LlmClient`

`LlmClient` should stop being the default abstraction users mentally depend on.

It can remain useful for:

- cache entries
- trait object storage in transitional code
- internal compatibility bridges
- low-level registry internals

But the target direction is:

- public functions accept family models
- provider factories return family models
- registry handles implement family models directly

`LlmClient` becomes a transitional/internal abstraction rather than the center of the design.

## Decision 4 - Redesign provider factory contracts

Current factories conceptually return a generic client and let handles downcast into a capability.

V4 should move to family-returning factories.

Illustrative direction:

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
}
```

This is the single most important structural shift in the refactor.

## Decision 5 - Registry handles should be model objects

Registry handles should behave like first-class model objects, not client proxies.

That means a language-model handle should directly implement `LanguageModel`,
an embedding-model handle should directly implement `EmbeddingModel`, and so on.

The handle is still allowed to:

- apply middleware
- apply provider-id/model-id override logic
- lazily build or cache provider-backed implementations
- inject retry/http config/interceptors

But it should do so behind the family trait boundary.

## Decision 6 - Keep `siumai-spec` naming where it is already good

We should not rename types just to look more like AI SDK.

Good existing examples that should remain unless there is a strong reason to change:

- `ChatRequest`
- `ChatResponse`
- `ChatMessage`
- `ContentPart`
- `ImageGenerationRequest`
- `RerankRequest`
- `TtsRequest`
- `SttRequest`

However, the top-level function families can still expose Rust-first names such as:

- `text::generate`
- `text::stream`
- `embedding::embed`
- `embedding::embed_many`
- `image::generate`
- `rerank::rerank`
- `speech::synthesize`
- `transcription::transcribe`

This gives us AI SDK-like architecture with Rust-native API design.

## Decision 7 - Separate family traits from legacy capability traits

The current capability traits remain useful as migration shims, but should not remain the final center.

Recommended policy:

- family traits are stable and preferred
- capability traits are compatibility/internal adaptation tools
- provider crates may implement capability traits temporarily during migration
- new provider work should target family traits first

## Decision 8 - Audio needs explicit separation

Audio is currently a transitional overlap area.

Target direction:

- `SpeechModel` owns text-to-speech
- `TranscriptionModel` owns speech-to-text
- optional extras live in extension traits
- a catch-all `AudioCapability` can remain temporarily for compatibility only

This avoids mixing TTS/STT semantics inside a broad compatibility trait.

## Target module shape

Recommended long-term stable surface in `siumai`:

- `siumai::text`
- `siumai::embedding`
- `siumai::image`
- `siumai::rerank`
- `siumai::speech`
- `siumai::transcription`
- `siumai::provider_ext::<provider>`
- `siumai::prelude::unified`
- `siumai::registry`

Compatibility-only surface:

- `siumai::compat`
- `Siumai::builder()` compatibility story
- legacy builder-centric wrappers

## Migration strategy

### Phase A - Freeze the contract layer

- treat `siumai-spec` as canonical
- avoid further casual renames
- document stable naming guidance

### Phase B - Add family-centered internal contracts

- add final family traits
- add shared metadata trait
- add adapters from existing capability traits

### Phase C - Migrate registry and factories

- make factories return family models
- make handles implement family traits directly
- keep cache/middleware/build-context behavior

### Phase D - Migrate providers one by one

Priority order:

1. OpenAI
2. Anthropic
3. Gemini
4. OpenAI-compatible base
5. Secondary providers

### Phase E - Shrink the compatibility surface

- move builder-only behavior into thin wrappers
- stop adding new public APIs to compat layers
- reduce direct user dependence on `LlmClient`

## Performance guardrails

- adapter assembly should happen during construction, not per request
- registry caches should store final family-model objects where possible
- request transformation should happen at most once per execution path
- config-first convenience APIs should reuse existing runtime layers rather than add parallel translation steps; wrapper providers may add provider-owned config/client entry types only if translation is still finalized during construction time
- new parity work should include at least smoke-level timing sanity checks on hot text generation and streaming paths
- streaming executors must honor injected transports directly and keep JSON-line decoding single-pass so no-network parity does not hide buffering or fallback regressions

## Acceptance standard for the refactor

The refactor is successful when all of the following are true:

1. The recommended public API no longer depends conceptually on `LlmClient`.
2. Registry handles are valid family model objects.
3. Builders remain convenient but no longer define architecture.
4. Provider complexity is isolated in provider/protocol crates.
5. Naming remains idiomatic for Rust rather than cloned from TypeScript.

## Final recommendation

The right strategy is **not** to remove builders and **not** to clone AI SDK naming.

The right strategy is:

- keep builders
- slim builders down
- copy the architectural layering
- keep Rust naming
- move the execution center to family model traits

That gives `siumai` a cleaner long-term architecture without sacrificing ergonomics.

