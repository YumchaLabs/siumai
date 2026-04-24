# Shared Type Surface Alignment - Audit

Last updated: 2026-04-24

This audit compares `repo-ref/ai/packages/ai/src/types/*` with the stable Rust facade. The goal is
to keep public type-surface checks mechanical and avoid false parity.

## `types/index.ts`

| AI SDK export | Rust surface | Status | Notes |
| --- | --- | --- | --- |
| `JSONSchema7` | `siumai::types::JSONSchema7`, `prelude::unified::JSONSchema7` | done | Alias to `serde_json::Value`, matching Rust's schema carrier. |
| `JSONValue` | `siumai::types::JSONValue`, `prelude::unified::JSONValue` | done | Alias to `serde_json::Value`. |
| `Embedding` | `siumai::types::Embedding`, `prelude::unified::Embedding` | done | Alias to `Vec<f32>`. |
| `EmbeddingModel` | `prelude::unified::EmbeddingModel`, `siumai::embedding::EmbeddingModel` | done | Real Rust model trait. |
| `ImageModel` | `prelude::unified::ImageModel`, `siumai::image::ImageModel` | done | Real Rust model trait. |
| `LanguageModel` | `prelude::unified::LanguageModel`, `siumai::text::LanguageModel` | done | Real Rust model trait. |
| `RerankingModel` | `prelude::unified::RerankingModel`, `siumai::rerank::RerankingModel` | done | Real Rust model trait. |
| `SpeechModel` | `prelude::unified::SpeechModel`, `siumai::speech::SpeechModel` | done | Real Rust model trait. |
| `TranscriptionModel` | `prelude::unified::TranscriptionModel`, `siumai::transcription::TranscriptionModel` | done | Real Rust model trait. |
| `Provider` | `prelude::unified::ProviderFactory` | done | `ProviderFactory` is the honest Rust provider-interface equivalent. Historical `siumai::Provider` remains a compat/top-level builder helper and is intentionally not treated as the AI SDK provider interface. |
| `ProviderMetadata` | `siumai::types::ProviderMetadata`, `prelude::unified::ProviderMetadata` | done | Alias to provider-id keyed metadata map. |
| `ProviderReference` | `siumai::types::ProviderReference`, `prelude::unified::ProviderReference` | done | Stable provider-reference map. |
| `CallWarning` / `Warning` | `siumai::types::{CallWarning, Warning}`, `prelude::unified::*` | done | Includes AI SDK `deprecated` warning category. |
| `FinishReason` | `siumai::types::FinishReason`, `prelude::unified::FinishReason` | done | Serializes AI SDK public values and accepts provider legacy values. |
| `ToolChoice` | `siumai::types::ToolChoice`, `prelude::unified::ToolChoice` | done | Forced-tool serde now uses `{ type: "tool", toolName: "..." }`. |
| `LanguageModelRequestMetadata` | `siumai::types::LanguageModelRequestMetadata`, `prelude::unified::*` | done | Converts from existing request info. |
| `LanguageModelResponseMetadata` | `siumai::types::LanguageModelResponseMetadata`, `prelude::unified::*` | done | Converts from existing response metadata. |
| `ImageModelResponseMetadata` | `siumai::types::ImageModelResponseMetadata`, `prelude::unified::*` | done | Converts from existing HTTP response info. |
| `SpeechModelResponseMetadata` | `siumai::types::SpeechModelResponseMetadata`, `prelude::unified::*` | done | `body` remains optional until runtime capture is universal. |
| `TranscriptionModelResponseMetadata` | `siumai::types::TranscriptionModelResponseMetadata`, `prelude::unified::*` | done | Converts from existing HTTP response info. |
| `EmbeddingModelUsage` | `siumai::types::EmbeddingModelUsage`, `prelude::unified::*` | done | AI SDK one-field usage shape. |
| `ImageModelUsage` | `siumai::types::ImageModelUsage`, `prelude::unified::*` | done | AI SDK image usage token totals. |
| `LanguageModelUsage` | `siumai::types::LanguageModelUsage`, `prelude::unified::*` | done | Projection from stable `Usage`. |
| `LanguageModelMiddleware` | `prelude::unified::LanguageModelMiddleware` | done | Real runtime middleware trait. |
| `EmbeddingModelMiddleware` | none | deferred | Needs real embedding middleware execution hooks first. |
| `ImageModelMiddleware` | none | deferred | Needs real image middleware execution hooks first. |

## Additional audited files

| AI SDK file | Rust surface | Status | Notes |
| --- | --- | --- | --- |
| `types/language-model.ts` `Source` | `siumai::types::Source`, `prelude::unified::Source` | done | Independent shared source citation shape with fixed `type: "source"`. |
| `types/video-model.ts` `VideoModel` | `prelude::unified::VideoModel`, `siumai::video::VideoModel` | done | Rust keeps task-oriented execution semantics. |
| `types/video-model.ts` `VideoModelProviderMetadata` | `siumai::types::VideoModelProviderMetadata`, `prelude::unified::*` | done | Alias to shared provider metadata. |
| `types/video-model-response-metadata.ts` | `siumai::types::VideoModelResponseMetadata`, `prelude::unified::*` | done | Includes optional provider metadata. |

## `@ai-sdk/provider-utils` root schema exports

| AI SDK export | Rust surface | Status | Notes |
| --- | --- | --- | --- |
| `Schema` | `siumai::types::Schema`, `prelude::unified::Schema` | done | JSON Schema carrier with optional Rust validator callback. |
| `ValidationResult` | `siumai::types::ValidationResult`, `prelude::unified::ValidationResult` | done | Explicit success/failure enum using `LlmError` for validation failures. |
| `FlexibleSchema` | `siumai::types::FlexibleSchema`, `prelude::unified::FlexibleSchema` | done | Supports concrete and lazy Rust schemas. |
| `jsonSchema` | `siumai::types::json_schema`, `prelude::unified::json_schema` | done | Rust-style name; creates a passive JSON Schema carrier. |
| `asSchema` | `siumai::types::as_schema`, `prelude::unified::as_schema` | done | Resolves concrete/lazy schemas. `as_schema_or_empty` covers the upstream `undefined` fallback. |
| `lazySchema` | `siumai::types::lazy_schema`, `prelude::unified::lazy_schema` | done | Cached lazy schema initialization. Upstream does not export this from `packages/ai/src/index.ts`, but provider-utils owns it. |
| `zodSchema` | none | deferred | Zod is TypeScript-specific. Rust should expose real schema-library adapters only when a Rust validator integration is added. |
| `StandardSchema` support | none | deferred | TypeScript standard-schema interop has no direct Rust equivalent in this crate today. |

## `@ai-sdk/provider-utils` root ID exports

| AI SDK export | Rust surface | Status | Notes |
| --- | --- | --- | --- |
| `IdGenerator` | `siumai::IdGenerator`, `prelude::unified::IdGenerator` | done | Cloneable `Arc<dyn Fn() -> String + Send + Sync>` generator. |
| `createIdGenerator` | `siumai::create_id_generator`, `prelude::unified::create_id_generator` | done | Rust-style name with `IdGeneratorOptions`; returns `Result` instead of throwing. |
| `generateId` | `siumai::generate_id`, `prelude::unified::generate_id` | done | Generates the AI SDK-compatible default 16-character non-cryptographic ID. |

## Current deferred work

- Add real `EmbeddingModelMiddleware` only after embedding helper/runtime calls can apply middleware.
- Add real `ImageModelMiddleware` only after image helper/runtime calls can apply middleware.
- Add Rust schema-library adapters only when they can validate through a real Rust backend; do not
  expose empty `zodSchema`/standard-schema placeholders.
- Revisit whether historical `siumai::Provider` should be renamed or moved further into compat in a
  breaking facade cleanup. Do not alias it to AI SDK `Provider`; the semantics are different.
