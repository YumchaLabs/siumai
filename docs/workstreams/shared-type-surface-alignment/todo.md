# Shared Type Surface Alignment - TODO

Last updated: 2026-04-24

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Shared public names

- [x] Audit `repo-ref/ai/packages/ai/src/types/index.ts` and the upstream shared metadata/usage
  files it re-exports.
- [x] Add a dedicated `siumai-spec/src/types/ai_sdk.rs` module for shared AI SDK-style public
  names.
- [x] Expose `JSONSchema7`, `JSONValue`, `CallWarning`, `ProviderMetadata`, and
  `ImageModelProviderMetadata`.
- [x] Expose `ProviderOptions` and `Context`.
- [x] Expose the shared `Embedding` vector alias from `types/embedding-model.ts`.
- [x] Expose the shared language-model `Source` citation shape from `types/language-model.ts`.
- [x] Expose provider-utils-style `ToolCall` and `ToolResult`.
- [x] Align shared `ToolChoice` serde with the AI SDK forced-tool object shape while preserving
  legacy Rust enum object deserialization.
- [x] Align shared `FinishReason` serde with the AI SDK kebab-case public values while preserving
  provider snake_case input compatibility.
- [x] Expose `EmbeddingModelUsage`, `ImageModelUsage`, and `LanguageModelUsage`.
- [x] Expose `LanguageModelRequestMetadata`, `LanguageModelResponseMetadata`,
  `ImageModelResponseMetadata`, `SpeechModelResponseMetadata`, and
  `TranscriptionModelResponseMetadata`.

## Track B - Runtime carrier parity

- [x] Add optional `headers` to stable `ResponseMetadata`.
- [x] Add the shared warning `deprecated` category to `Warning`.
- [x] Add honest conversions from `Usage`, `HttpRequestInfo`, `HttpResponseInfo`, and
  `ResponseMetadata` into the new shared public structs.
- [x] Keep speech response `body` optional instead of fabricating runtime capture.

## Track C - Facade and tests

- [x] Re-export the new shared names from `siumai::prelude::unified::*`.
- [x] Re-export the existing stable `EmbeddingModel` trait from `siumai::prelude::unified::*` so
  the direct model-family names match the audited AI SDK `types/index.ts` surface.
- [x] Re-export the existing runtime `LanguageModelMiddleware` trait from
  `siumai::prelude::unified::*`.
- [x] Re-export the implemented video family module and `VideoModel*` traits from
  `siumai::prelude::unified::*`.
- [x] Re-export `ProviderFactory` directly from `siumai::prelude::unified::*` as the honest Rust
  provider-interface equivalent while leaving historical `siumai::Provider` on compat/top-level
  construction paths.
- [x] Add public compile-guard coverage in `siumai/tests/public_surface_imports_test.rs`.
- [x] Add local unit coverage for warning/metadata/usage conversion behavior.
- [x] Expose Rust-style equivalents of the AI SDK usage helpers:
  `create_null_language_model_usage`, `add_language_model_usage`, and
  `add_image_model_usage`.

## Track D - Docs and changelog

- [x] Create a dedicated `docs/workstreams/shared-type-surface-alignment/` folder.
- [x] Add a shared type-surface audit matrix for `repo-ref/ai/packages/ai/src/types/*`.
- [x] Record the shared type-surface alignment slice in `CHANGELOG.md` `Unreleased`.

## Track E - Provider-utils schema surface

- [x] Audit `repo-ref/ai/packages/provider-utils/src/schema.ts`.
- [x] Expose `Schema`, `ValidationResult`, `FlexibleSchema`, and `LazySchema` through
  `siumai::types::*` and `siumai::prelude::unified::*`.
- [x] Expose Rust-style equivalents of AI SDK `jsonSchema`, `asSchema`, and `lazySchema`:
  `json_schema`, `as_schema`, `as_schema_or_empty`, and `lazy_schema`.
- [x] Keep Zod and TypeScript Standard Schema integration deferred instead of adding placeholders
  that cannot validate in Rust.

## Track F - Provider-utils ID surface

- [x] Audit `repo-ref/ai/packages/provider-utils/src/generate-id.ts`.
- [x] Expose `IdGenerator`, `IdGeneratorOptions`, `create_id_generator`, and `generate_id` from
  the root facade and `siumai::prelude::unified::*`.
- [x] Preserve upstream semantics: 16-character default random suffix, optional prefix/separator,
  custom alphabet, and non-cryptographic generation.
- [x] Return `Result` for invalid Rust options instead of mimicking JavaScript exceptions.

## Track G - Intentional deferrals

- [-] `RequestOptions`, `TimeoutConfiguration`, and `LanguageModelCallOptions` are tracked by their
  own dedicated workstreams.
- [-] `EmbeddingModelMiddleware` and `ImageModelMiddleware` are deferred until those model families
  have real middleware execution hooks; do not expose empty placeholder traits.
- [-] Do not pretend that every provider already captures speech/transcription response bodies in a
  stable cross-provider way.
- [-] Do not expose `zodSchema` or Standard Schema adapters until a real Rust schema backend owns
  the validation/conversion behavior.
