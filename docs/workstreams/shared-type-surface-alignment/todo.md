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
- [x] Add public compile-guard coverage in `siumai/tests/public_surface_imports_test.rs`.
- [x] Add local unit coverage for warning/metadata/usage conversion behavior.
- [x] Expose Rust-style equivalents of the AI SDK usage helpers:
  `create_null_language_model_usage`, `add_language_model_usage`, and
  `add_image_model_usage`.

## Track D - Docs and changelog

- [x] Create a dedicated `docs/workstreams/shared-type-surface-alignment/` folder.
- [x] Record the shared type-surface alignment slice in `CHANGELOG.md` `Unreleased`.

## Track E - Intentional deferrals

- [-] `RequestOptions`, `TimeoutConfiguration`, and `LanguageModelCallOptions` are tracked by their
  own dedicated workstreams.
- [-] `EmbeddingModelMiddleware` and `ImageModelMiddleware` are deferred until those model families
  have real middleware execution hooks; do not expose empty placeholder traits.
- [-] Do not pretend that every provider already captures speech/transcription response bodies in a
  stable cross-provider way.
