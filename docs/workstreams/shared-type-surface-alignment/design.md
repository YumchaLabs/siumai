# Shared Type Surface Alignment - Design

Last updated: 2026-04-24

## Problem

Compared with the shared AI package surface in `repo-ref/ai/packages/ai/src/types/index.ts`,
Siumai already had most of the runtime carriers, but the public shared type layer still drifted in
several important ways:

- the public Rust surface did not expose audited shared names such as `JSONValue`,
  `JSONSchema7`, `CallWarning`, `ProviderMetadata`, `ImageModelProviderMetadata`,
  `LanguageModelRequestMetadata`, `LanguageModelResponseMetadata`,
  `ImageModelResponseMetadata`, `SpeechModelResponseMetadata`,
  `TranscriptionModelResponseMetadata`, `EmbeddingModelUsage`, and `ImageModelUsage`
- the stable runtime carrier `ResponseMetadata` had no `headers` slot, so the public Rust surface
  could not honestly mirror the upstream response-metadata contract
- the shared warning surface still missed the AI SDK `deprecated` category
- the stable Rust prelude did not re-export these shared data structures, which kept public audits
  noisy even when the internal runtime had most of the needed information
- the current `Usage` type is a compatibility-oriented superset of provider V4 usage, but the AI
  package also exposes a higher-level `LanguageModelUsage` projection that Siumai did not model
  explicitly
- the AI SDK language-model `Source` citation shape was only available indirectly as a content
  part variant, not as a stable shared public data structure
- `LanguageModelMiddleware` existed in the Rust runtime but was not directly visible from the
  stable facade, while embedding/image middleware did not yet have honest runtime equivalents
- the root AI SDK package also re-exports schema helpers from `@ai-sdk/provider-utils`, but Siumai's
  existing `OutputSchema` / `SchemaValidator` types did not provide the same public `Schema`
  carrier contract
- the root AI SDK package re-exports `createIdGenerator`, `generateId`, and `IdGenerator`, but
  Siumai did not have a shared facade-level equivalent
- the root AI SDK package re-exports provider-utils tool helpers directly, while Siumai's real
  runtime tool helpers were only visible through the nested `tooling` module
- the root AI SDK package re-exports `parseJsonEventStream`, but Siumai only exposed the lower-level
  provider/runtime SSE JSON parser

This was not a single runtime bug. It was a shared contract gap:

- public comparison against `repo-ref/ai/packages/ai/src/types/*` stayed harder than necessary
- metadata and usage helpers had to reuse nearby internal carriers with mismatched names or shapes
- downstream Rust code had no stable, audited entry point for the AI SDK-style shared data layer

## Goals

- Audit the shared AI package surface as a public contract, not only as internal runtime data.
- Expose the stable shared names that already have honest Rust equivalents.
- Add missing shared metadata carriers where the runtime can provide them without pretending that
  all wiring is already complete.
- Keep AI SDK model-family names directly importable from the stable facade when a real Rust trait
  already exists.
- Keep provider-owned typed options and typed provider metadata in provider crates.
- Make future audits against `repo-ref/ai/packages/ai/src/types/*` cheaper and more mechanical.

## Non-goals

- Do not mirror every TypeScript prompt/helper type mechanically.
- Do not introduce `RequestOptions`, `TimeoutConfiguration`, or
  `LanguageModelCallOptions` in this workstream without a separate design.
- Do not fabricate richer runtime capture than Siumai currently has; for example, speech response
  `body` stays optional because the shared carrier is now present before the runtime is fully
  wired.
- Do not pretend that TypeScript-only Zod or Standard Schema objects exist in Rust. Rust adapters
  should be added only when backed by a real Rust schema conversion/validation implementation.

## Chosen design

### 1. Add one dedicated shared-type module in `siumai-spec`

`siumai-spec/src/types/ai_sdk.rs` becomes the public home for shared AI SDK-style names that are
not provider-owned:

- aliases:
  - `JSONSchema7`
  - `JSONValue`
  - `CallWarning`
  - `ProviderMetadata`
  - `ProviderOptions`
  - `Context`
  - `Embedding`
  - `Source`
  - `ImageModelProviderMetadata`
- usage/data structures:
  - `ToolCall`
  - `ToolResult`
  - `LanguageModelUsage`
  - `LanguageModelInputTokenDetails`
  - `LanguageModelOutputTokenDetails`
  - `EmbeddingModelUsage`
  - `ImageModelUsage`
  - `LanguageModelRequestMetadata`
  - `LanguageModelResponseMetadata`
  - `ImageModelResponseMetadata`
  - `SpeechModelResponseMetadata`
  - `TranscriptionModelResponseMetadata`

This keeps the shared audit surface local instead of scattering these names across unrelated
runtime modules.

### 2. Use honest conversion helpers instead of fake direct aliases

Some names can be direct aliases (`JSONValue`, `ProviderMetadata`), but the more
important shared carriers need explicit Rust structs:

- `LanguageModelUsage` is now a projection from `Usage`, because the AI package shape is not the
  same as provider V4 usage
- `ProviderOptions` and `Context` are honest aliases onto the existing open JSON-object carriers
- `ToolCall` and `ToolResult` are passive Rust data structures that mirror the provider-utils
  helper shape, including output-side metadata such as `providerMetadata`, `title`, invalid-tool
  `invalid` / `error`, and `preliminary`, without pretending Siumai already has one exact runtime
  source for every typed tool helper result
- `EmbeddingModelUsage` stays the audited one-field shape `{ tokens }`
- `ImageModelUsage` mirrors the AI package `ImageModelV4Usage` token totals
- `CallWarning` mirrors the strict AI SDK shared V4 warning union rather than aliasing the wider
  stable `Warning` compatibility enum; legacy warning variants normalize through conversion
  helpers before reaching AI SDK result payloads
- `Source` mirrors the AI package language-model source union with a fixed `type: "source"` marker
  and a strict `sourceType: "url" | "document"` payload
- `ToolChoice` remains the stable Rust enum but now serializes the forced-tool case as the AI SDK
  `{ type: "tool", toolName: "..." }` object instead of serde's externally tagged enum shape
- `FinishReason` remains the stable Rust enum but now serializes the AI SDK public values
  (`tool-calls`, `content-filter`, `other`) and accepts provider snake_case values on input
- request/response metadata structs expose the AI package fields directly and convert from the
  existing runtime carriers

The upstream `usage.ts` helper functions are also mirrored with Rust-style names:

- `create_null_language_model_usage()`
- `add_language_model_usage(...)`
- `add_image_model_usage(...)`

These helpers are thin wrappers around the stable usage structs. Aggregated language-model usage
intentionally drops raw provider usage metadata, matching the upstream aggregate helper behavior.

This avoids the trap of “matching the name but not the data structure”.

The provider-utils schema helpers are mirrored as honest Rust carriers:

- `Schema<T>` always owns a provider-facing `JSONSchema7`
- `ValidationResult<T>` models the AI SDK success/failure union with `LlmError` failures
- `FlexibleSchema<T>` accepts concrete and lazily-created Rust schemas
- `json_schema(...)`, `json_schema_with_validator(...)`, `lazy_schema(...)`,
  `as_schema(...)`, `as_schema_or_empty(...)`, and `empty_json_schema(...)` provide Rust-style
  equivalents of the upstream helper flow

The optional validator is a Rust callback. If a schema has no validator, callers can still pass the
JSON Schema to providers, but `Schema::validate(...)` returns `None` instead of fabricating a
validation result.

The provider-utils ID helpers are mirrored in `siumai-core::utils` and re-exported from the root
facade:

- `IdGenerator` is a cloneable `Arc<dyn Fn() -> String + Send + Sync>`
- `IdGeneratorOptions` mirrors prefix, separator, size, and alphabet controls
- `create_id_generator(...)` validates options and returns `Result<IdGenerator, LlmError>`
- `generate_id()` produces the default 16-character random ID

The implementation intentionally remains non-cryptographic, matching AI SDK's helper. Rust uses
`Result` for invalid options instead of modeling JavaScript exceptions.

The provider-utils tool surface is split into two honest Rust concepts:

- `siumai::types::Tool` remains the passive provider-facing schema shape
- `ExecutableTool` binds a passive tool to Rust closures, callbacks, approval checks, and
  to-model-output conversion
- `ToolSet` remains an alias to `ExecutableTools`
- `ToolExecuteFunction` is an alias to the options-aware runtime callback type
- `tool(...)` and `dynamic_tool(...)` are re-exported from the root facade and the unified prelude

This avoids serializing Rust closures into the spec-level tool shape while still matching the
provider-utils runtime helper ergonomics.

The provider-utils stream parser is exposed as a thin public wrapper:

- `parse_json_event_stream(...)` parses byte streams containing SSE `data:` JSON payloads
- `[DONE]` is ignored by default, matching the upstream helper
- invalid JSON is returned through the Rust stream error channel as `LlmError::ParseError`

This intentionally does not clone the TypeScript `ParseResult` union. Rust callers already have a
standard stream item error channel.

### 3. Extend stable runtime carriers where the public contract requires it

Two shared runtime carriers needed real semantic widening:

- `ResponseMetadata` now includes optional `headers`
- `Warning` now includes the `Deprecated { setting, message }` category

Those are not cosmetic changes. They are stable contract slots used directly by the audited AI SDK
surface.

### 4. Keep runtime honesty for partially wired metadata

Some public shared structs are now available before every runtime path is fully upgraded:

- `SpeechModelResponseMetadata.body` remains optional and current HTTP-only conversions leave it as
  `None`
- `LanguageModelRequestMetadata.body` converts from the current serialized request body and falls
  back to raw string JSON values when parsing is not possible

This is intentional. The public data structures now exist, but they do not claim stronger runtime
capture than Siumai currently has.

### 5. Re-export the shared layer through the stable facade

The stable Rust facade now re-exports the shared names through:

- `siumai::types::*`
- `siumai::prelude::unified::*`

That keeps the shared surface visible in the same public places where users already import the
stable family/request/response contracts.

The existing Rust-first `EmbeddingModel` trait is also re-exported directly from
`siumai::prelude::unified::*` so the facade matches the audited AI SDK direct model-family export
shape alongside `LanguageModel`, `ImageModel`, `RerankingModel`, `SpeechModel`, and
`TranscriptionModel`.

The implemented video family is also exported from the unified prelude as `video`, `VideoModel`,
`VideoModelV3`, and `VideoModelV4`. Upstream keeps video under `types/video-model.ts` rather than
the stable `types/index.ts` export list today, but the Rust facade already owns a stable
task-oriented video family and should keep that family importable with the rest of the model
surface.

For the AI SDK `Provider` interface, the honest Rust equivalent is the registry `ProviderFactory`
trait because it owns model-family factory methods. The historical `siumai::Provider` type remains
a top-level/compat construction helper, so this workstream exposes `ProviderFactory` directly
instead of aliasing the old builder entry point to an incompatible provider-interface meaning.

The existing runtime `LanguageModelMiddleware` trait is also re-exported directly. Embedding and
image middleware are intentionally not fabricated in this slice because there is no corresponding
embedding/image middleware execution path yet.

The schema helpers are also re-exported from `siumai::prelude::unified::*` so root-package audits
against `repo-ref/ai/packages/ai/src/index.ts` can see the provider-utils schema surface without
pulling from implementation modules.

The ID helpers are root facade exports as well as prelude exports because AI SDK exposes them from
the package root, not only from a nested utility module.

The tool runtime helpers are also root facade exports. The historical `tool!` macro remains
available in macro syntax, while `tool(...)` is the value-level helper aligned with provider-utils.

`parse_json_event_stream` is root-exported and prelude-exported as the public wrapper around the
existing provider-agnostic SSE JSON parser.

## Validation

This workstream is currently locked by:

- `cargo nextest run -p siumai-spec schema --no-fail-fast`
- `cargo nextest run -p siumai-core id --no-fail-fast`
- `cargo nextest run -p siumai-core tooling --no-fail-fast`
- `cargo nextest run -p siumai-core sse_json --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test`
- public compile guards in `siumai/tests/public_surface_imports_test.rs`
- local unit tests in `siumai-spec/src/types/{common,ai_sdk,schema}.rs`

## Deferred follow-up

- Design `RequestOptions`, `TimeoutConfiguration`, and `LanguageModelCallOptions` separately.
- Design embedding/image middleware once those model families have real middleware execution hooks.
- Revisit response-body capture for speech/transcription once the runtime has a stable place to
  preserve it across providers.
- Add real Rust schema-library adapters if/when the crate adopts a validator/converter backend.
- Keep TypeScript-only inference helpers (`InferToolInput` / `InferToolOutput`) deferred unless a
  meaningful Rust generic API emerges.
