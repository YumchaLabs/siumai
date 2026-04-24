# Generate Object Structured Output Alignment - Design

Last updated: 2026-04-24

## Problem

`repo-ref/ai/packages/ai/src/generate-object/*` exposes `generateObject` as a high-level
structured-output helper. Siumai already had lower-level JSON extraction helpers and provider
request `ResponseFormat::json_schema(...)`, but no single Rust entry point that:

- accepts the provider-utils-style `Schema` / `FlexibleSchema` surface
- applies JSON Schema response format before calling a language model
- parses and validates the returned JSON into a typed object
- returns the AI SDK-style result metadata shape

That gap made object generation feel like an internal recipe instead of a stable facade-level
contract.

## Goals

- Add a non-streaming Rust helper equivalent to AI SDK `generateObject`.
- Reuse the existing text-family model call path, retry/request options, and JSON extraction logic.
- Keep schema input compatible with the recently aligned provider-utils schema carriers.
- Project the stable result fields that Siumai can populate honestly today.
- Make the remaining `streamObject` and output-strategy gaps explicit instead of hiding them behind
  placeholder APIs.

## Non-goals

- Do not invent a browser/HTTP `Response` wrapper for `toJsonResponse(...)`.
- Do not fabricate provider HTTP response bodies or headers when the current chat response does not
  carry them.
- Do not add streaming partial-object semantics until the runtime owns incremental structured JSON
  parsing rather than only final-response parsing.
- Do not expose TypeScript-only `InferSchema`-style helper aliases.

## Chosen design

### 1. Add `structured_output::generate_object`

The helper accepts:

- a `LanguageModel`
- a `TextRequest`
- a `GenerateObjectSchema<T>`
- `GenerateObjectOptions`

It sets `TextRequest.response_format` to `ResponseFormat::json_schema(...)`, then delegates to the
same text-family prepared request execution path used by `text::generate`.

This keeps retries, timeouts, headers, tools, tool choice, and telemetry behavior consistent with
the text family instead of creating a parallel call path.

### 2. Model schema input as an honest Rust enum

`GenerateObjectSchema<T>` accepts:

- `Schema<T>`
- `LazySchema<T>`
- `FlexibleSchema<T>`
- raw `JSONSchema7`

When a typed Rust validator exists, final object parsing uses `Schema::validate(...)`. When no
validator exists, the helper falls back to `serde_json::from_value(...)`.

This mirrors the upstream provider-utils flow without pretending Rust has Zod or TypeScript
Standard Schema objects.

### 3. Return an AI SDK-style result projection

`GenerateObjectResult<T>` contains:

- `object`
- `reasoning`
- `finish_reason`
- `usage`
- `warnings`
- `request`
- `response`
- `provider_metadata`
- `raw_response`

`response.id` follows the AI SDK behavior of falling back to an SDK-generated id when the provider
does not return one. `response.model_id` uses the provider response model when present, otherwise
the language-model metadata.

`request.body` is serialized from the prepared request after response format and text call options
have been applied. This avoids exposing a metadata body that differs from the actual model request.

### 4. Keep unsupported output strategies explicit

Upstream `generateObject` also supports:

- `output: "array"` by wrapping array elements under an `elements` object
- `output: "enum"` by wrapping the enum string under a `result` object
- `output: "no-schema"`
- `experimental_repairText`

The non-streaming array and enum strategies are exposed as Rust-specific helper functions instead
of overloading one TypeScript-style option union:

- `generate_array(...)` wraps the element schema under `{ "elements": [...] }`, sends that schema
  to the provider, and returns the extracted `Vec<T>`
- `generate_enum(...)` wraps the allowed values under `{ "result": "..." }`, sends that schema to
  the provider, validates that the result is one of the allowed strings, and returns the extracted
  `String`

Repair callbacks are supported through `GenerateObjectOptions::with_repair_text_fn(...)`. The
callback receives the raw model text plus the parse or validation error and can return repaired
text for one retry.

`no-schema` remains deferred. It needs a first-class `ResponseFormat` representation for "JSON
output without JSON Schema"; using a permissive schema would be a different contract.

## Validation

This workstream is locked by:

- `cargo nextest run -p siumai structured_output --no-default-features --features openai --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test --no-default-features --features openai,anthropic,google,google-vertex --no-fail-fast`
- `cargo fmt -p siumai --check`
