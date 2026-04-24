# Generate Object Structured Output Alignment - TODO

Last updated: 2026-04-24

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Non-streaming object helper

- [x] Audit `repo-ref/ai/packages/ai/src/generate-object/generate-object.ts`.
- [x] Add `GenerateObjectSchema<T>` for schema, lazy schema, flexible schema, and raw JSON Schema
  input.
- [x] Add `GenerateObjectOptions` with schema name, schema description, strictness, and underlying
  text-generation options.
- [x] Add `structured_output::generate_object(...)` for the non-streaming object-schema path.
- [x] Parse final JSON text from the existing unified `ChatResponse` content.
- [x] Use typed Rust schema validators when available and fall back to serde deserialization when
  only a plain JSON Schema is available.

## Track B - Result metadata parity

- [x] Add `GenerateObjectResult<T>` with object, reasoning, finish reason, usage, warnings,
  request metadata, response metadata, provider metadata, and raw response.
- [x] Reuse AI SDK-style response id fallback semantics with the shared `generate_id()` helper.
- [x] Use language-model metadata as the model-id fallback when the provider response does not
  include a model id.
- [x] Serialize request metadata from the prepared request after JSON response format and text call
  options are applied.
- [-] Do not fabricate HTTP response body or headers until provider runtimes expose them on the
  stable chat response.

## Track C - Facade and coverage

- [x] Re-export `generate_object` and its result/options/schema types from the stable facade.
- [x] Re-export the same names from `siumai::prelude::unified::*`.
- [x] Add local unit coverage for response-format injection and typed schema validation.
- [x] Add public surface compile coverage for the new types and family helper call path.

## Track D - Deferred AI SDK output strategies

- [x] Support `output: "array"` as `structured_output::generate_array(...)` with the upstream
  `{ elements: [...] }` wrapper strategy.
- [x] Support `output: "enum"` as `structured_output::generate_enum(...)` with the upstream
  `{ result: "..." }` wrapper strategy.
- [-] Add a Rust output-strategy API before supporting `output: "no-schema"`.
- [-] Add repair callback support only with an explicit Rust error/context contract.
- [-] Add `streamObject` only after the runtime owns incremental structured JSON parsing and can
  expose `partialObjectStream`, `elementStream`, `textStream`, and `fullStream` honestly.
