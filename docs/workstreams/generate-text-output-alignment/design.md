# Generate Text Output Alignment - Design

Last updated: 2026-04-24

## Problem

AI SDK now recommends `generateText` with `output` specifications for structured generation. The
reference implementation in `repo-ref/ai/packages/ai/src/generate-text/output.ts` defines five
output modes:

- `text()`
- `object({ schema, name, description })`
- `array({ element, name, description })`
- `choice({ options, name, description })`
- `json({ name, description })`

Siumai already had Rust-first text-family calls and a `structured_output` module, but it lacked the
schema-less JSON response format needed to model `json()` honestly. The previous workaround would
have been a permissive JSON Schema, which is not equivalent to AI SDK's `{ type: "json" }`
provider contract.

## Goals

- Add a first-class schema-less `ResponseFormat::json_object()` representation.
- Map schema-less JSON response format through the major provider protocol layers using upstream
  semantics.
- Add non-streaming Rust helpers for the missing `output.json()` and `output.choice()` cases.
- Keep existing `generate_object`, `generate_array`, and `generate_enum` behavior stable.
- Document streaming partial-output parity as a separate runtime problem.

## Non-goals

- Do not add a TypeScript-style generic `Output<OUTPUT, PARTIAL, ELEMENT>` trait before there is a
  streaming partial JSON parser to support it honestly.
- Do not expose `partialOutput`, `elementStream`, or `parsePartialOutput` equivalents from final
  response parsing.
- Do not emulate schema-less JSON by sending an empty or permissive JSON Schema.

## Chosen Design

### 0. Partial JSON utility

AI SDK streaming output depends on `parsePartialJson`, which first attempts strict JSON parsing and
then runs a single-pass `fixJson` scanner over incomplete JSON text. Siumai now exposes the same
foundation as:

- `structured_output::fix_partial_json(...)`
- `structured_output::parse_partial_json(...)`
- `PartialJsonParseState`
- `PartialJsonParseResult`

This is deliberately lower-level than `streamText.output`: it enables honest partial parsing
without yet claiming `partialOutputStream` / `elementStream` parity.

### 1. Schema-less response format

`ResponseFormat::json_object()` serializes to:

```json
{ "type": "json" }
```

`ResponseFormat::json_schema(schema)` continues to serialize to:

```json
{ "type": "json", "schema": { "...": "..." } }
```

Deserialization uses `schema` presence to choose the Rust variant. This preserves existing
schema-backed callers while adding the missing AI SDK response format shape.

### 2. Provider mapping

Provider mapping follows `repo-ref/ai`:

- OpenAI Chat Completions and OpenAI-compatible chat map schema-less JSON to
  `{ "type": "json_object" }`.
- OpenAI Responses maps schema-less JSON to `text.format = { "type": "json_object" }`.
- Gemini maps schema-less JSON to `responseMimeType = "application/json"` without
  `responseSchema`.
- Ollama maps schema-less JSON to `format = "json"`.
- Anthropic and Bedrock keep schema-less JSON as a best-effort or ignored hint where upstream
  cannot enforce it without a schema.

### 3. Rust helper surface

`structured_output::generate_json(...)` uses `ResponseFormat::json_object()` and parses the final
model text as `serde_json::Value`.

`structured_output::generate_choice(...)` uses the same wrapped schema strategy as AI SDK
`choice()`:

```json
{
  "type": "object",
  "properties": {
    "result": { "type": "string", "enum": ["..."] }
  },
  "required": ["result"],
  "additionalProperties": false
}
```

`generate_enum(...)` remains the stricter `generateObject` enum equivalent and continues to reject
schema labels. `generate_choice(...)` accepts labels because AI SDK `Output.choice()` accepts
`name` and `description`.

## Validation

This workstream is locked by:

- `cargo nextest run -p siumai-spec response_format --no-fail-fast`
- `cargo nextest run -p siumai structured_output --no-default-features --features openai --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test --no-default-features --features openai,anthropic,google,google-vertex --no-fail-fast`
- provider protocol unit tests for OpenAI, Gemini, and Ollama response format mapping
- `cargo nextest run -p siumai-core partial_json --no-fail-fast`
