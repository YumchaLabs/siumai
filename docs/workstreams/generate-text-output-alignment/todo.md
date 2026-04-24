# Generate Text Output Alignment - TODO

Last updated: 2026-04-24

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Reference Audit

- [x] Audit `repo-ref/ai/packages/ai/src/generate-text/output.ts`.
- [x] Audit provider mapping for schema-less JSON response formats in OpenAI, OpenAI-compatible,
  Gemini, Anthropic, and Bedrock packages.
- [x] Confirm `choice()` is the generateText output name while `enum` remains the generateObject
  output strategy name.

## Track B - Core Response Format

- [x] Add `ResponseFormat::json_object()` for AI SDK `{ type: "json" }` without schema.
- [x] Preserve existing `ResponseFormat::json_schema(...)` serialization and deserialization.
- [x] Add public surface coverage for schema-less JSON response format.
- [x] Add AI SDK `parsePartialJson` / `fixJson` parity as Rust
  `fix_partial_json(...)`, `parse_partial_json(...)`, `PartialJsonParseState`, and
  `PartialJsonParseResult`.

## Track C - Provider Mapping

- [x] Map schema-less JSON to OpenAI Chat Completions `{ type: "json_object" }`.
- [x] Map schema-less JSON to OpenAI Responses `text.format = { type: "json_object" }`.
- [x] Map schema-less JSON to Gemini `responseMimeType = "application/json"` without schema.
- [x] Map schema-less JSON to Ollama `format = "json"`.
- [-] Anthropic schema-less JSON enforcement remains unsupported upstream and is not forced.
- [-] Bedrock schema-less JSON enforcement remains unsupported upstream and is not forced.

## Track D - Helper Surface

- [x] Add `structured_output::generate_json(...)`.
- [x] Add `structured_output::generate_choice(...)`.
- [x] Re-export the new helpers from the root facade and `prelude::unified::*`.
- [x] Keep `generate_enum(...)` strict for generateObject enum parity.

## Track E - Deferred Streaming Output

- [x] Land the partial JSON parser foundation needed by future streaming output transforms.
- [x] Add a narrow `partial_json_value_stream(...)` projection over existing `ChatStream`.
- [-] Do not expose the full AI SDK `StreamTextResult` multi-lane contract until tee/backpressure
  semantics are designed.
- [-] Do not add an `Output` trait that claims streaming parity before Track E exists.
