# Generate Text Output Alignment - Audit

Last updated: 2026-04-24

Reference files:

- `repo-ref/ai/packages/ai/src/generate-text/output.ts`
- `repo-ref/ai/packages/ai/src/generate-text/output-utils.ts`
- `repo-ref/ai/packages/ai/src/util/fix-json.ts`
- `repo-ref/ai/packages/ai/src/util/parse-partial-json.ts`
- `repo-ref/ai/packages/openai/src/chat/openai-chat-language-model.ts`
- `repo-ref/ai/packages/openai/src/responses/openai-responses-language-model.ts`
- `repo-ref/ai/packages/openai-compatible/src/chat/openai-compatible-chat-language-model.ts`
- `repo-ref/ai/packages/google/src/google-language-model.ts`
- `repo-ref/ai/packages/anthropic/src/anthropic-language-model.ts`
- `repo-ref/ai/packages/amazon-bedrock/src/bedrock-chat-language-model.ts`

## Output Matrix

| AI SDK output | Siumai status | Notes |
| --- | --- | --- |
| `text()` | Supported | Existing `text::generate(...)` returns the provider text response. |
| `object(...)` | Supported | `structured_output::generate_object(...)` sends JSON Schema and validates/deserializes the result. |
| `array(...)` | Supported | `structured_output::generate_array(...)` uses the upstream `{ elements: [...] }` wrapper. |
| `choice(...)` | Supported | `structured_output::generate_choice(...)` uses the upstream `{ result: "..." }` wrapper and permits labels. |
| `json(...)` | Supported | `structured_output::generate_json(...)` sends schema-less JSON response format and parses `serde_json::Value`. |

## Streaming Partial Output

| AI SDK stream lane | Siumai status | Notes |
| --- | --- | --- |
| `partialOutputStream` for JSON | Foundation supported | `partial_json_value_stream(...)` emits parsed partial JSON values from a consumed `ChatStream`. |
| `elementStream` for arrays | Deferred | Needs typed array projection over partial JSON plus validator semantics. |
| `fullStream` with enriched partial output | Deferred | Requires a tee-able Rust result object rather than a one-shot stream transformer. |
| `output` promise | Deferred | Existing final extraction helpers cover the parse, but not the combined stream result object. |

## Provider Response Format Mapping

| Provider family | Schema-backed JSON | Schema-less JSON |
| --- | --- | --- |
| OpenAI Chat Completions | `response_format.type = "json_schema"` | `response_format.type = "json_object"` |
| OpenAI Responses | `text.format.type = "json_schema"` | `text.format.type = "json_object"` |
| OpenAI-compatible | Existing structured-output policy | Falls back to `json_object` like upstream |
| Gemini | `responseMimeType` + `responseSchema` | `responseMimeType` only |
| Ollama | `format = <schema>` | `format = "json"` |
| Anthropic | Native output config or reserved JSON tool when schema exists | Not enforced, matching upstream limitations |
| Bedrock | Native Anthropic output config or reserved JSON tool when schema exists | Not enforced, matching upstream limitations |

## Remaining Gap

AI SDK output specs also define partial parsing and element stream transforms. Siumai intentionally
does not expose the full multi-lane stream result yet because the current text family returns
provider stream events directly. The underlying partial JSON parser and one-shot
`partial_json_value_stream(...)` projection are available, so a future streaming structured-output
workstream can build real array element streams and tee-able result handles on top of them.
