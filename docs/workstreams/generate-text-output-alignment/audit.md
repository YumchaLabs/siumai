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
does not expose stream-transform surface yet because the current text family returns provider
stream events directly. The underlying partial JSON parser is now available through
`parse_partial_json(...)`, so a future streaming structured-output workstream can build real
partial-output and array element streams on top of it rather than reparsing only final responses.
