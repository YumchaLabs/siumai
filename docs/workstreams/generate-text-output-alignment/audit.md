# Generate Text Output Alignment - Audit

Last updated: 2026-04-24

Reference files:

- `repo-ref/ai/packages/ai/src/generate-text/output.ts`
- `repo-ref/ai/packages/ai/src/generate-text/output-utils.ts`
- `repo-ref/ai/packages/ai/src/generate-text/generated-file.ts`
- `repo-ref/ai/packages/ai/src/generate-text/reasoning-output.ts`
- `repo-ref/ai/packages/ai/src/generate-text/tool-approval-request-output.ts`
- `repo-ref/ai/packages/ai/src/generate-text/tool-approval-response-output.ts`
- `repo-ref/ai/packages/ai/src/generate-text/tool-error.ts`
- `repo-ref/ai/packages/ai/src/generate-text/tool-output-denied.ts`
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

## Output Content Parts

| AI SDK output part | Siumai status | Notes |
| --- | --- | --- |
| `tool-call` | Supported as passive output shape | `ToolCall` now preserves the output-part `type: "tool-call"` discriminator plus provider metadata, title, invalid/error, dynamic, and provider-executed fields. |
| `tool-result` | Supported as passive output shape | `ToolResult` now preserves the output-part `type: "tool-result"` discriminator plus provider metadata, title, dynamic/provider-executed, and preliminary result fields. |
| `GeneratedFile` | Supported as passive output shape | `GeneratedFile` stores stable `base64` plus `mediaType` and exposes Rust byte decoding through `uint8_array()`. |
| `reasoning` | Supported as passive output shape | `ReasoningOutput` carries `type: "reasoning"`, text, and provider metadata. |
| `reasoning-file` | Supported as passive output shape | `ReasoningFileOutput` carries `type: "reasoning-file"`, a nested `GeneratedFile`, and provider metadata. |
| `tool-error` | Supported as passive output shape | `ToolError` carries the full AI SDK output-side error part shape with input, error payload, provider metadata, dynamic flag, and title. |
| `tool-output-denied` | Supported as passive output shape | `ToolOutputDenied` plus `StaticToolOutputDenied` / `TypedToolOutputDenied` aliases model the AI SDK denial part. |
| `tool-approval-request` | Supported as passive output shape | `ToolApprovalRequestOutput` carries the full nested `toolCall` plus optional `isAutomatic`. Runtime prompt continuity still keeps ID-oriented approval parts until the tool-loop projection is refactored. |
| `tool-approval-response` | Supported as passive output shape | `ToolApprovalResponseOutput` carries the full nested `toolCall`, `approved`, optional `reason`, and `providerExecuted`. |

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
