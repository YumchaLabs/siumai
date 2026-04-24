# Generate Object Structured Output Alignment - Audit

Last updated: 2026-04-24

Reference files:

- `repo-ref/ai/packages/ai/src/generate-object/generate-object.ts`
- `repo-ref/ai/packages/ai/src/generate-object/generate-object-result.ts`
- `repo-ref/ai/packages/ai/src/generate-object/stream-object-result.ts`
- `repo-ref/ai/packages/ai/src/generate-object/output-strategy.ts`
- `repo-ref/ai/packages/ai/src/generate-object/parse-and-validate-object-result.ts`

## Non-streaming `generateObject`

| AI SDK surface | Siumai status | Notes |
| --- | --- | --- |
| `generateObject(...)` | Supported as `structured_output::generate_object(...)` | Rust accepts an explicit `TextRequest` plus `GenerateObjectOptions` instead of TypeScript prompt unions. |
| `model: LanguageModel` | Supported | Rust helper requires `LanguageModel` so model metadata can backfill response metadata. |
| `schema` / `schemaName` / `schemaDescription` | Supported | `strict` is also exposed because Siumai response format already carries it. |
| `responseFormat: { type: "json", schema, name, description }` | Supported | Mapped to `ResponseFormat::json_schema(...)`. |
| `object` result | Supported | Parsed from final response text. |
| `reasoning` result | Supported | Concatenated from reasoning parts on `ChatResponse`. |
| `finishReason` result | Supported | Falls back to `FinishReason::Unknown` if the provider omitted it. |
| `usage` result | Supported | Projected through `LanguageModelUsage::from(Usage)`. |
| `warnings` result | Supported | Reuses shared `CallWarning`. |
| `request` result | Supported | Serialized from the prepared Rust request body. |
| `response` result | Partially supported | `id`, `timestamp`, and `model_id` are populated; headers/body are deferred because `ChatResponse` does not carry them. |
| `providerMetadata` result | Supported | Passed through from `ChatResponse.provider_metadata`. |
| `toJsonResponse(...)` | Deferred | Browser/HTTP `Response` is not a native stable Rust return shape here. |

## Output Strategy Matrix

| AI SDK output strategy | Siumai status | Rationale |
| --- | --- | --- |
| `object` | Supported | Directly maps to provider JSON Schema response format and Rust typed validation/deserialization. |
| `array` | Supported as `generate_array(...)` | Uses the upstream `{ elements: [...] }` wrapper schema and returns the extracted `Vec<T>`. |
| `enum` | Supported as `generate_enum(...)` | Uses the upstream `{ result: "..." }` wrapper schema and returns the extracted string after allowed-value validation. |
| `no-schema` | Deferred | Requires a separate contract because it intentionally has no provider-facing schema. |
| `experimental_repairText` | Deferred | Needs an explicit Rust repair callback context and error type before exposing a stable helper. |

## Streaming `streamObject`

`streamObject` is intentionally not exposed in this slice.

The upstream result has several live streams:

- `partialObjectStream`
- `elementStream`
- `textStream`
- `fullStream`

Current Siumai structured-output helpers parse final JSON from a full response or finished stream.
Exposing the upstream streaming names before incremental JSON object parsing exists would make the
public surface misleading. This remains a dedicated follow-up workstream.
