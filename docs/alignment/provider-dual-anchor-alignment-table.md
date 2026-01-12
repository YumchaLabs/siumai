# Provider Dual-Anchor Alignment Table (Official API + Vercel AI SDK)

This document is a **single-page, actionable index** for the Alpha.5 fearless refactor.

We use two anchors for every alignment decision:

1. **Official API docs**: the source of truth for wire correctness (HTTP endpoints, headers, JSON/SSE shapes).
2. **Vercel AI SDK (`repo-ref/ai`)**: the source of truth for semantics + developer experience (tool ids/names, defaults,
   warnings, and fixture behavior).

If both anchors cannot be satisfied simultaneously, we prefer **official correctness by default**, and add a documented
switch for Vercel semantics where it matters for fixtures/gateway compatibility.

## How to use this table

- Pick a provider + module row.
- Open the linked **official docs** and the **Vercel ref** files.
- Verify Siumai’s implementation + tests/fixtures match the anchors.
- If there is a gap: add a fixture/test first, then adjust mapping code, then mark the row as “Green”.

Legend:

- **Green**: covered by fixtures/tests and considered stable.
- **Yellow**: implemented but missing fixtures or has known lossy edges.
- **Red**: not implemented or knowingly diverges.

## OpenAI (`openai`) - Chat Completions + Responses

| Module | Official docs | Vercel ref | Siumai impl | Tests / fixtures | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Chat messages mapping | https://platform.openai.com/docs/api-reference/chat | `packages/openai/src/chat/*` | `siumai-protocol-openai/src/standards/openai/transformers/*` | `siumai/tests/openai_chat_messages_fixtures_alignment_test.rs`, `siumai/tests/fixtures/openai/chat-messages/*` | Green | Multimodal messages align with Vercel shape. |
| Chat Completions SSE parse/serialize | https://platform.openai.com/docs/api-reference/chat/streaming | `packages/openai/src/chat/*` | `siumai-protocol-openai/src/standards/openai/chat_completions_sse.rs` | `siumai/tests/*stream*_alignment_test.rs`, `siumai/tests/fixtures/openai-compatible/*` | Green | Used by OpenAI-compatible family as well. |
| Responses request mapping | https://platform.openai.com/docs/api-reference/responses | `packages/openai/src/responses/convert-to-openai-responses-input.ts` | `siumai-protocol-openai/src/standards/openai/transformers/request.rs` | `siumai/tests/openai_responses_input_fixtures_alignment_test.rs`, `siumai/tests/fixtures/openai/responses/input/*` | Green | Primary “gateway-grade” OpenAI surface. |
| Responses tools mapping (hosted tools) | https://platform.openai.com/docs/guides/tools | `packages/openai/src/responses/openai-responses-prepare-tools.ts`, `packages/openai/src/tool/*` | `siumai-core/src/tools.rs`, `siumai-core/src/hosted_tools/openai.rs`, `siumai-protocol-openai/src/standards/openai/utils.rs` | `siumai/tests/openai_responses_*_fixtures_alignment_test.rs`, `siumai/tests/fixtures/openai/responses/*` | Green | Tool ids/names follow Vercel fixtures. |
| Responses SSE parse/serialize | https://platform.openai.com/docs/api-reference/responses/streaming | `packages/openai/src/responses/*` | `siumai-protocol-openai/src/standards/openai/responses_sse.rs` | `siumai/tests/openai_responses_*_stream_alignment_test.rs`, `siumai/tests/fixtures/openai/responses-stream/*` | Green | Also powers cross-protocol transcoding via v3 parts. |
| Error mapping + retry classification | https://platform.openai.com/docs/guides/error-codes | `packages/openai/src/openai-error.ts`, `packages/provider-utils/src/response-handler.ts` | `siumai-protocol-openai/src/standards/openai/errors.rs`, `siumai-core/src/retry_api.rs` | `siumai/tests/openai_http_error_fixtures_alignment_test.rs` | Green | Aligns reason-phrase fallback for empty bodies. |

## Anthropic (`anthropic`) - Messages

| Module | Official docs | Vercel ref | Siumai impl | Tests / fixtures | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Messages request mapping | https://docs.anthropic.com/en/api/messages | `packages/anthropic/src/convert-to-anthropic-messages-prompt.ts` | `siumai-protocol-anthropic/src/standards/anthropic/transformers.rs`, `siumai-protocol-anthropic/src/standards/anthropic/utils.rs` | `siumai/tests/anthropic_messages_fixtures_alignment_test.rs`, `siumai/tests/fixtures/anthropic/messages/*` | Green | Beta header rules and tool schemas are fixture-driven; includes `context_management` + agent skills container injection. |
| Messages SSE parse/serialize | https://docs.anthropic.com/en/api/messages-streaming | `packages/anthropic/src/anthropic-messages-language-model.ts` | `siumai-protocol-anthropic/src/standards/anthropic/streaming.rs` | `siumai/tests/anthropic_messages_stream_fixtures_alignment_test.rs`, `siumai/tests/fixtures/anthropic/messages-stream/*` | Green | Also used for cross-protocol transcoding; tracks `event: error` streams. |
| Versioned tools + beta headers | https://docs.anthropic.com/en/docs/build-with-claude/tool-use | `packages/anthropic/src/*` | `siumai-core/src/tools.rs` (server tool specs), `siumai-protocol-anthropic/src/*` | Anthropic fixtures + parity tests | Green | Tool ids are stable (`anthropic.<tool>_<date>`), provider-native names are unversioned; streaming defaults enable `fine-grained-tool-streaming-2025-05-14` unless disabled; `effort` enables `effort-2025-11-24`. |
| Error mapping + retry classification | https://docs.anthropic.com/en/api/errors | `packages/anthropic/src/anthropic-error.ts` | `siumai-protocol-anthropic/src/standards/anthropic/errors.rs`, `siumai-core/src/retry_api.rs` | `siumai/tests/*error*_alignment_test.rs` | Green | `overloaded_error` treated as retryable (synthetic 529). |

## Gemini (`gemini` / alias `google`) - Google Generative AI (GenerateContent)

| Module | Official docs | Vercel ref | Siumai impl | Tests / fixtures | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| GenerateContent request mapping | https://ai.google.dev/api/generate-content | `packages/google/src/convert-to-google-generative-ai-messages.ts` | `siumai-protocol-gemini/src/standards/gemini/transformers.rs`, `siumai-protocol-gemini/src/standards/gemini/convert.rs` | `siumai/tests/google_generative_ai_fixtures_alignment_test.rs`, `siumai/tests/fixtures/google/generative-ai/*` | Green | JSON Schema → OpenAPI conversion matches Vercel semantics. |
| GenerateContent SSE parse/serialize | https://ai.google.dev/api/generate-content#method:-models.streamgeneratecontent | `packages/google/src/google-generative-ai-language-model.ts` | `siumai-protocol-gemini/src/standards/gemini/streaming.rs` | `siumai/tests/google_generative_ai_stream_fixtures_alignment_test.rs`, `siumai/tests/fixtures/gemini/*` | Green | STOP/tool-calls finish mapping is fixture-locked. |
| Provider-defined tools (`google.*`) mapping | https://ai.google.dev/gemini-api/docs/function-calling | `packages/google/src/google-prepare-tools.ts`, `packages/google/src/tool/*` | `siumai-core/src/tools.rs`, `siumai-protocol-gemini/src/standards/gemini/convert.rs` | `siumai/tests/gemini_tool_warnings_parity_test.rs`, `siumai/tests/fixtures/google/generative-ai/prepare-tools/*` | Green | File Search uses AIP-160 `metadataFilter` string expression: https://google.aip.dev/160 |
| Vertex RAG store tool | https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/use-vertexai-search#generate-content-using-gemini-api | `packages/google/src/tool/vertex-rag-store.ts` | `siumai-core/src/tools.rs`, `siumai-protocol-gemini/src/standards/gemini/convert.rs` | Gemini tool fixtures + parity tests | Green | `ragCorpus` required; `topK` (if present) must be positive. |

## Google Vertex (`vertex`) - Gemini + Imagen

| Module | Official docs | Vercel ref | Siumai impl | Tests / fixtures | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Vertex Gemini chat + streaming | https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini | `packages/google-vertex/src/*` | `siumai-provider-google-vertex/src/providers/vertex/*` + reuse `siumai-protocol-gemini` | `siumai/tests/vertex_chat_fixtures_alignment_test.rs`, `siumai/tests/fixtures/vertex/chat/*` | Green | Express-mode key query param behavior is tested. |
| Vertex embeddings | https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings | `packages/google-vertex/src/*` | `siumai-provider-google-vertex/src/providers/vertex/*` | `siumai/tests/vertex_embedding_fixtures_alignment_test.rs`, `siumai/tests/fixtures/vertex/embedding/*` | Green | Batch size cap is enforced in provider. |
| Vertex Imagen generate/edit | https://cloud.google.com/vertex-ai/generative-ai/docs/image/overview | `packages/google-vertex/src/imagen/*` | `siumai-provider-google-vertex/src/standards/vertex_imagen/*` | `siumai/tests/gemini_vertex_imagen_fixtures_alignment_test.rs`, `siumai/tests/fixtures/vertex/imagen/*` | Green | Includes edit/mask/referenceImages parity + response envelope tolerance. |
| Imagen headers + envelope behavior | https://cloud.google.com/vertex-ai/generative-ai/docs/image/overview | `packages/google-vertex/src/imagen/*` | `siumai-provider-google-vertex/src/*` | `siumai/tests/google_vertex_imagen_headers_alignment_test.rs`, `siumai/tests/google_vertex_imagen_response_envelope_alignment_test.rs` | Green | Locks user-agent and header merging behavior. |

## OpenAI-compatible adapters (DeepSeek / OpenRouter / SiliconFlow / etc.)

| Module | Official docs | Vercel ref | Siumai impl | Tests / fixtures | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Adapter request mapping | Provider-specific | `packages/openai-compatible/src/*` | `siumai-provider-openai-compatible/src/*`, `siumai-protocol-openai/src/standards/openai/compat/*` | `siumai/tests/openai_compatible_*_fixtures_alignment_test.rs`, `siumai/tests/fixtures/openai-compatible/*` | Green | Presets define base URL + quirks; protocol mapping reused. |
| Vendor quirks: tool-call JSON fallback | Provider-specific | `packages/openai-compatible/src/*` | `siumai-protocol-openai/src/standards/openai/compat/*` | `siumai/tests/openai_compatible_chat_response_fixtures_alignment_test.rs` | Green | Supports issue-driven fallbacks (e.g. missing `tool_calls[]`). |

## xAI (`xai`) - Responses API + Provider Tools

| Module | Official docs | Vercel ref | Siumai impl | Tests / fixtures | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Responses request/response mapping | https://docs.x.ai/docs | `packages/xai/src/responses/*` | `siumai-protocol-openai/src/standards/openai/*` (provider id = `xai`) | `siumai/tests/xai_responses_response_fixtures_alignment_test.rs`, `siumai/tests/fixtures/xai/responses/response/*` | Green | Locks provider tool mapping (`web_search`, `x_search`) and response normalization. |
| Responses SSE parse/stream part mapping | https://docs.x.ai/docs | `packages/xai/src/responses/*` | `siumai-protocol-openai/src/standards/openai/responses_sse.rs` | `siumai/tests/xai_responses_*_stream_alignment_test.rs`, `siumai/tests/fixtures/xai/responses-stream/*` | Green | Streams tool inputs and citations in a Vercel-aligned shape. |
| HTTP error mapping (OpenAI-compatible envelope) | https://docs.x.ai/docs | `packages/xai/src/xai-error.ts` | `siumai-protocol-openai/src/standards/openai/errors.rs` + OpenAI-compatible adapter `classify_http_error` | `siumai/tests/xai_http_error_fixtures_alignment_test.rs`, `siumai/tests/fixtures/xai/errors/*` | Green | Ensures OpenAI-compatible adapters preserve `error.message`. |

## Groq (`groq`) - OpenAI-compatible Chat + Audio

| Module | Official docs | Vercel ref | Siumai impl | Tests / fixtures | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Chat request mapping (OpenAI Chat Completions) | https://console.groq.com/docs/openai | `packages/groq/src/convert-to-groq-chat-messages.ts` | `siumai-provider-groq/src/providers/groq/spec.rs` (adapter) + reuse `siumai-protocol-openai` | `siumai/tests/groq_chat_request_fixtures_alignment_test.rs`, `siumai/tests/fixtures/groq/chat-requests/*` | Green | Groq quirks: `developer` role is downgraded to `system`, `max_completion_tokens` becomes `max_tokens`, and `stream_options` is omitted. |
| Error mapping | https://console.groq.com/docs/api-reference | `packages/groq/src/groq-error.ts` | `siumai-protocol-openai/src/standards/openai/errors.rs` (OpenAI-compatible error envelope) | `siumai/tests/groq_http_error_fixtures_alignment_test.rs`, `siumai/tests/fixtures/groq/errors/*` | Green | Preserves provider `error.message` and maps status into unified `LlmError`. |

## MiniMaxi (`minimaxi`) - Anthropic-compatible Chat + OpenAI-compatible Media

| Module | Official docs | Vercel ref | Siumai impl | Tests / fixtures | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Chat request mapping (Messages) | https://platform.minimaxi.com/docs/api-reference/text-anthropic-api | N/A | `siumai-provider-minimaxi/src/providers/minimaxi/spec.rs` + reuse `siumai-protocol-anthropic` | `siumai/tests/minimaxi_chat_request_fixtures_alignment_test.rs`, `siumai/tests/fixtures/minimaxi/chat-requests/*` | Green | Default chat base URL is `https://api.minimaxi.com/anthropic`; the URL normalizes to `/v1/messages`. |
| Error mapping | https://platform.minimaxi.com/docs/api-reference/text-anthropic-api | N/A | `siumai-provider-minimaxi/src/providers/minimaxi/spec.rs` | `siumai/tests/minimaxi_http_error_fixtures_alignment_test.rs`, `siumai/tests/fixtures/minimaxi/errors/*` | Green | Classifies both Anthropic-style and OpenAI-style error envelopes for robustness. |

## Amazon Bedrock (`bedrock`) - Converse + Rerank

| Module | Official docs | Vercel ref | Siumai impl | Tests / fixtures | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Converse request mapping | https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html | `packages/amazon-bedrock/src/bedrock-chat-language-model.ts`, `packages/amazon-bedrock/src/convert-to-bedrock-chat-messages.ts` | `siumai-provider-amazon-bedrock/src/standards/bedrock/chat.rs` | `siumai/tests/bedrock_chat_request_fixtures_alignment_test.rs`, `siumai/tests/fixtures/bedrock/chat-requests/*` | Green | Maps `common_params` into `inferenceConfig` and passes `additionalModelRequestFields`. |
| Converse response + streaming mapping | https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html | `packages/amazon-bedrock/src/__fixtures__/*` | `siumai-provider-amazon-bedrock/src/standards/bedrock/chat.rs` | `siumai/tests/bedrock_chat_response_alignment_test.rs`, `siumai/tests/bedrock_chat_stream_alignment_test.rs`, `siumai/tests/fixtures/bedrock/chat/*` | Green | Fixture-driven tool-use + JSON tool behavior. |
| Rerank response mapping | https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-rerank.html | `packages/amazon-bedrock/src/reranking/*` | `siumai-provider-amazon-bedrock/src/standards/bedrock/rerank.rs` | `siumai/tests/bedrock_rerank_response_alignment_test.rs`, `siumai/tests/fixtures/bedrock/rerank/*` | Green | Minimal parity for Vercel fixture response shape. |
| HTTP error mapping | https://docs.aws.amazon.com/bedrock/latest/APIReference/welcome.html | `packages/amazon-bedrock/src/bedrock-error.ts` | `siumai-provider-amazon-bedrock/src/standards/bedrock/errors.rs` | `siumai/tests/bedrock_http_error_fixtures_alignment_test.rs`, `siumai/tests/fixtures/bedrock/errors/*` | Green | Preserves JSON `message` and maps throttling/auth errors. |

## Ollama (`ollama`) - Chat

| Module | Official docs | Vercel ref | Siumai impl | Tests / fixtures | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Chat request mapping | https://github.com/ollama/ollama/blob/main/docs/api.md | N/A | `siumai-provider-ollama/src/standards/ollama/utils.rs` | `siumai/tests/ollama_chat_request_fixtures_alignment_test.rs`, `siumai/tests/fixtures/ollama/chat-requests/*` | Green | Locks options merge semantics (`extra_params` + `think` override). |
| HTTP error mapping | https://github.com/ollama/ollama/blob/main/docs/api.md | N/A | `siumai-provider-ollama/src/providers/ollama/spec.rs` | `siumai/tests/ollama_http_error_fixtures_alignment_test.rs`, `siumai/tests/fixtures/ollama/errors/*` | Green | Preserves JSON `error` string. |

## Cohere (`cohere`) - Rerank

| Module | Official docs | Vercel ref | Siumai impl | Tests / fixtures | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Rerank request/response mapping | https://docs.cohere.com/v2/reference/rerank | `packages/cohere/src/reranking/*` | `siumai-provider-cohere/src/standards/cohere/rerank.rs` | `siumai/tests/cohere_rerank_fixtures_alignment_test.rs`, `siumai/tests/fixtures/cohere/rerank/*` | Green | Vercel-aligned body/response mapping. |
| HTTP error mapping | https://docs.cohere.com/v2/reference/rerank | N/A | `siumai-provider-cohere/src/standards/cohere/errors.rs` | `siumai/tests/cohere_http_error_fixtures_alignment_test.rs`, `siumai/tests/fixtures/cohere/errors/*` | Green | Preserves provider `message` when present. |

## TogetherAI (`togetherai`) - Rerank

| Module | Official docs | Vercel ref | Siumai impl | Tests / fixtures | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Rerank request/response mapping | https://docs.together.ai/reference/rerank-1 | `packages/togetherai/src/reranking/*` | `siumai-provider-togetherai/src/standards/togetherai/rerank.rs` | `siumai/tests/togetherai_rerank_fixtures_alignment_test.rs`, `siumai/tests/fixtures/togetherai/rerank/*` | Green | Vercel-aligned body/response mapping. |
| HTTP error mapping | https://docs.together.ai/reference/rerank-1 | N/A | `siumai-provider-togetherai/src/standards/togetherai/errors.rs` | `siumai/tests/togetherai_http_error_fixtures_alignment_test.rs`, `siumai/tests/fixtures/togetherai/errors/*` | Green | Preserves provider error envelope message. |

## Where “remaining gaps” live

- Global audit checklist: `docs/alignment/provider-implementation-alignment.md`
- Backlog derived from the checklist: `docs/alignment/provider-implementation-backlog.md`
- Fixture-driven checklist: `docs/alignment/vercel-ai-fixtures-alignment.md`
- Core trio module map: `docs/alignment/core-trio-module-alignment.md`
