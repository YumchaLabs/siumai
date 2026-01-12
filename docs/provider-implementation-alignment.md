# Provider Implementation Alignment (Vercel AI SDK + Official APIs)

This document is a **hands-on audit checklist** for aligning Siumai’s provider implementations with:

- The Vercel AI SDK reference implementation (`repo-ref/ai`)
- The providers’ official HTTP APIs (wire formats, endpoints, streaming protocols)

It intentionally focuses on **module-level parity** (what exists where, and what to compare),
and links to the **tests/fixtures** that lock down behavior.

If you want a “core trio” (OpenAI / Anthropic / Gemini) **module mapping table** that mirrors Vercel file layout,
see `docs/core-trio-module-alignment.md`.

If you want a high-level capability matrix instead, see `docs/provider-feature-alignment.md`.

## How to use this doc

1. Pick a provider section.
2. Compare Siumai vs Vercel on:
   - request body mapping
   - response parsing
   - streaming parsing/serialization
   - tool mapping and tool-loop semantics
   - error mapping and retry semantics
3. Add/adjust fixtures/tests until parity is stable.

## Cross-cutting alignment rules

- **Single source of truth for “what a provider supports”**: `siumai-registry/src/native_provider_metadata.rs`
- **Fixtures-driven parity**: `docs/vercel-ai-fixtures-alignment.md` + `scripts/audit_vercel_fixtures.py`
- **Gateway streaming bridge**: `docs/streaming-bridge-alignment.md`
- **Unsupported v3 stream parts**: `V3UnsupportedPartBehavior` (Drop vs AsText)

## Vercel package ↔ Siumai crates (quick map)

| Vercel package | Siumai crate(s) | Notes |
| --- | --- | --- |
| `@ai-sdk/provider` | `siumai-core` | Unified request/response types, stream parts, conversion helpers. |
| `@ai-sdk/provider-utils` | `siumai-core`, `siumai-registry` | HTTP/stream utilities + registry wiring. |
| `@ai-sdk/openai` | `siumai-provider-openai`, `siumai-protocol-openai` | Native OpenAI + OpenAI-like protocol family. |
| `@ai-sdk/openai-compatible` | `siumai-provider-openai-compatible`, `siumai-protocol-openai` | Vendor preset layer + protocol family mapping. |
| `@ai-sdk/azure` | `siumai-provider-azure`, `siumai-protocol-openai` | Azure routing/headers + shared OpenAI-like mapping. |
| `@ai-sdk/anthropic` | `siumai-provider-anthropic`, `siumai-protocol-anthropic` | Native Anthropic + protocol family mapping. |
| `@ai-sdk/google` | `siumai-provider-gemini`, `siumai-protocol-gemini` | Google Generative AI (GenerateContent) + mapping. |
| `@ai-sdk/google-vertex` | `siumai-provider-google-vertex` | Vertex AI (Gemini + Imagen) + auth/base URL normalization. |
| `@ai-sdk/groq` | `siumai-provider-groq` | Groq is OpenAI-compatible (chat + audio); Siumai reuses the OpenAI-like protocol family with a small adapter. |
| `@ai-sdk/xai` | `siumai-provider-xai` + `siumai-protocol-openai` | xAI Grok provider with Responses API fixtures and provider tool mapping parity. |
| N/A | `siumai-provider-minimaxi` | MiniMaxi uses Anthropic-compatible chat plus OpenAI-compatible media endpoints (TTS/images/files/video/music). |

## OpenAI (Chat Completions + Responses)

**Vercel reference**
- `repo-ref/ai/packages/openai/src/chat/*`
- `repo-ref/ai/packages/openai/src/responses/*`

**Siumai implementation**
- Protocol: `siumai-protocol-openai/src/standards/openai/*`
- Provider: `siumai-provider-openai/src/*`
- Azure reuse: `siumai-provider-azure/src/*`

**Official endpoints (derived from `ProviderSpec`)**
- Default `base_url`: `https://api.openai.com/v1`
- Chat:
  - Chat Completions: `${base_url}/chat/completions`
  - Responses: `${base_url}/responses` (when `use_responses_api` is enabled)
- Embeddings (default `ProviderSpec`): `${base_url}/embeddings`
- Images (default `ProviderSpec`): `${base_url}/images/generations`
- Audio/files (default `ProviderSpec`): `${base_url}/*` (provider-specific transformers)

**What to verify (official API + Vercel parity)**
- [x] Chat Completions request mapping (`messages[]`, `tools[]`, `tool_choice`, file/image parts)
- [x] Chat Completions SSE parsing + serialization (deltas, tool calls, finish reasons)
- [x] Responses request mapping (`input`, `tools`, hosted tools, `include[]`, `tool_choice`)
- [x] Responses SSE parsing + serialization (`response.*` events, tool parts, MCP approvals)
- [x] Provider-defined tools naming parity (`openai.web_search`, `openai.file_search`, etc.)
- [x] Error mapping parity (HTTP status → unified `LlmError` classification)

**Parity tests (fixtures + streaming)**
- `siumai/tests/openai_*_fixtures_alignment_test.rs`
- `siumai/tests/openai_responses_*_stream_alignment_test.rs`
- Gateway/transcoding:
  - `siumai/tests/transcoding_openai_to_*_alignment_test.rs`
  - `siumai/tests/transcoding_openai_to_openai_chat_completions_tool_approval_policy_test.rs`

## Anthropic (Messages)

**Vercel reference**
- `repo-ref/ai/packages/anthropic/src/*`

**Siumai implementation**
- Protocol: `siumai-protocol-anthropic/src/standards/anthropic/*`
- Provider: `siumai-provider-anthropic/src/*`

**Official API audit (this repo)**
- `docs/anthropic-official-api-alignment.md`

**Official endpoints (derived from `ProviderSpec`)**
- Default `base_url`: `https://api.anthropic.com` (Siumai appends `/v1` if missing)
- Messages:
  - If base ends with `/v1`: `${base}/messages`
  - Else: `${base}/v1/messages`
- Models:
  - If base ends with `/v1`: `${base}/models` / `${base}/models/{id}`
  - Else: `${base}/v1/models` / `${base}/v1/models/{id}`

**What to verify (official API + Vercel parity)**
- [x] Required headers: `x-api-key`, `anthropic-version`, `content-type: application/json` (+ `anthropic-beta` when needed)
- [x] Messages request mapping (`system`, `messages[].content[]`, tool blocks, citations/files)
- [x] Image blocks use base64 `source` with `media_type` (URL images are downgraded to text; tool_result image content also includes `media_type`)
- [x] Tool result JSON outputs are stringified (Anthropic `tool_result.content` is string or blocks)
- [x] SSE parsing (content_block_start/delta/stop, thinking, tool_use/tool_result)
- [x] SSE serialization (gateway/proxy): can re-emit valid Anthropic SSE frames
- [x] Streaming error frames are prefixed with `event: error` (official/Vercel wire behavior)
- [x] Streaming ingress: if an `error` event terminates the stream without `message_stop`, emit `StreamEnd` with `finish_reason=error`
- [x] Streaming serialization order: close all content blocks before `message_delta`/`message_stop`
- [x] Beta headers and feature gates parity (tool search, code execution, memory, prompt caching)
- [x] Agent skills: inject `container.skills[]`, auto-enable `code-execution-2025-08-25,skills-2025-10-02,files-api-2025-04-14` betas, warn when code execution tool is missing
- [x] Streaming provider metadata: propagate `container` into finish events + `StreamEnd` provider metadata
- [x] Context management: inject `context_management` request body, auto-enable `context-management-2025-06-27`, map response `context_management` to provider metadata `contextManagement`
- [x] Fine-grained tool streaming: when streaming and `toolStreaming` not disabled, auto-enable `fine-grained-tool-streaming-2025-05-14`
- [x] Effort: inject `output_config.effort`, auto-enable `effort-2025-11-24`
- [x] Error mapping parity (`api_error`, rate limits, overloaded, etc.)

**Parity tests**
- `siumai/tests/anthropic_messages_fixtures_alignment_test.rs`
- `siumai/tests/anthropic_messages_stream_fixtures_alignment_test.rs`
- Gateway/transcoding:
  - `siumai/tests/transcoding_*_to_anthropic_alignment_test.rs`

## Gemini (Google Generative AI: GenerateContent)

**Vercel reference**
- `repo-ref/ai/packages/google/src/*`

**Siumai implementation**
- Protocol: `siumai-protocol-gemini/src/standards/gemini/*`
- Provider: `siumai-provider-gemini/src/*`
- Local OpenAPI spec (reference): `docs/gemini_OPENAPI3_0.json`

**Official endpoints (derived from `GeminiChatSpec`)**
- Default `base_url`: `https://generativelanguage.googleapis.com/v1beta`
- Chat:
  - Non-stream: `${base_url}/models/{model}:generateContent`
  - Stream: `${base_url}/models/{model}:streamGenerateContent?alt=sse`
  - `model` accepts resource-style IDs (e.g. `models/gemini-2.0-flash`) and is normalized.

**What to verify (official API + Vercel parity)**
- [x] GenerateContent request mapping (contents/parts, safety settings, generation config)
- [x] Streaming parsing (SSE `data: { candidates: ... }` frames)
- [x] Streaming serialization (gateway/proxy): emit valid GenerateContent SSE frames
- [x] Provider-defined tools mapping (`google.*` tool ids; mixed-tool warnings)
- [x] `thoughtSignature` propagation rules (providerMetadata keys and per-part metadata)
- [x] Tool result semantics:
  - Streaming `functionCall` → tool-call deltas
  - `functionResponse` is naturally next-request input; prefer tool-loop gateways
  - Finish reason mapping parity (STOP/tool-calls, content-filter reasons, MALFORMED_FUNCTION_CALL → error, fallback → other)

**Parity tests**
- `siumai/tests/google_generative_ai_fixtures_alignment_test.rs`
- `siumai/tests/google_generative_ai_stream_fixtures_alignment_test.rs`
- `siumai/tests/gemini_tool_warnings_parity_test.rs`
- Gateway/transcoding:
  - `siumai/tests/transcoding_*_to_gemini_alignment_test.rs`
  - `siumai/tests/gemini_function_response_gateway_roundtrip_test.rs`

## Google Vertex (Gemini + Imagen)

**Vercel reference**
- `repo-ref/ai/packages/google-vertex/src/*`

**Siumai implementation**
- Provider: `siumai-provider-google-vertex/src/*`
- Imagen protocol helpers live under provider crate (current phase): `siumai-provider-google-vertex/*`

**Official endpoints (derived from `GeminiChatSpec` + Vertex wrapper)**
- Base URL format (publisher path): `https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google`
- Chat:
  - Non-stream: `${base}/models/{model}:generateContent`
  - Stream: `${base}/models/{model}:streamGenerateContent?alt=sse`
- Express-mode auth: appends `?key=...` when no `Authorization` header is present.

**What to verify (official API + Vercel parity)**
- [x] Regional base URL and resource-style model IDs normalization
- [x] ADC/service-account auth behavior and headers (User-Agent, project/location routing)
- [x] Gemini via Vertex GenerateContent parity (request/response/streaming)
- [x] Imagen generate/edit/mask/referenceImages request mapping (and future fields tolerance)

**Parity tests**
- `siumai/tests/vertex_chat_fixtures_alignment_test.rs`
- `siumai/tests/vertex_embedding_fixtures_alignment_test.rs`
- `siumai/tests/vertex_imagen*_fixtures_alignment_test.rs`

## MiniMaxi (Anthropic-compatible chat + OpenAI-compatible endpoints)

**Official docs**
- Anthropic-compatible chat: https://platform.minimaxi.com/docs/api-reference/text-anthropic-api
- OpenAI-compatible chat (vendor alt): https://platform.minimaxi.com/docs/api-reference/text-openai-api
- File management: https://platform.minimaxi.com/docs/api-reference/file-management-intro

**Siumai implementation**
- Provider: `siumai-provider-minimaxi/src/providers/minimaxi/*`
- Protocols reused:
  - `siumai-protocol-anthropic` (chat/messages)
  - `siumai-protocol-openai` (image/audio + OpenAI-compatible error envelope fallback)

**What to verify (official correctness)**
- [x] Chat URL routing: default base `https://api.minimaxi.com/anthropic` -> `/anthropic/v1/messages`
- [x] Chat auth: `x-api-key` header (Anthropic-style headers)
- [x] Media auth: `Authorization: Bearer ...` (OpenAI-compatible endpoints)
- [x] Error envelope robustness: accept both Anthropic and OpenAI-compatible formats

**Parity tests**
- Fixtures:
  - `siumai/tests/fixtures/minimaxi/chat-requests/*`
  - `siumai/tests/fixtures/minimaxi/errors/*`
- Tests:
  - `siumai/tests/minimaxi_chat_request_fixtures_alignment_test.rs`
  - `siumai/tests/minimaxi_http_error_fixtures_alignment_test.rs`
- Integration-style:
  - `siumai/tests/mock_api/minimaxi_mock_api_test.rs`

## Azure OpenAI

**Vercel reference**
- `repo-ref/ai/packages/azure/src/azure-openai-provider.ts`

**Siumai implementation**
- Provider: `siumai-provider-azure/src/providers/azure_openai/spec.rs`
- Shared protocol family: `siumai-protocol-openai/*`

**Official endpoints (derived from `AzureOpenAiSpec::build_url`)**
- Base URL (resource-style): `https://{resource}.openai.azure.com/openai`
- Two routing modes:
  - Deployment-based (recommended): `${base}/deployments/{deploymentId}{path}?api-version={apiVersion}`
  - Non-deployment: `${base}/v1{path}?api-version={apiVersion}`
- Chat:
  - Responses mode: `path=/responses`
  - Chat Completions mode: `path=/chat/completions`

**What to verify (official API + Vercel parity)**
- [x] URL building rules (deployment vs `/v1`, api-version, query-string stability)
- [x] Responses SSE parsing/serialization parity (including hosted tools and reasoning events)
- [x] Azure model-router compatibility (deployment id vs model id)
- [x] Error mapping parity (Azure error body shape vs OpenAI-compatible)

**Parity tests**
- `siumai/tests/azure_openai_provider_url_fixtures_alignment_test.rs`
- `siumai/tests/azure_openai_provider_request_fixtures_alignment_test.rs`
- `siumai/tests/azure_*_stream_alignment_test.rs`

## OpenAI-Compatible (DeepSeek / OpenRouter / etc.)

**Vercel reference**
- `repo-ref/ai/packages/openai-compatible/src/*`

**Siumai implementation**
- Presets/wiring: `siumai-provider-openai-compatible/src/*`
- Protocol family mapping: `siumai-protocol-openai/src/standards/openai/compat/*`

**What to verify (official API + Vercel parity)**
- [x] Adapter routing rules (`RequestType::Chat` / embedding / image / rerank routes)
- [x] Best-effort response normalization (reasoning extraction, tool-call JSON fallback)
- [x] ProviderOptions merge semantics (provider-id keyed JSON merge hook)
- [x] Streaming SSE parse + serialize stability across vendor quirks

**Parity tests**
- `siumai/tests/*deepseek*_alignment_test.rs`
- `siumai/tests/openai_chat_azure_model_router_stream_alignment_test.rs`
- `siumai/tests/fixtures/openai-compatible/*`

## Gateway/proxy (cross-provider streaming)

**Goal**
- Preserve semantic events across protocols by bridging through Vercel-aligned v3 stream parts.

**Key docs/code**
- Doc: `docs/streaming-bridge-alignment.md`
- V3 parts: `siumai-core/src/streaming/stream_part.rs`
- Bridge: `siumai-core/src/streaming/bridge.rs`

**What to verify**
- [ ] Cross-protocol transcoding tests cover text + tool-call + tool-result + finish + error paths
- [ ] `tool-approval-request`, `raw`, `file` v3 parts follow `V3UnsupportedPartBehavior`
- [ ] Tool-loop gateway keeps exactly one downstream stream open across tool rounds

**Tests/examples**
- Tests: `siumai/tests/transcoding_*_alignment_test.rs`
- Tool-loop: `siumai-extras/src/server/tool_loop.rs`
- Example: `siumai-extras/examples/tool-loop-gateway.rs`
