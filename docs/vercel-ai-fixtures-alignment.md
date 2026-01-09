# Vercel AI SDK Fixtures Alignment

This document tracks fixture-driven semantic alignment work against the Vercel AI SDK.
It is intended to be a pragmatic checklist: each item should correspond to a fixture set
plus a test that validates Siumai’s request/response mapping.

## Conventions

- Fixture root: `siumai/tests/fixtures/<provider>/<suite>/...`
- Suites are case-based: `siumai/tests/fixtures/<provider>/<suite>/<case>/...`
- Each suite should include:
  - `request.json` (unified request shape used by the test)
  - `expected_body.json` (provider wire body produced by transformers)
  - `expected_warnings.json` (optional; unified warnings emitted by provider middleware)
  - `expected_url.txt` (final URL produced by ProviderSpec)
  - `response.json` (provider wire response)
  - `expected_response.json` (unified response produced by transformers)
- Provider-defined tools follow Vercel JSON shape: `{ "type": "provider", "id", "name", "args": { ... } }`
- Tests live in `siumai/tests/*_fixtures_alignment_test.rs` and are feature-gated.

## Google Vertex Imagen (via Vertex provider)

Provider id: `vertex`

### Done

- [x] `imagen-3.x` `:predict` URL routing for `ImageGenerationRequest`
- [x] `ImageEditRequest` support (mask/inpaint) with `model` in URL
- [x] `referenceImages` pass-through (via `extra_params` and typed options)
- [x] Vercel-aligned `referenceImages` shape for editing (no `mimeType`)
- [x] Typed `edit` options parity (`mode` / `baseSteps` / `maskMode` / `maskDilation`)
- [x] Vercel-aligned provider options allowlist (drops unknown keys)
- [x] Edit without mask (`EDIT_MODE_CONTROLLED_EDITING`)
- [x] Edit without mask defaults to `EDIT_MODE_INPAINT_INSERTION`
- [x] Edit with mask still supports `EDIT_MODE_CONTROLLED_EDITING` (mask reference image is retained)
- [x] Edit without mask still accepts explicit inpaint modes (e.g. `EDIT_MODE_INPAINT_REMOVAL`)
- [x] Imagen 4 request parameter mapping (preview/fast/ultra family)
- [x] Negative prompt precedence (`request` > `extra_params` > `providerOptions`)
- [x] Vercel-style warning for unsupported `size` (ignored by Vertex Imagen)
- [x] Vercel-style response envelope (`timestamp`, `modelId`, response headers)
- [x] Broaden response extraction variants (`bytesBase64Encoded` vs nested `image.bytesBase64Encoded`)

### In progress

- [x] Fixture-driven alignment tests under `siumai/tests/fixtures/vertex/imagen/*`

### Next

- [ ] Add fixture coverage for future response metadata fields

## Google Vertex (Chat & Embedding)

Provider id: `vertex`

### Done

- [x] Express mode: API key appended as `?key=...` (no `x-goog-api-key` header)
- [x] Base URL defaults: `GOOGLE_VERTEX_API_KEY` => express base URL; `GOOGLE_VERTEX_PROJECT` + `GOOGLE_VERTEX_LOCATION` => `v1beta1` projects base URL
- [x] Enterprise mode: `Authorization: Bearer ...` disables `key` query param
- [x] Chat URL routing via `:generateContent` / `:streamGenerateContent?alt=sse`
- [x] Text embedding URL routing via `:predict`
- [x] Vercel-aligned embedding request body (`instances[*].task_type/title`, `parameters.outputDimensionality/autoTruncate`)
- [x] Provider options fallback for embeddings (`providerOptions["vertex"]` then `providerOptions["google"]`)
- [x] Fixture-driven embedding alignment tests under `siumai/tests/fixtures/vertex/embedding/*`

## Google Generative AI (Gemini)

Provider id: `gemini` (tools use `google.*` ids)

### Done

- [x] `google-code-execution.1` (provider tool -> `tools: [{ codeExecution: {} }]`; response + streaming emit `tool-call`/`tool-result`)
- [x] `google-google-search.1` (Gemini 2.x -> `tools: [{ googleSearch: {} }]`)
- [x] `google-search-retrieval.1` (Gemini 1.x -> `tools: [{ googleSearchRetrieval: {} }]`)
- [x] `google-search-retrieval-dynamic.1` (Gemini 1.5 Flash -> `dynamicRetrievalConfig`)
- [x] `google-url-context.1` (Gemini 2.x -> `tools: [{ urlContext: {} }]`)
- [x] `google-enterprise-web-search.1` (Gemini 2.x -> `tools: [{ enterpriseWebSearch: {} }]`)
- [x] `google-google-maps.1` (Gemini 2.x -> `tools: [{ googleMaps: {} }]`)
- [x] `google-vertex-rag-store.1` (Gemini 2.x -> `tools: [{ retrieval: { vertex_rag_store: ... } }]`)
- [x] `google-file-search.1` (Gemini 2.5 -> `tools: [{ fileSearch: { ... } }]`)
- [x] `google-mixed-tools.1` (mixed tools -> provider tools only; emits warnings)
- [x] `google-mixed-tools-tool-choice.1` (mixed tools + tool_choice -> provider tools only; no toolConfig)
- [x] `google-url-context-unsupported.1` (unsupported tool -> omitted from request; emits warnings)
- [x] `google-file-search-unsupported.1` (unsupported tool -> omitted from request; emits warnings)
- [x] `google-code-execution-unsupported.1` (unsupported tool -> omitted from request; emits warnings)
- [x] `google-unknown-tool.1` (unknown provider tool -> omitted from request; emits warnings)
- [x] `google-enterprise-web-search-unsupported.1` (unsupported tool -> omitted from request; emits warnings)
- [x] `google-vertex-rag-store-unsupported.1` (unsupported tool -> omitted from request; emits warnings)
- [x] `google-google-maps-unsupported.1` (unsupported tool -> omitted from request; emits warnings)
- [x] Function tools + `tool_choice` (`toolConfig`) parity via fixtures (`google-function-*.1`)
- [x] File Search availability on Gemini 3 models (`google-file-search-gemini-3.1`)
- [x] `thoughtSignature` pass-through into per-part `providerMetadata.google.thoughtSignature` (text/reasoning/tool-call)
- [x] Streaming `reasoning-start`/`reasoning-delta`/`reasoning-end` custom events carry `providerMetadata.google.thoughtSignature`
- [x] Vercel-aligned response `provider_metadata["google"]` keys (`promptFeedback`, `groundingMetadata`, `urlContextMetadata`, `safetyRatings`)
- [x] ProviderMetadata key selection parity (`google` vs `vertex`) when provider id includes `vertex`

### Next

- [x] Add tool warning parity for additional unsupported tools (see `siumai/tests/gemini_tool_warnings_parity_test.rs`)

## OpenAI-Compatible Chat Prepare Tools

Provider id: `openai_compatible` (runtime provider id varies, e.g. `deepseek`, `openrouter`)

### Done

- [x] Function tools mapping to Chat Completions `tools[]` (incl. `strict` passthrough)
- [x] `tool_choice` mapping parity (`auto`/`required`/`none`/specific tool)
- [x] Provider-defined tools are omitted from Chat Completions requests and emit warnings
- [x] Fixture-driven alignment tests under `siumai/tests/fixtures/openai-compatible/prepare-tools/*`

## OpenAI-Compatible Chat Response (Reasoning)

Provider id: `openai_compatible` (Chat Completions)

### Done

- [x] Reasoning extraction parity (`reasoning_content` > `thinking` > `reasoning`)
- [x] Compatibility: if `finish_reason` indicates tool calls but `tool_calls` is missing, parse `{ "name", "arguments" }` JSON from `message.content` into a tool-call part
- [x] SiliconFlow/SiliconCloud compatibility: also parse tool-call JSON from `message.content` even when `finish_reason` is `stop` (vendor quirk; issue #11)
- [x] Fixture-driven response alignment tests under `siumai/tests/fixtures/openai-compatible/chat-response/*`

## OpenAI-Compatible Chat Messages

Provider id: `openai_compatible` (Chat Completions)

### Done

- [x] User image file parts map to `image_url` content parts (supports `image/*` -> `image/jpeg`)
- [x] Assistant tool-call parts map to `tool_calls[]` with JSON-string arguments; assistant content is concatenated text
- [x] Tool role emits one `role: tool` message per tool result; tool approval responses are omitted
- [x] Non-image file parts in Chat Completions raise an error (Vercel parity), validated via `expected_error.txt`
- [x] `openaiCompatible` message/part metadata is hoisted into OpenAI message objects and `tool_calls[]` (Vercel parity)
- [x] Fixture-driven message alignment tests under `siumai/tests/fixtures/openai-compatible/chat-messages/*`

## OpenAI Responses Web Search

Provider id: `openai` (Responses API)

### Done

- [x] Request body tool mapping (`Tool::ProviderDefined` -> Responses `tools[]`) for `web_search`
- [x] Omit `stream` when not streaming (Vercel-aligned wire body)
- [x] Streaming SSE converter emits `toolName: "webSearch"` when request tool name is `webSearch`
- [x] Built-in web search auto-adds `include: ["web_search_call.action.sources"]` (Vercel-aligned)
- [x] Web search option mapping parity (`externalWebAccess`, `filters.allowedDomains`) via fixtures

## OpenAI Responses File Search

Provider id: `openai` (Responses API)

### Done

- [x] Request body tool mapping (`Tool::ProviderDefined` -> Responses `tools[]`) for `file_search`
- [x] `providerOptions.openai.include = ["file_search_call.results"]` injects Responses `include[]`
- [x] Streaming SSE converter emits `toolName: "fileSearch"` when request tool name is `fileSearch`

## OpenAI Responses Code Interpreter

Provider id: `openai` (Responses API)

### Done

- [x] Request body tool mapping (`Tool::ProviderDefined` -> Responses `tools[]`) for `code_interpreter`
- [x] Built-in code interpreter auto-adds `include: ["code_interpreter_call.outputs"]` (Vercel-aligned)
- [x] Streaming SSE converter emits `toolName: "codeExecution"` when request tool name is `codeExecution`
- [x] Streaming emits document sources from `container_file_citation` annotations
- [x] Code interpreter container mapping parity (string id + `fileIds[]`) via fixtures

## OpenAI Responses Image Generation

Provider id: `openai` (Responses API)

### Done

- [x] Request body tool mapping (`Tool::ProviderDefined` -> Responses `tools[]`) for `image_generation`
- [x] Streaming SSE converter emits `toolName: "generateImage"` when request tool name is `generateImage`
- [x] Image generation option mapping parity (`outputFormat`, `outputCompression`, etc.) via fixtures

## OpenAI Responses Local Shell / Shell / Apply Patch

Provider id: `openai` (Responses API)

### Done

- [x] Request body tool mapping for `local_shell` / `shell` / `apply_patch`
- [x] Streaming SSE converter emits `toolName: "shell"` for `local_shell_call` / `shell_call`
- [x] Streaming SSE converter emits `toolName: "apply_patch"` for `apply_patch_call`

## OpenAI Responses MCP Tool Calling

Provider id: `openai` (Responses API)

### Done

- [x] Request body tool mapping for `openai.mcp` (snake_case keys, default `require_approval: "never"`)
- [x] Streaming SSE converter emits `toolName: "mcp.<tool>"` for `mcp_call` + `tool-result`
- [x] Streaming SSE converter emits `tool-approval-request` for `mcp_approval_request` (Vercel `toolCallId: "id-0"` style)
- [x] Streaming SSE `response.completed` backfills missing MCP tool-call/results (fixture contains 2 `mcp_call` items)
- [x] Prompt tool approval response input mapping (`tool-approval-response` -> `mcp_approval_response` + `item_reference` by default)

## OpenAI Responses Streaming (Misc)

Provider id: `openai` (Responses API)

### Done

- [x] Finish reason for incomplete responses (`finishReason.raw` from `incomplete_details.reason`, `unified: "length"`)
- [x] Logprobs aggregation into finish `providerMetadata.*.logprobs` (normalized token/logprob + top_logprobs)
- [x] Configurable `providerMetadata` key for stream parts (`openai` vs `azure`)
- [x] Error stream parts parity (`error` + `response.failed` -> `ChatStreamEvent::Error` + `finish`)

## OpenAI Responses Input (Fixtures)

Provider id: `openai` (Responses API)

### Done

- [x] System message mode parity (`system`/`developer`/`remove`) via `providerOptions.openai.systemMessageMode`
- [x] User file part parity for images/PDFs (URL, base64, file_id via `fileIdPrefixes`) in `input[]`
- [x] User file part parity for `fileIdPrefixes` edge-cases (undefined/empty array/multiple prefixes)
- [x] User file part parity for image detail (`detail`) via provider metadata
- [x] Unsupported user file types raise errors (Vercel parity) via `expected_error.txt`
- [x] Assistant message parity (`output_text`) and function tool call parity (`function_call`) in `input[]`
- [x] Assistant parity for multiple tool-call parts (multiple `function_call` items)
- [x] Tool message output parity for hosted tools (`local_shell_call_output`, `shell_call_output`, `apply_patch_call_output`)
- [x] Reasoning input parity for `store=false` (`type: "reasoning"` summary merge by `itemId`, encrypted content mapping) via fixtures
- [x] Vercel-style warnings parity for `store=false` reasoning edge-cases (empty append + non-OpenAI reasoning parts)
- [x] Tool message parity for multipart tool outputs (`output: [{ input_text|input_image|input_file }]`) via fixtures
- [x] Provider-executed tool result parity for `store=true` (`item_reference` by toolCallId fallback)
- [x] MCP tool-approval-response parity (store=true/false, dedupe, execution-denied skip) via fixtures

## OpenAI Responses Response (Fixtures)

Provider id: `openai` (Responses API)

### Done

- [x] Basic text response parsing (message `output_text` -> `ChatResponse.content`)
- [x] Logprobs extraction into `providerMetadata.*.logprobs` (normalized token/logprob + top_logprobs)
- [x] Function tool calls parsing (`function_call` -> `tool-call` parts + inferred `tool_calls` finish reason + per-part `providerMetadata.openai.itemId`)
- [x] Hosted tool calls parsing (`local_shell_call` / `shell_call` / `apply_patch_call` -> `tool-call` parts + inferred `tool_calls` finish reason + per-part `providerMetadata.openai.itemId`)
- [x] MCP output parity (`mcp_call` -> `tool-call` + `tool-result`, `mcp_approval_request` -> `tool-call` + `tool-approval-request`, skip `mcp_list_tools`)
- [x] Provider tool calls parsing (`file_search_call` -> `tool-call` + `tool-result` parts, `toolName: "fileSearch"`, empty input)
- [x] Provider tool calls parsing (`web_search_call` -> `tool-call` + `tool-result` parts, action type normalization, `toolName: "webSearch"`, empty input)
- [x] Provider tool calls parsing (`code_interpreter_call` -> `tool-call` + `tool-result` parts, `toolName: "codeExecution"`, `{ code, containerId }` input)
- [x] Provider tool calls parsing (`image_generation_call` -> `tool-call` + `tool-result` parts, `toolName: "generateImage"`, empty input)
- [x] Provider tool calls parsing (`computer_call` -> `tool-call` + `tool-result` parts, `toolName: "computer_use"`, empty input)
- [x] Reasoning parsing (`reasoning` -> `type: "reasoning"` parts, summary blocks + `providerMetadata.openai.reasoningEncryptedContent`)
- [x] Usage details parity (cached + reasoning tokens)
- [x] Response metadata parity (`system_fingerprint`, `service_tier`) + message `itemId` surfaced via `provider_metadata.openai.itemId`

## OpenAI HTTP Errors (Fixtures)

Provider id: `openai` (OpenAI-compatible error envelope)

### Done

- [x] Error envelope mapping parity (`openai-error.1` -> typed `LlmError` with lossless `error.message`)

## Azure OpenAI Responses Response (Fixtures)

Provider id: `azure` (Vercel provider string: `azure.responses`)

### Done

- [x] Response parsing fixtures aligned with Vercel `@ai-sdk/azure` (web search preview, code interpreter, image generation, reasoning)
- [x] Fixtures validated via `AzureOpenAiSpec` transformer selection (locks provider split behavior)

## Azure OpenAI Responses Web Search Preview (Streaming)

Provider id: `openai` (tool id `openai.web_search_preview`, Azure Responses SSE stream)

### Done

- [x] Streaming SSE converter emits `toolName: "web_search_preview"` (Vercel snapshot-aligned)
- [x] Azure streaming fixtures (code_interpreter / image_generation / reasoning-encrypted-content) with `providerMetadata.azure`
- [x] Streaming alignment tests run via `AzureOpenAiSpec` (locks provider split behavior)
- [x] Streaming SSE emits `toolName: "file_search"` when request tool name is `file_search` (Vercel `@ai-sdk/azure` parity)

## Azure OpenAI Provider Request (URL + Headers)

Provider id: `azure` (Vercel package: `@ai-sdk/azure`)

### Done

- [x] URL routing parity for `/openai/v1/responses` + `api-version` query (default + custom)
- [x] Legacy deployment-based URL mode (`/openai/deployments/{deployment}/responses`)
- [x] `api-key` auth header + provider/request header merge parity
- [x] Fixture-driven request alignment tests under `siumai/tests/fixtures/azure/openai-provider/request/*`
- [x] URL routing parity for `chat`/`embeddings`/`images`/`audio` via fixtures under `siumai/tests/fixtures/azure/openai-provider/url/*`

## OpenAI Responses Reasoning (Streaming)

Provider id: `openai` (Responses API)

### Done

- [x] Streaming SSE emits `reasoning-start` / `reasoning-delta` / `reasoning-end` (incl. encrypted content) via fixtures

## OpenAI Responses Text (Streaming)

Provider id: `openai` (Responses API)

### Done

- [x] Streaming SSE emits `stream-start` / `response-metadata` / `text-start` / `text-delta` / `text-end` / `finish` via fixtures

## xAI Responses (Fixtures)

Provider id: `xai` (Responses API)

### Done

- [x] doGenerate code_interpreter tool call (xai-code-execution-tool.1)
- [x] doGenerate web_search tool call + sources (xai-web-search-tool.1)
- [x] doGenerate x_search tool call + sources (xai-x-search-tool.1)
- [x] Provider presets for Responses API response + SSE (avoid shared heuristics): `XaiResponsesResponseTransformer`, `XaiResponsesEventConverter`

## xAI Responses (Streaming)

Provider id: `xai` (Responses API)

### Done

- [x] Streaming SSE emits `web_search` tool input (xai-web-search-tool.1)
- [x] Streaming SSE emits `x_search` custom tool input mapping (xai-x-search-tool.1)
- [x] Streaming SSE emits xAI-style `reasoning-*` / `text-*` ids + `finishReason.raw: "completed"` via fixtures

## OpenAI Responses Multi-response Streams (Streaming)

Provider id: `openai` (Responses API)

### Done

- [x] Emit `response-metadata` for each `response.created`, but only one final `finish` + `StreamEnd` for the last response (e.g., `responses-stream/shell/openai-shell-tool.1`)
- [x] Emits `response-metadata` per `response.created` (multi-response streams)

## OpenAI Chat Completions (Streaming)

Provider id: `openai` (Chat Completions API)

### Done

- [x] Azure model router: initial chunk can emit empty `id`/`model` (fall back to request model); aligned via `azure-model-router.1`

## OpenAI Chat Messages (Files)

Provider id: `openai` / `azure` (Chat Completions)

### Done

- [x] PDF user file parts map to `{ type: "file", file: { file_id | filename + file_data } }` (Vercel `@ai-sdk/openai` parity)
- [x] Audio user file parts map to `{ type: "input_audio", input_audio: { data, format } }` for wav/mp3 (URL inputs are rejected; Vercel parity)
- [x] System message modes (`system`/`developer`/`remove`) in Chat Completions message conversion (Vercel parity)
- [x] User messages with a single text part map to string `content` (Vercel parity)
- [x] User messages with mixed text + image parts map to `[{ type: "text" }, { type: "image_url" }]` (Vercel parity)
- [x] Image detail via `providerMetadata.openai.imageDetail` on image file parts
- [x] Assistant tool-call parts map to `tool_calls[]`; tool results map to `role: tool` messages (Vercel parity)
- [x] Tool output type stringification parity (`text`/`error-text`/`execution-denied`/`json`/`error-json`/`content`) in `role: tool` messages (Vercel parity)
- [x] Fixture-driven message alignment tests under `siumai/tests/fixtures/openai/chat-messages/*`

### Next

- [x] Warning parity for `systemMessageMode: "remove"` (Vercel emits a warning)

## OpenAI Responses Tool Input (Streaming)

Provider id: `openai` (Responses API)

### Done

- [x] Streaming SSE emits `tool-input-start` / `tool-input-delta` / `tool-input-end` for `apply_patch`, `codeExecution`, function tools, and `webSearch` via fixtures

## OpenAI Responses Tool Choice (Fixtures)

Provider id: `openai` (Responses API)

### Done

- [x] `tool_choice` mapping parity for built-in tools (`web_search`, `code_interpreter`, `file_search`, `image_generation`, `mcp`, `apply_patch`)
- [x] `tool_choice` mapping parity for function tools (`{ type: "function", name }`)
- [x] Custom provider tool name mapping parity (resolve selected `toolName` against request tools)

## Tool Name Mapping (Core)

### Done

- [x] Vercel-style tool name mapping helper (double mapping between canonical tool types and provider custom names)
- [x] Unit tests aligned with Vercel `ToolNameMapping` behavior

## Function Tools Strict Mode

### Done

- [x] Pass through `strict` for OpenAI Chat Completions tool schema (`tool.function.strict`)
- [x] Pass through `strict` for OpenAI Responses tool schema (`tool.strict`)

## Anthropic Messages (Fixtures)

Provider id: `anthropic`

### Done

- [x] `anthropic-settings.1` (temperature/top_k/stop_sequences request mapping)
- [x] `anthropic-temperature-top_p-mutual-exclusivity.1` (topP ignored when temperature is set; warning)
- [x] `anthropic-thinking-enabled.1` (thinking config; max_tokens + budget; clears temperature/topP/topK; warnings)
- [x] `anthropic-thinking-default-budget.1` (default budget 1024; warning; max_tokens capped)
- [x] `anthropic-max-output-tokens-limited.1` (known model max token cap; warning)
- [x] `anthropic-max-output-tokens-unknown-model-not-limited.1` (unknown model not capped)

## Anthropic Messages (Streaming)

Provider id: `anthropic`

### Done

- [x] `anthropic-tools-null.1` / `anthropic-tools-empty.1` (tools omitted when null/empty)
- [x] Function tool request mapping parity (basic / toolChoice / input_examples / allowed_callers / cache_control)
- [x] Cache control breakpoint limit for tools (max 4; emits warning)
- [x] Unsupported provider-defined tool warnings (drops tool; emits warning)
- [x] `anthropic-mcp.1` (MCP servers request + streaming tool-call/result)
- [x] `anthropic-web-search-tool.1` (provider tool + streaming tool-call/result)
- [x] `anthropic-web-fetch-tool.1` (provider tool + streaming tool-call/result)
- [x] `anthropic-web-fetch-tool.2` (provider tool; no title in results)
- [x] `anthropic-web-fetch-tool.error` (provider tool error mapping, `error_code: "unavailable"`)
- [x] `anthropic-tool-search-regex.1` (server tool use + result -> `tool_search` tool events; supports `defer_loading`)
- [x] `anthropic-tool-search-bm25.1` (server tool use + result -> `tool_search` tool events; supports `defer_loading`)
- [x] `anthropic-tool-no-args` (streaming tool_use start + no-args tool call)
- [x] `anthropic-json-tool.1` (responseFormat json schema -> json tool + response mapped to JSON text)
- [x] `anthropic-json-tool.1` streaming: treat `json` tool_use as final JSON text + `stop` (AI SDK semantics)
- [x] `anthropic-json-tool.2` streaming: ignore text prefix block, stream the reserved `json` tool_use as text deltas (AI SDK semantics)
- [x] `anthropic-json-other-tool.1` (responseFormat json schema + other tool response)
- [x] `anthropic-json-output-format.1` (supported model -> `output_format` json schema)
- [x] `anthropic-memory-20250818.1` (provider tool + `context-management-2025-06-27` beta)
- [x] `anthropic-code-execution-20250825.1` (provider tool + `code-execution-2025-08-25` beta; stable `code_execution` toolName)
- [x] `anthropic-code-execution-20250825.2` (code execution result preserves `file_id` outputs)
- [x] `anthropic-code-execution-20250825.pptx-skill` (agent skills container in request body + `skills/files-api` betas + providerMetadata container.skills)
- [x] `anthropic-programmatic-tool-calling.1` (caller metadata + stable `code_execution` toolName)

### Next

- [x] Bring `providerMetadata.dynamic` parity for streaming tool parts (MCP tool-call/result)

## XAI Responses (Streaming)

Provider id: `xai`

### Done

- [x] `xai-web-search-tool.1` (provider tool; tool-input-delta from `arguments`, no tool-result when payload missing)
- [x] `xai-x-search-tool` (x_search triggers `custom_tool_call` items; `x_keyword_search` maps to `x_search`, web_search calls use empty tool-input-delta, no tool-result)

## DeepSeek Chat Completions (Fixtures)

Provider id: `deepseek` (OpenAI-compatible Chat Completions)

### Done

- [x] Streaming chunks parity for text / reasoning_content / tool_calls (`deepseek-*.chunks.txt`)
- [x] Response parsing parity for `reasoning_content` and tool call message parts (`deepseek-*.json`)

## Upstream Fixture Coverage Notes

- The Vercel repo (`repo-ref/ai`) also contains `__fixtures__` for providers that are not implemented in Siumai yet (e.g., Amazon Bedrock). To fully align those fixtures, we need to add the corresponding provider crates and protocol mappings first.
