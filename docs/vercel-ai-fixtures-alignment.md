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
- [x] Vercel-aligned provider options allowlist (drops unknown keys)
- [x] Edit without mask (`EDIT_MODE_CONTROLLED_EDITING`)
- [x] Imagen 4 request parameter mapping (preview/fast/ultra family)
- [x] Negative prompt precedence (`request` > `extra_params` > `providerOptions`)
- [x] Vercel-style warning for unsupported `size` (ignored by Vertex Imagen)
- [x] Vercel-style response envelope (`timestamp`, `modelId`, response headers)
- [x] Broaden response extraction variants (`bytesBase64Encoded` vs nested `image.bytesBase64Encoded`)

### In progress

- [x] Fixture-driven alignment tests under `siumai/tests/fixtures/vertex/imagen/*`

### Next

- [ ] Add fixture coverage for future response metadata fields

## OpenAI Responses Web Search

Provider id: `openai` (Responses API)

### Done

- [x] Request body tool mapping (`Tool::ProviderDefined` -> Responses `tools[]`) for `web_search`
- [x] Omit `stream` when not streaming (Vercel-aligned wire body)
- [x] Streaming SSE converter emits `toolName: "webSearch"` when request tool name is `webSearch`
- [x] Built-in web search auto-adds `include: ["web_search_call.action.sources"]` (Vercel-aligned)

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

## OpenAI Responses Image Generation

Provider id: `openai` (Responses API)

### Done

- [x] Request body tool mapping (`Tool::ProviderDefined` -> Responses `tools[]`) for `image_generation`
- [x] Streaming SSE converter emits `toolName: "generateImage"` when request tool name is `generateImage`

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

## Azure OpenAI Responses Web Search Preview (Streaming)

Provider id: `openai` (tool id `openai.web_search_preview`, Azure Responses SSE stream)

### Done

- [x] Streaming SSE converter emits `toolName: "web_search_preview"` (Vercel snapshot-aligned)

## Anthropic Messages (Streaming)

Provider id: `anthropic`

### Next

- [ ] `anthropic-json-tool.1` (tool call + tool result)
- [ ] `anthropic-tool-no-args` (tool call without args)
- [ ] `anthropic-web-search-tool.1` (provider tool)
- [ ] `anthropic-web-fetch-tool.1` (provider tool)
- [ ] `anthropic-mcp.1` (MCP tool calling)

## XAI Responses (Streaming)

Provider id: `xai`

### Next

- [ ] `xai-web-search-tool.1` (provider tool)
