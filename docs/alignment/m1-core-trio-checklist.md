# M1 Checklist: Core Trio Parity (OpenAI / Anthropic / Gemini)

This checklist operationalizes **M1** from `docs/alignment/provider-feature-targets.md`:
core parity for the “big three” providers so multi-provider apps and gateways remain stable during Alpha.5.

Rule of thumb: each checkbox should be backed by at least one **fixture test**, **mock-api test**, or a
small **example**.

## 1) Unified language model (chat + streaming)

- [x] OpenAI chat + streaming request/response parity (fixtures)
  - Tests: `siumai/tests/openai_chat_messages_fixtures_alignment_test.rs`,
    `siumai/tests/openai_responses_*_fixtures_alignment_test.rs`,
    `siumai/tests/openai_responses_*_stream_alignment_test.rs`
- [x] Anthropic Messages request/response parity (fixtures)
  - Tests: `siumai/tests/anthropic_messages_fixtures_alignment_test.rs`,
    `siumai/tests/anthropic_messages_stream_fixtures_alignment_test.rs`
- [x] Gemini/Vertex GenerateContent request/response parity (fixtures)
  - Tests: `siumai/tests/google_generative_ai_fixtures_alignment_test.rs`,
    `siumai/tests/google_generative_ai_stream_fixtures_alignment_test.rs`,
    `siumai/tests/vertex_chat_fixtures_alignment_test.rs`

## 2) Tools: function tools

- [x] OpenAI Responses tool calls + deltas + tool inputs (fixtures)
  - Tests: `siumai/tests/openai_responses_tool_input_stream_alignment_test.rs`,
    `siumai/tests/openai_responses_tool_choice_fixtures_alignment_test.rs`,
    `siumai/tests/openai_responses_function_tools_strict_fixtures_alignment_test.rs`
- [x] Anthropic tool_use + tool_result mapping (fixtures)
  - Tests: `siumai/tests/anthropic_messages_stream_fixtures_alignment_test.rs`
- [x] Gemini functionCall streaming → `ToolCallDelta` (fixtures)
  - Tests: `siumai/tests/google_generative_ai_stream_fixtures_alignment_test.rs`

## 3) Tools: provider-defined tools (Vercel-aligned)

- [x] Provider-defined tool shape (`Tool::ProviderDefined`) + factories documented
  - Doc: `docs/alignment/provider-defined-tools-alignment.md`
  - Factories: `siumai::tools::*` / `siumai::hosted_tools::*`
- [x] OpenAI provider-defined tools parity (web search / file search / code interpreter / apply_patch / MCP)
  - Tests: `siumai/tests/openai_responses_web_search_*`,
    `siumai/tests/openai_responses_file_search_*`,
    `siumai/tests/openai_responses_code_interpreter_*`,
    `siumai/tests/openai_responses_apply_patch_*`,
    `siumai/tests/openai_responses_mcp_*`
- [x] Anthropic provider tools parity (web search / web fetch / tool_search / MCP / code execution)
  - Tests: `siumai/tests/anthropic_messages_*_fixtures_alignment_test.rs`,
    `siumai/tests/anthropic_messages_stream_fixtures_alignment_test.rs`
- [x] Gemini provider tools parity (`google.*` tool ids) + warnings for unsupported mixes
  - Tests: `siumai/tests/google_generative_ai_fixtures_alignment_test.rs`,
    `siumai/tests/vertex_chat_fixtures_alignment_test.rs`

## 4) V3 stream parts (gateway parts)

- [x] Typed v3 parts exist and can be parsed/formatted
  - Code: `siumai-core/src/streaming/stream_part.rs`
- [x] Bridge Gemini/Anthropic custom parts → OpenAI Responses parts (gateway fidelity)
  - Doc: `docs/alignment/streaming-bridge-alignment.md`
  - Code: `siumai-core/src/streaming/bridge.rs`

## 5) Cross-protocol streaming transcoding (gateway/proxy)

- [x] Gemini → OpenAI (Responses + Chat Completions)
  - Test: `siumai/tests/transcoding_gemini_to_openai_alignment_test.rs`
- [x] Gemini → Anthropic
  - Test: `siumai/tests/transcoding_gemini_to_anthropic_alignment_test.rs`
- [x] Anthropic → OpenAI (Responses + Chat Completions)
  - Test: `siumai/tests/transcoding_anthropic_to_openai_alignment_test.rs`
- [x] Anthropic → Gemini
  - Test: `siumai/tests/transcoding_anthropic_to_gemini_alignment_test.rs`
- [x] OpenAI Responses → Anthropic
  - Test: `siumai/tests/transcoding_openai_to_anthropic_alignment_test.rs`
- [x] OpenAI Responses → Gemini
  - Test: `siumai/tests/transcoding_openai_to_gemini_alignment_test.rs`

## 6) Tool-loop gateway (execute tools in-process, keep one stream open)

- [x] Tool-loop stream helper (inject v3 tool-result between steps)
  - Code: `siumai-extras/src/server/tool_loop.rs`
- [x] Multi-protocol gateway example using tool-loop
  - Example: `siumai-extras/examples/tool-loop-gateway.rs`
- [x] Gemini gateway-only tool-result replay (`functionResponse`) parse + serialize
  - Test: `siumai/tests/gemini_function_response_gateway_roundtrip_test.rs`

## 7) Fixture safety net

- [x] Upstream fixture drift audit script
  - Script: `scripts/audit_vercel_fixtures.py`
  - Doc entry: `docs/alignment/vercel-ai-fixtures-alignment.md`

## Known limitations (accepted in M1)

- V3 parts are not always representable in every target wire protocol; gateways must choose a policy:
  - Drop (strict)
  - Lossy downgrade to text (best-effort)
  - Target-specific replay (e.g. Gemini `functionResponse`)
  - Implementation: `siumai-core/src/streaming/mod.rs` + `siumai-extras/src/server/axum.rs`
- Some “tool results” are naturally next-request inputs (Gemini); the tool-loop gateway is the recommended way to preserve semantics.

## Follow-ups (post-M1 candidates)

- [x] Define and test a consistent policy for `tool-approval-request` across all transcoding targets
  - Tests: `siumai/tests/transcoding_openai_to_anthropic_alignment_test.rs`,
    `siumai/tests/transcoding_openai_to_gemini_alignment_test.rs`,
    `siumai/tests/transcoding_openai_to_openai_chat_completions_tool_approval_policy_test.rs`
- [x] Expand v3 parts coverage for `raw` / `file` in gateway pipelines (documented behavior + tests)
  - Test: `siumai-extras/src/server/axum.rs:1`
- [x] Add a small “M1 smoke matrix” command (single script) that runs only the core-trio gateway/transcoding tests
  - Windows: `scripts/test-m1.bat`
  - Unix: `scripts/test-m1.sh`
