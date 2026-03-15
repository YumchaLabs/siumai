# Fearless Refactor V4 - Tools Parity

Last updated: 2026-03-07

This document defines the **Stable** (family-level) expectations for tool calling in Siumai V4.
It is inspired by Vercel AI SDK semantics, but uses Rust-first naming and types.

## Scope

This is about **tool calling parity** across providers:

- Tool definition (`tools`) and selection strategy (`tool_choice`)
- Tool-call and tool-result representation in `ChatResponse` / streaming parts
- Minimal invariants for the "tool loop" (model -> tool-call -> tool-result -> model)

Out of scope (extension-only for now):

- Provider-hosted tools (e.g. Google search / grounding) beyond best-effort mapping; see `hosted-search-surface.md` for the V4 boundary decision.
- Provider-specific approval flows unless surfaced as standardized parts
- Advanced orchestration (parallel execution, retries, sandboxing) beyond core primitives

## Canonical request semantics

Stable request fields are defined in `ChatRequest`:

- `tools: Option<Vec<Tool>>`
- `tool_choice: Option<ToolChoice>`

Canonical meanings:

- `ToolChoice::Auto`: model may call zero or more tools.
- `ToolChoice::Required`: model must call at least one tool (provider mapping may differ).
- `ToolChoice::None`: model must not call tools.
  - Providers without an explicit `"none"` mode (e.g. Anthropic) implement this by **removing tools** from the final request body.
- `ToolChoice::Tool { name }`: model must call a specific tool.

### Acceptance tests (no network)

- Conversion tests exist in protocol crates (wire format mapping):
  - OpenAI: `convert_tool_choice` emits `"auto" | "required" | "none" | {type:function,...}`.
  - Anthropic: `convert_tool_choice(None)` yields `None` (tools removed by transformer).
  - Gemini: `convert_tool_choice` emits `toolConfig.functionCallingConfig`.
- Request transformer tests must lock the behavior for `ToolChoice::None`:
  - OpenAI: `"tool_choice": "none"` is emitted when function tools are present.
  - Anthropic: `tool_choice` is absent and `tools` are removed from the body.
  - Gemini: `toolConfig.functionCallingConfig.mode == "NONE"` for function tools.
  - Ollama: `tools` are omitted from the final request body because the protocol has no explicit `tool_choice` field.

## Canonical response semantics

Stable response types represent tool activity as **content parts**:

- `ContentPart::ToolCall { tool_call_id, tool_name, arguments, provider_executed, provider_metadata }`
- `ContentPart::ToolResult { tool_call_id, tool_name, output, provider_executed, provider_metadata }`

Important invariants:

1. **Tool-call id stability**: `tool_call_id` must be stable within a single response/stream.
2. **Name stability**: `tool_name` must match the tool name from the request (or provider-defined tool name).
3. **Argument JSON**: tool-call arguments must be a JSON value (object preferred; string parsing is best-effort).
4. **Provider executed flag**:
   - `provider_executed = Some(true)` when the provider runs the tool (provider-defined tools).
   - `None` or `Some(false)` when the tool is user-executed (function tools).

### Provider mapping notes

This section captures *expected* mappings at a high level (not exhaustive):

- OpenAI Chat Completions: `choices[].message.tool_calls[]` -> `ContentPart::ToolCall`.
- OpenAI Responses API: provider tool items map into `tool-call` / `tool-result` where possible.
- Anthropic: `content[]` tool-use blocks -> `ContentPart::ToolCall`.
- Gemini: functionCall parts / streaming deltas -> `ContentPart::ToolCall`.

## Tool loop invariants (Stable helpers)

The Stable surface should make it easy to implement a loop like:

1. Send a request with `tools` + `tool_choice`.
2. If the response contains tool calls:
   - execute them (or deny execution)
   - feed back tool results as tool-role messages
3. Repeat until the model responds without tool calls or a configured limit is reached.

### Proposed acceptance criteria (next)

- A no-network contract test per core provider validates:
  - tool-call extraction from a representative raw response JSON
  - tool-result messages are serialized in the provider wire format consistently
  - streaming assembly produces the same final `tool-call` parts as non-streaming

This work is tracked in `provider-feature-matrix.md` under **Tools: tool loop invariants**.

Current status (core providers):

- OpenAI: tool-call extraction + tool-result request serialization + streaming/non-streaming equivalence tests exist.
- Anthropic: tool-use extraction + tool-result block request serialization + streaming/non-streaming equivalence tests exist.
- Gemini: functionCall extraction + functionResponse request serialization + streaming/non-streaming equivalence tests exist.
- OpenAI-compatible: tool-call extraction + streaming/non-streaming equivalence tests exist in the compat core path, and runtime-provider coverage now also locks that Stable `tool_choice` is not overridden by raw provider options. The direct `OpenAiCompatibleClient` path now also has transport-boundary capture tests for runtime xAI and DeepSeek requests.
- OpenRouter: public alignment tests and direct `OpenAiCompatibleClient` transport-boundary coverage now also lock that raw `providerOptions.openrouter.tool_choice` does not override Stable `tool_choice`, while vendor params such as `transforms` still merge. Common OpenRouter request knobs can now also go through `OpenRouterOptions` / `OpenRouterChatRequestExt` instead of raw provider-option maps.
- Perplexity: public alignment tests and direct `OpenAiCompatibleClient` transport-boundary coverage now also lock that raw `providerOptions.perplexity.tool_choice` does not override Stable `tool_choice`, while generic vendor params still merge on the shared OpenAI-compatible path. Common hosted-search knobs can now also go through `PerplexityOptions` / `PerplexityChatRequestExt` rather than raw `with_provider_option("perplexity", ..)` calls.
- xAI: runtime-provider routing now also has explicit streaming/non-streaming tool-call equivalence coverage on top of the OpenAI-compatible invariant, and Stable `tool_choice` now wins over raw `providerOptions.xai.tool_choice`; the same rule is now verified at the final direct-client transport boundary for both non-streaming and streaming requests.
- DeepSeek: runtime-provider smoke tests now also lock Stable `tool_choice` precedence over raw `providerOptions.deepseek.tool_choice`, plus tool-call response mapping and streaming/non-streaming tool-call equivalence. The same Stable `tool_choice` precedence is now also verified on the generic `OpenAiCompatibleClient` transport boundary together with DeepSeek reasoning-option normalization.
- Groq: provider-level smoke tests confirm adapter request/response normalization preserves the OpenAI-compatible tool-loop invariant, and Stable `tool_choice` now also wins over raw `providerOptions.groq.tool_choice`.
- Ollama: tool-result request serialization, non-streaming tool-call extraction, and streaming/non-streaming equivalence tests now exist; `ToolChoice::None` is implemented by omitting tools, while stricter forcing modes remain protocol-limited.
