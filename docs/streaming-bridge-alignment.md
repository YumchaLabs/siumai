# Streaming Bridge Alignment (Alpha.5)

This document describes how `siumai` bridges provider-specific, Vercel-aligned stream parts into **OpenAI Responses SSE** stream parts so that users can build **gateways/proxies** by re-serializing streams.

## Why a bridge is needed

In `siumai`, providers can emit Vercel-aligned custom events via:

- `ChatStreamEvent::Custom { event_type, data }`

Examples:

- Gemini emits `gemini:tool`, `gemini:source`, `gemini:reasoning`
- Anthropic emits `anthropic:tool-call`, `anthropic:tool-result`, `anthropic:source`, `anthropic:finish`, etc.

The OpenAI Responses SSE serializer can only serialize:

- Standard events (`ContentDelta`, `ThinkingDelta`, `ToolCallDelta`, `UsageUpdate`, `StreamEnd`, `Error`, ...)
- OpenAI stream parts (`openai:*`)

So, to re-serialize a Gemini/Anthropic stream into OpenAI Responses SSE without losing tool/source/reasoning boundaries, we add a bridge:

- `siumai_core::streaming::OpenAiResponsesStreamPartsBridge`

## Current scope (best-effort)

The bridge focuses on the shared subset required for gateway usage:

- Tool calls / tool results
- Sources / citations
- Reasoning boundaries (when available as custom parts)
- Pass-through for standard deltas (`ContentDelta`, `ThinkingDelta`, `ToolCallDelta`, ...)

Unknown custom event types are passed through unchanged.

## Mapping table (to OpenAI Responses stream parts)

### Gemini

- `gemini:tool` (`type=tool-call`) -> `openai:tool-call`
- `gemini:tool` (`type=tool-result`) -> `openai:tool-result`
- `gemini:source` -> `openai:source`
- `gemini:reasoning` (`type=reasoning-start|reasoning-delta|reasoning-end`) -> `openai:reasoning-start|openai:reasoning-delta|openai:reasoning-end`

### Anthropic

- `anthropic:tool-call` -> `openai:tool-call`
- `anthropic:tool-result` -> `openai:tool-result`
- `anthropic:source` -> `openai:source`
- `anthropic:finish` -> `openai:finish`
- `anthropic:text-start|text-delta|text-end` -> `openai:text-start|openai:text-delta|openai:text-end`
- `anthropic:reasoning-start|reasoning-end` -> `openai:reasoning-start|openai:reasoning-end`

## Tool output index reuse (serializer behavior)

To keep `output_index` stable across `tool-call` and `tool-result` parts even when the caller does not provide `outputIndex`, the OpenAI Responses SSE serializer maintains a per-`toolCallId` output index map.

This enables the bridge to omit `outputIndex` and avoid index collisions with message items.

## Limitations / Notes

- Tool raw items are emitted as best-effort `custom_tool_call` items in the OpenAI Responses output stream.
- Provider-native schemas differ; the bridge preserves the original JSON payload in `rawItem.output` when possible.

