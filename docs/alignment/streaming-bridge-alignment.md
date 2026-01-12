# Streaming Bridge Alignment (Alpha.5)

See also: `docs/alignment/provider-implementation-alignment.md`.

This document describes how `siumai` bridges provider-specific, Vercel-aligned stream parts into **OpenAI Responses SSE** stream parts so that users can build **gateways/proxies** by re-serializing streams.

## Why a bridge is needed

In `siumai`, providers can emit Vercel-aligned custom events via:

- `ChatStreamEvent::Custom { event_type, data }`

Examples:

- Gemini emits `gemini:tool`, `gemini:source`, `gemini:reasoning` (legacy: `gemini:tool-call`, `gemini:tool-result`)
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
- `tool-approval-request` is only representable in OpenAI Responses stream parts; when transcoding to other
  downstream wire formats (Chat Completions / Anthropic / Gemini), it follows `V3UnsupportedPartBehavior`
  (drop in strict mode, lossy text downgrade in `AsText` mode).
- V3 `raw` and `file` parts do not have a stable, first-class representation in any of the target wire formats.
  Gateways should treat them as unsupported v3 parts and apply `V3UnsupportedPartBehavior` consistently:
  drop in strict mode, lossy downgrade to text in `AsText` mode.

## Related: Multi-target SSE transcoding

For gateways that need to expose multiple downstream protocol surfaces from the same upstream backend,
`siumai-extras` provides `to_transcoded_sse_response(...)` (OpenAI Responses / OpenAI Chat Completions / Anthropic Messages / Gemini GenerateContent).

The transcoder is intentionally policy-driven:

- `TranscodeSseOptions::strict()` drops v3 parts that do not have a native representation in the target wire protocol.
- `TranscodeSseOptions::lossy_text()` downgrades some unsupported v3 parts into text deltas (best-effort).
- `bridge_openai_responses_stream_parts=false` disables the additional OpenAI Responses bridging layer and preserves the original custom parts.

## Related: Tool-loop gateway (execute tools in-process)

Some providers (notably Gemini) treat tool results as inputs to the *next request* (e.g. `functionResponse`),
instead of emitting a first-class "tool-result" stream frame that every downstream protocol can consume.

If you are building a gateway that must:

- execute tools locally, and
- keep a single downstream SSE connection open across multiple tool-call rounds

use `siumai-extras::server::tool_loop::tool_loop_chat_stream(...)`.

This helper emits Vercel-aligned v3 `tool-result` parts (`ChatStreamEvent::Custom`) between steps so that
`to_transcoded_sse_response(...)` can re-encode tool results into the selected target protocol (best-effort).

See: `siumai-extras/examples/tool-loop-gateway.rs`.
