# AI SDK Structural Alignment - Runtime Consumer Parity

Last updated: 2026-04-02

This note tracks the downstream consumers of the upgraded stable stream-part lane, not just the
provider parsers themselves.

Primary AI SDK references:

- `repo-ref/ai/packages/ai/src/generate-object/stream-object.ts`
- `repo-ref/ai/packages/ai/src/agent/tool-loop-agent.ts`
- `repo-ref/ai/packages/ai/src/generate-text/core-events.ts`

Primary Siumai anchors:

- `siumai-extras/src/highlevel/object.rs`
- `siumai-extras/src/server/tool_loop.rs`
- `siumai-extras/src/server/axum/sse.rs`
- `siumai-extras/examples/streaming-orchestrator.rs`

## Why this exists

The provider/protocol work is only half of the parity story.

Once `ChatStreamEvent::Part(ChatStreamPart)` became the stable semantic carrier, every higher-level
consumer had to answer the same question:

- does it read the stable part lane first
- does it keep legacy deltas only as compatibility input
- does it avoid double-applying tool arguments when both lanes appear in one stream

That is the same practical constraint that exists in AI SDK `streamObject`, `streamText`, and tool
loop agent flows.

## What is aligned now

### 1. `stream_object` no longer depends on legacy tool deltas only

Current Siumai behavior:

- `ChatStreamPart::ToolInputDelta` now feeds the structured object accumulator directly
- `ChatStreamPart::ToolCall` can replace the buffered input with the finalized JSON payload
- legacy `ToolCallDelta` is still accepted, but only wins when the stable part lane has not
  already claimed the accumulator

Why this matters:

- it matches the AI SDK direction where structured/high-level consumers operate on the semantic
  stream parts instead of protocol-specific delta quirks
- it prevents mixed streams from appending the same tool arguments twice

### 2. The extras tool loop now deduplicates legacy and stable tool-call accumulation

Current Siumai behavior:

- the tool loop keeps a source marker per tool call id (`LegacyDelta` vs `StablePart`)
- stable `ToolInputStart` / `ToolInputDelta` / `ToolCall` can drive the next-turn tool execution
  path without waiting for legacy shadow events
- if a mixed stream emits both lanes for the same tool call id, only one source is allowed to own
  that accumulator

Why this matters:

- it closes the last high-level consumer bug where protocol parsers could be correct but the tool
  loop would still behave as if only legacy deltas were authoritative
- it mirrors the AI SDK design pressure in `tool-loop-agent.ts`: semantic tool-call state must be
  collected once and then executed once

### 3. Axum SSE can now expose the stable semantic lane

Current Siumai behavior:

- `ChatStreamEvent::Part { part }` is forwarded as `event: part`
- `ChatStreamEvent::PartWithReplay { part, replay }` is forwarded as `event: part` with both the
  semantic part and replay payload

Why this matters:

- previously the upgraded semantic lane was largely invisible once a caller left the core crate
- this makes the AI-SDK-aligned runtime surface inspectable in gateway/debugger style adapters

## Remaining gaps

### 1. Public transport guidance is still implicit

We now emit `event: part`, but the public docs still do not clearly say whether:

- external consumers should prefer `event: part`
- or treat it as an internal/debugging-oriented adapter surface

Recommendation:

- document `event: part` as the preferred semantic export lane for Siumai-owned SSE adapters
- keep legacy `reasoning` / `custom` compatibility events only for older consumers

### 2. Some test suites still assert legacy-only deltas

This is acceptable for compatibility, but the remaining direction should be:

- stable semantic assertions first
- legacy delta assertions only where a provider intentionally keeps shadow compatibility output

### 3. The runtime naming story is still heavier than AI SDK

Even after these consumer fixes, Siumai still has:

- `ChatStreamEvent`
- `ChatStreamPart`
- `LanguageModelV3StreamPart`
- `LanguageModelV4StreamPart`

That is workable, but the preferred mental model should continue to converge on:

- `ChatStreamEvent` as runtime envelope
- `ChatStreamPart` as the main semantic payload
- `LanguageModelV4*` aliases as the public typed compatibility naming

## Next recommended cut

If we continue the fearless refactor, the next runtime-consumer-focused cut should be:

1. move more public assertions/tests from legacy `ToolCallDelta` expectations to stable
   `Part(ToolCall)` expectations
2. explicitly document the extras SSE `event: part` contract
3. continue trimming consumers that still inspect loose custom payloads before stable parts
