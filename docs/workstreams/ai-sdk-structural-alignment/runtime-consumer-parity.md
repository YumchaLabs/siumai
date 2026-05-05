# AI SDK Structural Alignment - Runtime Consumer Parity

Last updated: 2026-04-13

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

### 1. `stream_object` no longer depends on legacy text/tool deltas only

Current Siumai behavior:

- stable `ChatStreamPart::TextDelta` now feeds `StreamObjectEvent::TextDelta`, partial-object
  extraction, and final fallback accumulation directly
- `ChatStreamPart::ToolInputDelta` now feeds the structured object accumulator directly
- `ChatStreamPart::ToolCall` can replace the buffered input with the finalized JSON payload
- legacy `ToolCallDelta` is still accepted, but only wins when the stable part lane has not
  already claimed the accumulator

Why this matters:

- it matches the AI SDK direction where structured/high-level consumers operate on the semantic
  stream parts instead of protocol-specific delta quirks or shadow text deltas
- it prevents mixed streams from appending the same tool arguments twice

### 2. The extras tool loop now deduplicates legacy and stable tool-call accumulation

Current Siumai behavior:

- the tool loop keeps a source marker per tool call id (`LegacyDelta` vs `StablePart`)
- stable `ToolInputStart` / `ToolInputDelta` / `ToolCall` can drive the next-turn tool execution
  path without waiting for legacy shadow events
- stable `TextDelta` parts now also feed the step text accumulator, so assistant history keeps
  semantic-only text even when providers skip legacy `ContentDelta` shadows
- locally executed tool results now also re-enter the downstream stream as stable
  `Part(ToolResult)` before the legacy `gateway:tool-result` compatibility event
- if a mixed stream emits both lanes for the same tool call id, only one source is allowed to own
  that accumulator

Why this matters:

- it closes the last high-level consumer bug where protocol parsers could be correct but the tool
  loop would still behave as if only legacy deltas were authoritative
- it mirrors the AI SDK design pressure in `tool-loop-agent.ts`: semantic tool-call state must be
  collected once and then executed once

### 3. Streamed orchestration fallback now honors stable text parts

Current Siumai behavior:

- streamed orchestrator steps now accumulate stable `ChatStreamPart::TextDelta` into `acc_text`
- if an upstream semantic stream finishes without a populated `StreamEnd.response`, the fallback
  `ChatResponse::new(MessageContent::Text(...))` path now preserves that stable text

Why this matters:

- it keeps `StepResult.text()` and `on_finish`-style completion surfaces aligned with AI SDK
  expectations even when the upstream transport is semantic-only
- it removes another hidden dependency on legacy `ContentDelta` shadow emission from provider
  parsers or gateway bridges

### 4. Axum SSE can now expose the stable semantic lane

Current Siumai behavior:

- `ChatStreamEvent::Part { part }` is forwarded as `event: part`
- `ChatStreamEvent::PartWithReplay { part, replay }` is forwarded as `event: part`
- both cases now use the same JSON envelope:
  - `{ "part": <ChatStreamPart>, "replay": <ChatStreamReplay | null> }`

Why this matters:

- previously the upgraded semantic lane was largely invisible once a caller left the core crate
- this makes the AI-SDK-aligned runtime surface inspectable in gateway/debugger style adapters
- the plain-text Axum helper now also consumes stable `TextDelta` parts directly, so semantic-only
  streams no longer lose text just because they skipped the legacy `ContentDelta` shadow lane

### 5. Shared stream wrappers now avoid duplicate fallback text on semantic-only streams

Current Siumai behavior:

- shared `StreamFactory` content-presence detection now treats stable `Part(TextDelta)` /
  `PartWithReplay(TextDelta)` as real text, not only legacy `ContentDelta`
- `SimulateStreamingMiddleware` now also treats stable text parts as existing text before deciding
  whether it should synthesize fallback legacy chunks from `StreamEnd.response`

Why this matters:

- semantic-only streams no longer pick up an extra tail copy of the final text just because a
  wrapper only tracked the legacy delta lane
- it keeps compatibility synthesis as a fallback path instead of letting it overwrite the stable
  semantic lane

### 6. Public gateway examples now treat stable text parts as first-class stream text

Current Siumai behavior:

- the main streaming examples, migration snippets, gateway transform example, and bridge/transcode
  tests now match stable `Part(TextDelta)` / `PartWithReplay(TextDelta)` alongside legacy
  `ContentDelta`
- gateway stream-hook guidance now explicitly tells callers to mutate stable text/reasoning parts,
  not only legacy shadow deltas

Why this matters:

- public examples no longer teach downstream users to write new legacy-only stream consumers
- future gateway/customization code is more likely to stay aligned with the stable runtime lane by
  default

### 7. Second-pass public streaming examples now default to stable semantic text/reasoning lanes

Current Siumai behavior:

- the remaining public streaming-oriented examples now read stable text/reasoning deltas first:
  - `siumai/examples/03-advanced-features/middleware/advanced-middleware.rs`
  - `siumai/examples/04-provider-specific/anthropic/web_search_streaming.rs`
  - `siumai/examples/04-provider-specific/minimaxi/minimaxi_basic.rs`
  - `siumai/examples/04-provider-specific/openai/responses-streaming-tools.rs`
  - `siumai/examples/04-provider-specific/openai/responses-websocket-incremental.rs`
  - `siumai/examples/05-integrations/registry/quickstart.rs`
  - `siumai/examples/06-extensibility/custom_provider_implementation.rs`
  - `siumai-extras/examples/streaming-orchestrator.rs`
- the custom-provider example now emits stable `ChatStreamEvent::Part(ChatStreamPart::TextDelta)`
  directly instead of teaching new provider authors to synthesize only legacy `ContentDelta`
- the MiniMaxi example now treats stable `ReasoningDelta` / `TextDelta` as the stream lane

Why this matters:

- the public facade is now much less likely to regress into legacy-only stream consumption patterns
  during future provider/example updates
- new custom-provider implementations now start from the stronger semantic stream contract instead
  of inheriting a compatibility-first design from the example code

## Remaining gaps

### 1. Public transport guidance is now explicit for Axum SSE

The Axum adapter contract is now:

- external consumers should prefer `event: part`
- `event: part` always carries the stable `{ part, replay }` envelope
- legacy `delta` / `tool` / `reasoning` / provider `custom` frames remain compatibility output,
  not the preferred semantic export lane

### 2. Some test suites still assert legacy-only deltas

This is acceptable for compatibility, but the remaining direction should be:

- stable semantic assertions first
- legacy delta assertions only where a provider intentionally keeps shadow compatibility output

One intentional example-level exception still exists:

- `siumai-extras/examples/gateway-loss-policy.rs` currently synthesizes a legacy-shaped cross-
  protocol stream on purpose to demonstrate lossy-bridge policy behavior; that example should only
  change if/when we decide the demonstration should also show an equivalent stable-part lane

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
2. continue trimming consumers that still inspect loose custom payloads before stable parts
