# Anthropic Vertex Stream Compatibility Alignment - Design

Last updated: 2026-04-11

## Problem

The current `anthropic_vertex` client had a public-stream compatibility regression after the
streaming stack moved more aggressively onto stable runtime parts:

- indexed Anthropic `text_delta` and `thinking_delta` events were emitted as stable
  `ChatStreamEvent::Part(ChatStreamPart::{TextDelta,ReasoningDelta})`
- but callers still reading the legacy textual delta lane no longer saw
  `ContentDelta` / `ThinkingDelta`
- provider-level regression tests around structured output and reasoning streams therefore stopped
  observing any incremental text/reasoning content

There was a second adjacent issue on the non-streaming side:

- Anthropic `redacted_thinking` and metadata-only reasoning blocks intentionally preserve provider
  metadata
- but `ChatResponse::reasoning()` and `ChatMessage::reasoning()` were also returning those empty
  placeholder strings as if they were real reasoning text

That combination drifted away from the practical AI SDK compatibility contract:

- stable stream parts should be the canonical runtime lane
- but public compatibility helpers should not break legacy textual delta consumers
- metadata-only reasoning placeholders should not look like user-visible model thoughts

## Design

### 1. Restore legacy textual shadow deltas at the shared stream-factory boundary

The fix is intentionally applied in the shared stream factory/executor layer instead of pushing the
legacy compatibility burden back into each protocol converter.

That means:

- protocol converters can keep emitting stable `TextDelta` / `ReasoningDelta` parts for indexed
  Anthropic blocks
- public chat streams expose typed `TextDelta` / `ReasoningDelta` parts without legacy shadows
- the same rule now applies across:
  - SSE fallback paths
  - direct JSON stream factories
  - transport-backed JSON stream executors

This keeps stable parts canonical for client-side stream consumers.

### 2. Keep the replay scope intentionally narrow

This pass only replays textual shadow events:

- `TextDelta -> ContentDelta`
- `ReasoningDelta -> ThinkingDelta`

It does not broadly synthesize every possible legacy event for every stable part. That keeps the
compatibility layer narrow and avoids unnecessary duplicate tool/event traffic on the public
stream.

### 3. Filter metadata-only empty reasoning at the accessor boundary

`ChatResponse::reasoning()` and `ChatMessage::reasoning()` now filter empty/whitespace reasoning
strings.

This preserves the underlying content parts and provider metadata, but stops helper accessors from
reporting metadata-only redacted/signature placeholders as visible reasoning text.

## Validation

Locked by:

- `siumai-provider-google-vertex` reasoning/structured-output regression tests
- `siumai-core::streaming::factory` shadow-delta regression test
- `siumai-spec` reasoning accessor regression tests
- `cargo nextest run -p siumai-provider-google-vertex --no-default-features --features google-vertex`
- `cargo nextest run -p siumai --test public_surface_imports_test --features google-vertex`
