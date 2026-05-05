# Typed Stream Only TODO

## Milestone 1 - Public Consumption

- [x] Add typed-only `ChatStreamEvent` helper methods for text and reasoning deltas.
- [x] Convert examples away from dual-lane stream helpers.
- [x] Remove `text::StreamDeltaExtractor`.
- [x] Verify examples compile with all features.

## Milestone 2 - Core Producers

- [x] Remove `StreamFactory` typed-to-legacy textual shadow expansion.
- [x] Replace `EventBuilder::add_content_delta` with typed text part emission or remove it.
- [x] Replace `EventBuilder::add_thinking_delta` with typed reasoning part emission or remove it.
- [x] Replace `EventBuilder::add_tool_call_delta` with typed tool input/call parts or remove it.
- [x] Audit `UsageUpdate` and route usage through `ChatStreamPart::Finish` or a typed usage part.
- [x] Move Gemini stream parser off legacy text/reasoning/tool/usage shadow events.
- [x] Move Anthropic stream parser off legacy text/reasoning/usage shadow events.
- [x] Move OpenAI-compatible stream parser off legacy text/reasoning/tool/usage shadow events.

## Milestone 3 - Consumers

- [x] Update stream simulation middleware to synthesize typed text parts.
- [x] Update Axum text/SSE helpers to consume typed parts.
- [x] Update `StreamProcessor` to aggregate typed parts only.
- [x] Update structured-output stream parsing to consume typed text and typed tool input parts.
- [x] Update gateway SSE rendering to consume typed parts only.
- [x] Update orchestrator stream collection to consume typed parts only.

## Milestone 4 - Protocol Serializers

- [x] Remove serializer-local duplicate text/reasoning suppression state.
- [x] Make OpenAI Chat serializer consume typed text/tool/finish parts only.
- [x] Make OpenAI Responses serializer consume typed parts only.
- [x] Make Gemini serializer consume typed parts only.
- [x] Make Anthropic serializer consume typed parts only.

## Milestone 5 - Removal

- [x] Remove `ContentDelta`.
- [x] Remove `ThinkingDelta`.
- [x] Remove `ToolCallDelta`.
- [x] Remove `UsageUpdate`.
- [x] Remove legacy projection tests and update migration notes.
