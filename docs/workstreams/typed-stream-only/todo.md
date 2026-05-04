# Typed Stream Only TODO

## Milestone 1 - Public Consumption

- [x] Add typed-only `ChatStreamEvent` helper methods for text and reasoning deltas.
- [x] Convert examples away from dual-lane stream helpers.
- [x] Remove `text::StreamDeltaExtractor`.
- [x] Verify examples compile with all features.

## Milestone 2 - Core Producers

- [x] Remove `StreamFactory` typed-to-legacy textual shadow expansion.
- [ ] Replace `EventBuilder::add_content_delta` with typed text part emission or remove it.
- [ ] Replace `EventBuilder::add_thinking_delta` with typed reasoning part emission or remove it.
- [ ] Replace `EventBuilder::add_tool_call_delta` with typed tool input/call parts or remove it.
- [ ] Audit `UsageUpdate` and route usage through `ChatStreamPart::Finish` or a typed usage part.

## Milestone 3 - Consumers

- [x] Update stream simulation middleware to synthesize typed text parts.
- [ ] Update `StreamProcessor` to aggregate typed parts only.
- [ ] Update structured-output stream parsing to consume typed text and typed tool input parts.
- [ ] Update gateway SSE rendering to consume typed parts only.
- [ ] Update orchestrator stream collection to consume typed parts only.

## Milestone 4 - Protocol Serializers

- [ ] Remove serializer-local duplicate text/reasoning suppression state.
- [ ] Make OpenAI Chat serializer consume typed text/tool/finish parts only.
- [ ] Make OpenAI Responses serializer consume typed parts only.
- [ ] Make Gemini serializer consume typed parts only.
- [ ] Make Anthropic serializer consume typed parts only.

## Milestone 5 - Removal

- [ ] Remove `ContentDelta`.
- [ ] Remove `ThinkingDelta`.
- [ ] Remove `ToolCallDelta`.
- [ ] Remove `UsageUpdate`.
- [ ] Remove legacy projection tests and update migration notes.
