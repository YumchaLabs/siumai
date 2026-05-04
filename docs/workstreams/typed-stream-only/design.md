# Typed Stream Only Design

## Goal

Make `ChatStreamPart` the only canonical stream semantics in Siumai.

Legacy transport-style events (`ContentDelta`, `ThinkingDelta`, `ToolCallDelta`, `UsageUpdate`) are
redundant with typed parts and create ambiguous consumers. The target architecture is:

```text
provider wire event
  -> ChatStreamEvent::Part / PartWithReplay
  -> protocol serializers, gateway, high-level helpers
```

`StreamStart`, `StreamEnd`, `Error`, and `Custom` may remain as lifecycle, error, and extension
envelopes while the typed stream-part migration lands. Text, reasoning, tools, sources, usage, and
metadata should flow through typed parts.

## Why Now

- `Part` / `PartWithReplay` now covers text, reasoning, tools, sources, files, finish, response
  metadata, and replay hints.
- Keeping typed and legacy deltas side by side forces serializers and examples to carry duplicate
  suppression state.
- Public examples should teach one stream model, not a compatibility merge of two models.

## Direction

1. Add typed-only convenience methods on `ChatStreamEvent` for user-facing examples.
2. Remove compatibility helpers that intentionally consume both lanes.
3. Migrate provider parsers and `EventBuilder` away from legacy delta constructors.
4. Migrate gateway, orchestrator, structured-output, and tests to `part_ref()` / typed parts.
5. Delete legacy delta variants once the workspace has no production usage.

## Non-Goals

- Do not preserve duplicate typed + legacy event emission.
- Do not add new compatibility projection APIs unless a release migration explicitly needs them.
- Do not keep serializer-local duplicate suppression as a permanent design.
