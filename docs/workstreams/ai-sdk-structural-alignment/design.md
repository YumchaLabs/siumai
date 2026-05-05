# AI SDK Structural Alignment - Design

Last updated: 2026-03-31

## Context

The workspace already completed the medium-granularity crate split and most provider/protocol
surfaces are now easier to audit against `repo-ref/ai`.

Update:

- D5 is now implemented on the stable surface: `Usage` carries AI SDK-style
  `inputTokens` / `outputTokens` / `raw`, provider/protocol paths use normalized helpers to
  preserve compatibility totals without reintroducing provider-specific usage patch layers, and
  the old `prompt/completion/total` counts are now compatibility accessors rather than public
  storage fields.
- The currently migrated OpenAI/OpenAI-compatible/Anthropic/Gemini replay paths now also preserve
  provider-unknown / `null` totals instead of silently rebuilding them as zero-valued legacy
  counts, and Gemini usage replay now preserves `cachedContentTokenCount` / `trafficType` while
  counting reasoning tokens inside total output usage.

That split solved ownership problems, but it did not finish the deeper semantic alignment with the
AI SDK provider contracts in `packages/provider/src/language-model/{v3,v4}` and
`packages/provider/src/shared/{v3,v4}`.

Recent work closed the highest-leverage prompt/content gaps:

- Anthropic streaming usage and metadata round-trips no longer drop the extended usage fields
  reported in issue `#17`.
- Request-side `providerOptions` now exist on messages, request-capable content parts, and
  tool-result output/content shapes.
- Stable content now includes first-class V4 `custom` and `reasoning-file` parts.
- Stable tool-result content now models the explicit AI SDK V4 subtypes instead of a coarse
  image/file union.

That changes the next refactor target:

- request-side boundary cleanup still needs final fallback removal
- provider coverage for the new stable parts is still intentionally scoped
- the runtime stream transport is still weaker than the AI SDK stream-part union even though the
  typed overlay is now V4-capable
- the runtime transport now has a semantic escape hatch in `ChatStreamEvent::Part(ChatStreamPart)`,
  but provider emitters/parsers still need to migrate onto it consistently

This workstream exists to finish that structural convergence.

## Why this workstream exists

The current repository is intentionally Vercel-aligned in spirit, but several important semantic
slots still drift:

1. some request controls still have compatibility fallback paths through response-style metadata
2. some newly added V4 prompt/content parts still have only partial provider coverage
3. some AI SDK stream-part semantics only exist as `Custom` events
4. the stable response usage surface historically reflected legacy OpenAI totals rather than the AI SDK
   input/output/raw token model

As long as those gaps stay open, fixture parity will keep requiring local exceptions and provider
code will keep mixing input control with output observation.

## Goals

1. Make the stable request/response boundary explicit
   - request-time provider knobs live in `providerOptions`
   - response-time provider observations live in `providerMetadata`

2. Make the stable content model V4-capable
   - message-level and part-level `providerOptions`
   - strict URL/document `source` support
   - complete `tool-approval-*` support
   - V4 `custom` and `reasoning-file`
   - explicit tool-result content variants:
     - `file-data`
     - `file-url`
     - `file-id`
     - `image-data`
     - `image-url`
     - `image-file-id`
     - `custom`

3. Make the stable stream model stronger than today's ad hoc `Custom` escape hatch
   - either by upgrading the typed stream-part layer to a V4 superset
   - or by introducing a new primary V4 stream-part surface with explicit compatibility shims

4. Align shared warning and usage semantics with the AI SDK
   - warnings should have the same semantic categories
   - usage should expose input/output/raw token structure

5. Keep the Rust-facing public story pragmatic
   - do not mirror TypeScript naming mechanically
   - do preserve the AI SDK semantic slots where they materially affect parity and maintenance

## Non-goals

- Do not create many tiny crates just to mirror the AI SDK folder count.
- Do not rename public types purely for naming parity when semantic parity is enough.
- Do not force every provider-specific custom event into the stable surface immediately.
- Do not block provider progress on a single giant migration PR.

## External references

Primary AI SDK references for this workstream:

- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-prompt.ts`
- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-stream-part.ts`
- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-source.ts`
- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-usage.ts`
- `repo-ref/ai/packages/provider/src/shared/v4/shared-v4-provider-options.ts`
- `repo-ref/ai/packages/provider/src/shared/v4/shared-v4-provider-metadata.ts`
- `repo-ref/ai/packages/provider/src/shared/v4/shared-v4-warning.ts`

Relevant current Siumai anchors:

- `siumai-spec/src/types/chat/content/part.rs`
- `siumai-spec/src/types/chat/content/tool_result.rs`
- `siumai-spec/src/types/chat/message.rs`
- `siumai-spec/src/types/usage.rs`
- `siumai-spec/src/types/streaming.rs`
- `siumai-core/src/streaming/stream_part.rs`
- `siumai-core/src/streaming/processor.rs`
- `siumai-core/src/standards/openai/utils.rs`
- `siumai-protocol-anthropic/src/standards/anthropic/utils/content.rs`
- `siumai-protocol-openai/src/standards/openai/transformers/request/responses.rs`
- `siumai-protocol-gemini/src/standards/gemini/convert.rs`

## Design principles

### P1 - Boundary correctness beats historical convenience

If a field is conceptually request input, it must not be modeled as response metadata just because
older code already consumes it that way.

### P2 - Stable types should be semantic supersets, not lowest-common-denominator shells

The stable model should preserve important AI SDK semantics even when a given provider cannot use
all of them.

### P3 - Escape hatches stay, but they stop being the primary story

`Custom` stream events and raw provider maps remain important, but major cross-provider concepts
must become first-class stable fields.

### P4 - Migrate in layers, not in one giant patch

The correct migration order is:

1. fix the stable types
2. add compatibility shims
3. migrate protocol serializers/parsers
4. migrate provider-owned helpers
5. tighten tests and docs

### P5 - Provider-owned typing remains provider-owned

Provider crates should still own typed request options and typed response metadata helpers.
This workstream changes the stable transport slots, not that ownership model.

## Proposed target model

### D1 - Separate input and output provider channels cleanly

Target rule:

- `providerOptions`
  - request-time only
  - allowed on request, message, content part, and tool-result content where AI SDK supports it
- `providerMetadata`
  - response-time only
  - allowed on response, stream parts, and response-side content parts

Backward-compatible migration is allowed temporarily, but new code should stop reading request
controls from response metadata fields.

### D2 - Promote the stable content model to a V4-capable superset

Target additions/adjustments:

- message-level `providerOptions`
- part-level `providerOptions`
- `ContentPart::Source` carries a strict `SourcePart::{Url, Document}` shape
- `tool-approval-request.providerMetadata`
- `tool-approval-response.reason`
- V4 `custom`
- V4 `reasoning-file`
- explicit V4 tool-result content variants plus a provider-keyed file-id helper
- tool-result output/content provider options where the AI SDK allows them

### D3 - Make provider coverage explicit instead of pretending every part is universal

The stable content model is now wide enough to carry the important V4 parts. The next rule is:

- add true provider support where the underlying API has a stable equivalent
- otherwise degrade explicitly and document that behavior

Current examples:

- Gemini: true `reasoning-file`
- OpenAI Responses: true `openai.compaction` and explicit tool-result file/image/id inputs
- Anthropic: true tool-result `tool_reference` plus explicit image/PDF/url tool-result mapping

### D4 - Make the stable stream model explicitly V4-aware

Today the repository has:

- a thin stable `ChatStreamEvent`
- a richer typed `TypedStreamPart` overlay

First-phase decision already landed:

- keep the historical `TypedStreamPart` name for compatibility
- upgrade that overlay into a V4-capable superset instead of introducing a second primary type
  immediately
- add a first-class runtime semantic channel:
  - `ChatStreamEvent::Part(ChatStreamPart)`
- use the richer overlay plus that runtime part channel as the main protocol/gateway semantic
  contract while legacy `ChatStreamEvent` transport variants remain for compatibility

Remaining target:

- keep `ChatStreamEvent` as the runtime transport abstraction
- keep `ChatStreamEvent::Part(ChatStreamPart)` as the first-class semantic carrier for major AI SDK
  stream parts
- attach protocol-only same-protocol replay details through a separate runtime replay carrier
  instead of widening `ChatStreamPart` or overloading generic `providerMetadata`
- but only when the detail is actually outside the AI SDK stable surface; if AI SDK already models
  the fidelity on stable part `providerMetadata` (for example Anthropic reasoning `signature` /
  `redactedData`), keep it on the semantic part lane and let serializers consume that directly
- keep strengthening the upgraded typed overlay as the main stable stream-part contract
- add or alias a separately named `LanguageModelV4StreamPart` only if that materially improves
  public ergonomics or migration clarity later
- keep adapters both ways and reduce the amount of major semantics that only survive as raw custom
  events
- migrate provider parsers/serializers toward emitting and consuming the runtime part channel
  directly where it improves parity

### D5 - Introduce an AI-SDK-shaped stable usage view

The current `Usage` type remains useful for backward compatibility, but the stable parity target is:

- input tokens:
  - total
  - noCache
  - cacheRead
  - cacheWrite
- output tokens:
  - total
  - text
  - reasoning
- raw

The migration can preserve the current totals-based `Usage` via adapters or a compatibility view.

Implementation note:

- this now exists as a compatibility superset on `Usage`
- legacy `prompt/completion/total` counts remain only as compatibility seeds plus serde/accessors,
  not as public storage fields
- provider/protocol code should prefer `normalized_input_tokens()`,
  `normalized_output_tokens()`, and `raw_usage_value()`
- stable callers should construct usage through `Usage::builder()`, `Usage::new()`, or
  `Usage::with_legacy_fields(...)` rather than struct literals
- new cleanup work should remove duplicated provider-local usage rebuilding rather than invent yet
  another usage shape

### D6 - Continue using fearless refactor rules

This workstream explicitly allows:

- internal breaking changes
- compatibility layers during migration
- moving responsibility between `siumai-spec`, `siumai-core`, and protocol/provider crates

It does not require preserving every historical internal shape.

## Recommended ownership by crate

### `siumai-spec`

Owns the stable semantic contracts:

- prompt/content/message/request/response shapes
- stable usage and warning shapes
- stable stream transport enums

### `siumai-core`

Owns:

- compatibility shims and adapters
- stream aggregation rules
- typed stream-part conversion helpers
- request/response transformation helpers that are provider-family-agnostic

### Protocol crates

Own:

- protocol-specific parse/serialize/state machines
- exact wire mapping from the stable contracts
- protocol-family typed metadata helpers where already established

### Provider crates

Own:

- typed provider request option builders/extensions
- typed provider metadata accessors
- provider-specific guardrails and parity tests

## Migration strategy

Phase order:

1. lock the audit and target semantics
2. finish low-risk shared type gaps
3. add message/part-level `providerOptions`
4. migrate request converters away from input-via-metadata behavior
5. add V4 content/tool-result gaps
6. tighten protocol coverage for the new stable parts
7. introduce the new usage model and bridge it to the current one
8. tighten public-path and fixture parity tests

## Success criteria

This workstream is complete when:

- request converters do not need response-style metadata fields for request-only features
- the stable prompt/content model can represent the important AI SDK V4 parts without lossy hacks
- protocol support for those stable parts is explicit, tested, and only degraded where the
  provider truly lacks an equivalent
- the stable stream model can represent the important AI SDK V4 stream parts without routing major
  semantics through `Custom`
- usage and warnings no longer need repeated provider-specific patch layers to recover obvious
  shared semantics

## Related documents

- `docs/alignment/core-trio-module-alignment.md`
- `docs/alignment/vercel-ai-fixtures-alignment.md`
- `docs/workstreams/fearless-refactor-v4/reasoning-alignment.md`
- `docs/workstreams/fearless-refactor-v4/typed-metadata-boundary-matrix.md`
