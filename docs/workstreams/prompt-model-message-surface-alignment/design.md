# Prompt Model Message Surface Alignment - Design

Last updated: 2026-04-21

## Context

`repo-ref/ai` exposes a prompt-layer shared contract that is intentionally narrower than the
runtime chat/message superset used across Siumai today:

- `ModelMessage`
- prompt content-part structs
- `Prompt`
- prompt standardization helpers

Siumai already had a richer stable chat surface:

- `ChatMessage`
- `ContentPart`
- provider/runtime-oriented metadata and extension fields

That richer surface is useful and should remain the main interoperable runtime model. The missing
piece was an honest shared prompt contract that downstream parity work can audit against the AI SDK
without pretending the two models are structurally identical.

## Goal

- Add public AI SDK-style shared prompt/message structs on the stable Rust facade.
- Keep `ChatMessage` / `ContentPart` as the richer superset contract.
- Make narrowing explicit through fallible conversions instead of silently dropping unsupported
  fields.

## Non-goals

- Do not alias `ModelMessage` directly onto `ChatMessage`.
- Do not silently coerce `developer` messages into `system`.
- Do not silently drop response-side `providerMetadata`.
- Do not expand the prompt contract beyond what `repo-ref/ai` actually exports.

## Chosen design

### 1. Introduce prompt-owned narrowed structs

The shared prompt layer now owns dedicated Rust structs/enums for:

- prompt content parts
- role-specific model messages
- `Prompt` / `StandardizedPrompt`
- prompt/message validation and conversion errors

This keeps the public contract auditable against the AI SDK naming and shape without collapsing the
rest of Siumai's richer runtime model.

### 2. Treat `ChatMessage` as a strict superset, not the same contract

`ChatMessage` and `ContentPart` continue to support fields that are outside the AI SDK prompt
surface, for example:

- `developer` role
- response-side `providerMetadata`
- audio and source parts
- image `detail`
- runtime/provider-oriented tool-call and tool-result extensions

When converting into `ModelMessage`, those fields are rejected explicitly with
`ModelMessageConversionError`.

### 3. Keep prompt validation separate from message conversion

`Prompt::standardize()` mirrors the upstream `standardize-prompt.ts` role:

- exactly one of `prompt` or `messages` must be set
- normalized messages must not be empty
- `prompt: string` becomes a single user message

This validation remains separate from the richer message/content narrowing rules.

### 4. Enforce prompt discriminators during deserialization

The shared prompt structs now also require exact prompt-wire discriminators at the serde boundary:

- message `role` must match the target prompt struct exactly
- part `type` must match the target part struct exactly

This prevents invalid wire payloads from deserializing into superficially well-typed Rust structs
with the wrong internal discriminator, which is the closest Rust equivalent to the upstream
`modelMessageSchema` object validation for this layer.

### 5. Convert back into the richer chat runtime losslessly where possible

The shared prompt types implement conversion back into `ChatMessage` / `ChatRequest` by rebuilding
the subset that the prompt contract actually carries. Because the prompt layer is narrower, this
direction is infallible after prompt validation succeeds.

### 6. Keep prompt-owned optional tool-approval metadata ergonomic on the Rust side

Upstream `ToolApprovalResponse` is just a plain object type, so TypeScript callers set optional
fields like `reason` and `providerExecuted` directly. On the Rust side, the stable prompt struct
now exposes matching builder helpers for those optional fields.

This is not a new wire-format feature. It is a Rust-surface ergonomics completion step that keeps
the prompt-owned approval response shape symmetric with other shared helper structs such as
`ToolCall` and `ToolResult`.

### 7. Expose provider-options builders across prompt-owned structs

The shared prompt/message structs already stored upstream-aligned `providerOptions` on text/image/
file/reasoning/custom/tool parts and on all four model-message variants, but that field was still
awkward to populate from Rust because most prompt-owned structs only exposed `new(...)`.

The stable Rust prompt surface now treats that metadata consistently with the rest of Siumai's
builder-heavy API style:

- prompt parts and model-message structs expose `provider_options_map()`
- prompt parts and model-message structs expose `provider_options_map_mut()`
- prompt parts and model-message structs expose `with_provider_options_map(...)`
- prompt parts and model-message structs expose `provider_option(...)`
- prompt parts and model-message structs expose `with_provider_option(...)`
- `ToolCallPart` also exposes `with_provider_executed(...)` so its optional AI SDK metadata is not
  stranded behind direct field mutation

This keeps the stable facade auditable against upstream field presence while remaining ergonomic
for normal Rust construction patterns.

### 8. Keep tool-result provider-options helpers consistent with the rest of the shared surface

`ToolResultOutput` and nested `ToolResultContentPart` already exposed provider-option storage and
single-provider insertion helpers, but their API names still lagged behind the wider shared-type
convention used across requests, image/video inputs, tools, and now prompt-owned message structs.

The stable Rust surface now also exposes:

- `provider_options_map()` and `provider_options_map_mut()` aliases
- `with_provider_options_map(...)`
- `provider_option(...)`

The older `provider_options()` / `provider_options_mut()` names remain valid; this slice only adds
the missing convention-aligned entry points so the shared surface is more regular and easier to
audit mechanically.

### 9. Add field-level builders for prompt-part optional metadata

Some prompt-owned content parts still had straightforward shared fields that were easy to express
in upstream object literals but awkward to populate from Rust builders:

- `ImagePart.mediaType`
- `FilePart.filename`

The stable Rust prompt surface now exposes small focused builders for those fields so callers no
longer need direct field mutation for the common cases.

### 10. Close the `ToolResultContentPart::FileUrl.mediaType` structural gap

The upstream shared `file-url` tool-result part supports an optional `mediaType` field during the
current migration period. Siumai's stable `ToolResultContentPart::FileUrl` shape had already
mirrored `url` and `providerOptions`, but was still missing that optional media type field.

This slice adds the missing field plus a focused `with_media_type(...)` builder, and locks the
shape with serde/unit/facade coverage.

## Follow-up

The next audit step is to keep comparing these prompt-owned structs against `repo-ref/ai` as more
shared AI SDK data structures are added, especially around any future prompt content additions that
might still be provider-owned or experimental upstream.
