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

## Follow-up

The next audit step is to keep comparing these prompt-owned structs against `repo-ref/ai` as more
shared AI SDK data structures are added, especially around any future prompt content additions that
might still be provider-owned or experimental upstream.
