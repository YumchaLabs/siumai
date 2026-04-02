# AI SDK Structural Alignment - Prompt Boundary Review

Last updated: 2026-03-30

## Scope

This note reviews one specific structural mismatch against the AI SDK V4 prompt contract:

- AI SDK prompt/message/content parts use `providerOptions` on request-side inputs
- they do not expose `providerMetadata` on prompt-side content parts
- Siumai currently uses one shared `ContentPart` family for both prompt/input and
  response/observation flows

That makes the Rust surface pragmatic, but wider than the AI SDK prompt model.

Primary references:

- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-prompt.ts`
- `siumai-spec/src/types/chat/content/part.rs`
- `siumai-spec/src/types/chat/content/tool_result.rs`
- `siumai-spec/src/types/chat/message.rs`

## What the AI SDK does

The AI SDK prompt contract is intentionally narrow:

- `LanguageModelV4Message` has `providerOptions`
- request-capable prompt parts have `providerOptions`
- prompt-side parts do not carry response-side `providerMetadata`
- response/stream metadata travels on response-side content, sources, stream parts, and typed
  provider metadata helpers

That separation keeps request controls and response observations structurally distinct.

## What Siumai does today

Siumai currently uses a shared stable content layer that can represent both:

- request-time input knobs:
  - `providerOptions`
- response-time observation fields:
  - `providerMetadata`

This applies to several `ContentPart` variants, including:

- `text`
- `file` / `image` / `audio`
- `reasoning`
- `reasoning-file`
- `custom`
- `tool-call`
- `tool-result`
- `tool-approval-request`
- `source`

That means Siumai is currently a semantic superset of the AI SDK prompt surface rather than a
strict prompt-shape clone.

## Benefits of the current superset

- One unified Rust enum can survive request, response, and bridge/gateway paths.
- Cross-protocol replay is simpler because response-side provider observations do not need a second
  content tree.
- Existing public APIs avoid a large split between prompt-only and response-only content types.

## Costs of the current superset

- The stable prompt surface is structurally wider than the AI SDK prompt contract.
- New code can accidentally read request controls from `providerMetadata`.
- Serialization code must stay disciplined about which side of the boundary it is operating on.
- The shape alone does not prevent callers from mixing input and output concerns.

## Review conclusion

The recommended short-term direction is:

1. Keep the shared stable content superset for now.
2. Treat `providerOptions` as the only canonical request-time control channel.
3. Treat `providerMetadata` as response-time observation only.
4. Keep legacy metadata-as-input reads only behind explicit compatibility shims.
5. Do not add new request features that require reading `providerMetadata`.

This keeps the Rust surface pragmatic without pretending the shape is a strict AI SDK prompt clone.

## Fearless-refactor threshold

A deeper split should be considered only if one of these becomes true:

- request converters keep needing ad hoc guards to avoid response-only fields
- public docs become too hard to explain because prompt and response semantics blur together
- new provider integrations need materially different request-only vs response-only content trees

If that threshold is crossed, the next refactor should prefer:

- a prompt-only/request-only view or adapter over the existing stable content model
- compatibility aliases/builders instead of deleting the current unified content types outright

## Practical rule for the current branch

Until a deeper split is justified, use this rule consistently:

- request path: read `providerOptions`
- response path: write `providerMetadata`
- compatibility path: only read `providerMetadata` on request conversion when a documented shim
  still exists
