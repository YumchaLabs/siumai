# Completion Family Alignment - Design

Last updated: 2026-04-03

## Context

AI SDK's OpenAI-compatible, OpenAI, and Azure OpenAI provider surfaces all expose a dedicated
`completionModel()` path and route it through `/completions`.

Siumai previously had only the chat family as a first-class public path. That left a structural
gap in two places:

- there was no stable Rust completion request/response family
- OpenAI-family completion support could only be approximated with chat-oriented compatibility
  hooks or was missing on the native provider path

That gap was large enough to show up as a Red item in the structural alignment matrix.

## Goals

1. Add a stable completion-family contract that matches the AI SDK execution story closely enough
   to become the Rust baseline.
2. Keep the family-model-first architecture introduced by the refactor instead of hiding
   completion inside generic-client special cases.
3. Make the audited OpenAI-family providers execute real `/completions` requests for both
   generate and stream flows.
4. Preserve one semantic runtime stream model across chat and completion.

## Non-goals

- Do not invent a second runtime stream protocol just for completions.
- Do not widen the stable shared request boundary with provider-specific completion knobs that can
  stay in `providerOptions`.
- Do not force every provider crate to expose completion support unless the upstream AI SDK surface
  actually has a completion family to mirror.

## Final design

### D1 - Stable spec family

`siumai-spec` now owns a dedicated completion family:

- `CompletionRequest`
- `CompletionResponse`

`CompletionRequest` keeps the prompt as `Vec<ChatMessage>` instead of collapsing it to a raw
string. This is intentional:

- it reuses the stable prompt/content/provider-options boundary that the rest of the refactor
  already established
- it lets providers apply AI SDK-compatible prompt materialization rules on the provider boundary
- it keeps unsupported constructs explicit instead of silently flattening them too early

The request shape also carries:

- `tools`
- `tool_choice`
- `common_params`
- `response_format`
- `provider_options_map`
- `http_config`

This mirrors the AI SDK completion call-option story closely while still staying Rust-first.

### D2 - Core family and capability surface

`siumai-core` now treats completion as a first-class family:

- `traits::CompletionCapability`
- `completion::CompletionModel`
- `LlmClient::as_completion_capability()`
- `ProviderCapabilities::completion`

This keeps the architecture aligned with the family-first refactor:

- providers expose capability
- family models provide the stable execution surface
- generic clients remain adapters, not the architectural center

### D3 - One runtime stream model

Completion streaming intentionally reuses:

- `ChatStream`
- `ChatStreamHandle`
- `ChatStreamEvent`

Rejected alternative:

- adding a separate `CompletionStreamEvent`

Reason:

- downstream stream consumers already understand the shared semantic stream lane
- completion streaming in AI SDK still fundamentally emits text deltas, metadata, usage, and a
  terminal envelope, all of which already fit the shared runtime contract
- a second stream family would duplicate processors, adapters, and SSE/gateway logic for little
  gain

### D4 - Registry and facade layering

The registry and facade now expose completion as a normal family path:

- `ProviderFactory::completion_model_family*`
- `ProviderRegistryHandle::completion_model(...)`
- `CompletionModelHandle`
- `siumai::completion::{complete, stream, stream_with_cancel}`

The registry keeps a dedicated completion-family cache so the family model path does not have to
fall back to ad hoc generic-client reconstruction on every call.

### D5 - OpenAI-family provider behavior

The audited OpenAI-family providers now implement completion parity on the real
`/completions` path.

Prompt materialization follows the same main rules as the AI SDK reference:

- if the first message is `system`, prepend it as a plain text prelude and drop it from the
  turn-by-turn loop
- `user` messages must materialize from text parts only
- `assistant` messages must materialize from text parts only
- assistant `tool-call` content is rejected as unsupported
- `tool` messages are rejected as unsupported
- a trailing `assistant:\n` prefix is always appended
- default stop sequences include `\nuser:`

Unsupported options are surfaced explicitly instead of being silently ignored:

- `topK`
- `tools`
- `toolChoice`
- non-text `responseFormat`

OpenAI-compatible provider-option handling follows the audited AI SDK compatibility lanes:

- deprecated `providerOptions['openai-compatible']`
- canonical `providerOptions.openaiCompatible`
- provider-owned options under the provider id

Known compat completion options such as `logitBias` are normalized onto the expected wire keys
(`logit_bias`).

Native OpenAI completion now also follows the audited AI SDK completion path directly:

- non-stream completion uses `/completions`
- streamed completion uses `/completions` SSE with `stream_options.include_usage = true`
- `providerOptions.openai` completion knobs normalize `logitBias`, `logprobs`, and `user`
- completion response metadata keeps the provider-owned `openai` namespace, including raw
  `choices[0].logprobs`

Native Azure OpenAI completion follows the same family behavior with Azure-specific transport
rules:

- non-stream completion uses `/completions?api-version=...`
- streamed completion uses `/completions` SSE on the same Azure deployment route
- completion provider options merge audited `openai` and `azure` namespaces
- completion response metadata keeps the provider-owned `azure` namespace while preserving raw
  `choices[0].logprobs`

### D6 - Capability policy for OpenAI-family providers

The OpenAI-compatible factory/config/provider surface now exposes completion capability whenever
the provider participates in the audited OpenAI-compatible language-model path, and the native
OpenAI/Azure factories now advertise the same family capability on their direct provider path.

This matches the AI SDK provider shape more closely:

- `chatModel(...)`
- `completionModel(...)`
- shared provider-level settings

## Rejected alternatives

### R1 - Keep completion as a chat-only shim

Rejected because it preserves the original structural gap:

- public API remains misleading
- registry/factory capability routing stays incomplete
- `/completions`-specific warnings, prompt materialization, and provider-option handling remain
  scattered and fragile

### R2 - Collapse completion prompts to plain `String` on the stable boundary

Rejected because it throws away:

- stable provider-options placement on messages/parts
- explicit unsupported constructs
- the ability to share prompt/content audits with chat-family structures

### R3 - Add a new completion-only stream event enum

Rejected because it would create a second runtime consumer story with very little semantic value.

## Remaining follow-up

- Re-audit the upstream provider packages to see whether any additional provider surfaces beyond
  the audited OpenAI-family/native-compat lanes should also expose public completion-family
  support.
- Add broader public-path parity fixtures for completion if we want the same fixture density as
  the chat family.
- Keep completion-specific workstream notes separate from the larger structural alignment folder so
  future provider audits can be tracked without reopening the original Red architecture item.
