# Fireworks Unified Provider Surface - Design

Last updated: 2026-04-07

## Historical problem

AI SDK models Fireworks as one provider surface:

- `chatModel()`
- `completionModel()`
- `embeddingModel()`
- `imageModel()`

with a split runtime behind that surface:

- chat/completion/embedding/transcription use the shared OpenAI-compatible boundary
- image generation and edit use provider-owned Fireworks workflow and `image_generation` routes

Siumai had drifted from that shape in a few important ways:

- `fireworks` had been promoted to a first-class stable provider identity, but not to a fully
  unified runtime/provider surface
- the public `Provider::fireworks()` path still inherited the generic OpenAI-compatible image
  story, which intentionally did not expose Fireworks provider-owned image routes
- registry/native metadata and provider catalog output did not own the Fireworks image capability
  story end-to-end
- the shared compat family-default table did not provide the audited Fireworks image default model
- the compat chat request path did not yet normalize the Fireworks-specific request-shaping quirks
  that AI SDK applies for `thinking.budgetTokens`, `reasoningHistory`, and Fireworks reasoning
  effort levels

## Implemented design

### Architectural rule

This workstream now treats one point as fixed by the audited AI SDK reference:

- AI SDK does expose a shared image interface through `ProviderV4.imageModel(...)`,
  `ImageModelV4`, and `generateImage(...)`
- that shared layer only standardizes invocation and result aggregation
- it does not mean Fireworks image execution should be collapsed into a generic
  OpenAI-compatible image runtime

For Fireworks specifically, upstream keeps image execution provider-owned because the runtime
contract is not OpenAI-compatible in any simple sense: it spans workflow routes, async polling, and
`image_generation` model-specific endpoints.

### 1. Canonical id stays `fireworks`

`fireworks` remains the built-in provider id on the stable/provider-catalog/public facade surface.
The main change is semantic: that id now means the AI SDK-style unified Fireworks provider rather
than only a promoted provider identity layered on top of a generic compat runtime.

### 2. Registry-level aggregation instead of a duplicate text provider crate

The main composition lives in `siumai-registry/src/registry/factories/fireworks.rs`.

That factory builds:

- a shared OpenAI-compatible client for chat/completion/embedding/transcription
- a provider-owned Fireworks image client for generation/edit

The unified client then exposes both lanes behind one `fireworks` provider id.

### 3. Text/audio families stay on the shared compat runtime

AI SDK Fireworks still fundamentally behaves like an OpenAI-compatible provider for:

- chat
- completion
- embeddings
- transcription

Siumai keeps those families on the shared compat runtime instead of creating a second Fireworks
text stack.

### 4. Image generation/edit stay provider-owned

The provider-owned image lane now mirrors the audited AI SDK Fireworks image model behavior:

- sync workflow models use `/workflows/{model}/text_to_image`
- async Kontext models use `/workflows/{model}` plus `/get_result` polling
- legacy image-generation models use `/image_generation/{model}`

Image editing is restricted to Kontext models, matching the audited AI SDK surface.

### 5. Default-model semantics follow the audited split

The unified Fireworks provider now keeps separate defaults for:

- chat: `accounts/fireworks/models/llama-v3p1-8b-instruct`
- image generation: `accounts/fireworks/models/flux-1-dev-fp8`
- image edit fallback: `accounts/fireworks/models/flux-kontext-pro`

That prevents a text default model from leaking into provider-owned image execution.

### 6. Request-shaping semantics now match the audited Fireworks package more closely

The provider-owned image client now mirrors the audited AI SDK behavior for:

- `size` vs `aspectRatio` warnings
- single-input edit behavior
- mask warnings on Kontext edits
- `providerOptions.fireworks` merge precedence
- async poll failures and `data:` URL handling

The shared compat chat path now also normalizes the audited Fireworks request-shaping quirks:

- `thinking.budgetTokens -> thinking.budget_tokens`
- `reasoningHistory -> reasoning_history`
- `reasoning_effort` levels `minimal -> low` and `xhigh -> high`

## Public/runtime consequences

- `Provider::fireworks()` and `Siumai::builder().fireworks()` now build a unified provider surface
- registry metadata and provider catalog now advertise Fireworks image capability on the canonical
  provider id
- curated Fireworks model metadata now includes image-model ids
- public/runtime tests now pin provider-owned image generation/edit routing instead of leaving
  Fireworks image support intentionally unsupported

## Validation

The implemented surface is currently locked by:

- registry contract tests for unified capabilities, sync workflow routing, default image fallback,
  async Kontext edit routing, and provider-specific override precedence
- public-path tests for unified provider capabilities, provider-owned default image routing, and
  async edit polling
- compat/spec tests for Fireworks request-shaping normalization on the chat path

## Remaining follow-up

- Keep the curated Fireworks model list aligned with the audited `repo-ref/ai` package.
- Continue the broader compat-provider parity audit for any remaining provider-owned options or
  response metadata drift.
- Revisit later whether Fireworks should ever gain a dedicated provider-owned typed crate or
  remain a registry-layer hybrid wrapper like DeepInfra.
