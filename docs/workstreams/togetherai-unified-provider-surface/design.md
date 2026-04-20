# TogetherAI Unified Provider Surface - Design

Last updated: 2026-04-20

## Problem

AI SDK models TogetherAI as one provider surface:

- `chatModel()`
- `completionModel()`
- `embeddingModel()`
- `imageModel()`
- `rerankingModel()`

with a split runtime behind that surface:

- chat/completion/embedding/speech/transcription use the shared OpenAI-compatible boundary
- image generation/edit use provider-owned `POST /images/generations`
- reranking uses provider-owned `POST /rerank`

Siumai had drifted in two steps:

- `together` was an OpenAI-compatible preset for
  chat/completion/embedding/image/audio-style families
- `togetherai` was a native rerank-only provider
- after the first unification pass, canonical `togetherai` still routed image generation/edit
  through the generic shared OpenAI-compatible image runtime

That drift caused four real problems:

1. the canonical provider id no longer matched AI SDK
2. the public default model for `togetherai` was rerank-led instead of chat-led
3. top-level builder/registry ergonomics did not represent the actual provider surface users
   expect from TogetherAI
4. image edit semantics still drifted from AI SDK:
   Siumai could fall back to multipart `/images/edits`, while AI SDK TogetherAI uses JSON
   `/images/generations` with `image_url` and rejects mask edits

## Goals

- Make `togetherai` the canonical AI SDK-style provider id.
- Expose one public provider surface for chat/completion/embedding/speech/transcription plus
  provider-owned image and rerank.
- Keep the existing provider-owned native rerank implementation.
- Reuse the shared OpenAI-compatible runtime only for the text/audio families instead of building a
  second native TogetherAI text runtime.
- Keep TogetherAI-specific typed image/rerank option helpers on the public Rust surface.
- Keep low-level compatibility only where it does not preserve a second public TogetherAI identity.

## Non-goals

- Do not write a brand-new native TogetherAI chat/completion implementation.
- Do not remove `provider_ext::togetherai` native typed rerank access.
- Do not preserve the historical `together` public builder alias once canonical `togetherai`
  lower-level compat entrypoints exist.
- Do not promote every AI SDK extra provider in the same pass; DeepInfra and Vertex MaaS remain
  follow-up audits.

## Chosen design

### Architectural rule

This workstream now treats one point as fixed by the audited AI SDK reference:

- AI SDK does expose a shared image interface through `ProviderV4.imageModel(...)`,
  `ImageModelV4`, and `generateImage(...)`
- that shared layer standardizes the stable image call/result boundary only
- TogetherAI image generation/edit should still stay provider-owned because the audited runtime
  semantics use provider-specific JSON `/images/generations` behavior rather than a generic compat
  image executor

So the right target is "shared interface, provider-owned runtime", not "one executor for every
provider that happens to expose `imageModel()`".

### 1. Canonical id stays `togetherai`

`togetherai` is now the canonical provider id for the public surface and registry metadata.

The old public compat alias `together` is retired from builder/discovery surfaces.
Lower-level compatibility can still accept legacy `together` ids in a few internal/config paths for
migration, but it is no longer a first-class public TogetherAI story.

### 2. Registry-level aggregation instead of duplicate provider implementations

The main provider composition happens in
`siumai-registry/src/registry/factories/togetherai.rs`.

That factory now aggregates three concrete clients:

- OpenAI-compatible `OpenAiCompatibleClient` for
  chat/completion/embedding/speech/transcription
- provider-owned `TogetherAiImageClient` for image generation/edit
- native `TogetherAiClient` for rerank

This keeps the runtime implementation thin and avoids maintaining two different TogetherAI text
stacks.

### 3. Unified client wrapper owns the public capability surface

`TogetherAiUnifiedClient` is a registry-layer wrapper that delegates:

- chat/completion/embedding/speech/transcription capabilities to the compat client
- image generation plus image extras to the provider-owned image client
- rerank capability to the native provider-owned client

This is the smallest refactor that lets `Provider::togetherai().build().await` expose the same
high-level shape as AI SDK without forcing a deeper provider-crate merger first.

### 4. Compat config still knows `togetherai`

`siumai-provider-openai-compatible` still includes a `togetherai` compat preset with the same:

- base URL
- primary chat default
- completion/embedding/speech/transcription family defaults
- shared family-default metadata reused by the unified provider for default-model lookup
- capability flags for the shared compat lanes

as the historical `together` preset.

That gives the registry a canonical AI SDK-style provider id for the shared text/audio lanes
without forcing the canonical image family back onto the generic compat image runtime.

### 5. Public facade follows the unified story

`Provider::togetherai()` now returns the normal unified `SiumaiBuilder` instead of a rerank-only
native builder.

The provider-owned native typed rerank surface still exists under `provider_ext::togetherai`.
The same extension module now also exposes typed TogetherAI image options/request helpers plus the
audited `TogetherAIErrorData` error envelope.

This preserves both stories:

- unified top-level provider ergonomics for most users
- native typed rerank/image escape hatches for users who specifically need provider-owned
  extensions

## Behavioral changes

- `togetherai` default text model is now the chat default
  `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`
- the unified image fallback model is now `black-forest-labs/FLUX.1-schnell` when callers do not
  explicitly choose an image model
- top-level `Provider::togetherai()` and `Siumai::builder().togetherai()` now build unified
  clients that can chat, generate/edit images, and rerank
- canonical TogetherAI image generation and edit both now call `POST {base}/images/generations`
- TogetherAI image edit now sends `image_url`, warns when multiple images are supplied, and
  rejects `mask` edits before transport
- public `TogetherAiImageOptions` now mirrors the audited AI SDK image option lane under
  `providerOptions.togetherai`
- the public alias surface now also exposes the exact AI SDK package-style names
  `TogetherAIImageModelOptions`, deprecated `TogetherAIImageProviderOptions`,
  `TogetherAIRerankingModelOptions`, and deprecated `TogetherAIRerankingOptions`, while keeping
  the older Rust `TogetherAi*` aliases available for compatibility
- `provider_ext::togetherai` now also exposes `TogetherAIErrorData`, so TogetherAI package-surface
  audits no longer have to fall back to generic compat error types just to decode provider errors
- registry metadata for `togetherai` now advertises the full AI SDK-style capability set instead of
  rerank only

## Validation strategy

- contract tests on the registry factory, including provider-owned image routing
- public-path parity tests across `Siumai::builder()`, `Provider::togetherai()`, provider-owned
  image generation/edit, config-first native rerank, and registry handles
- import-surface compile tests for the unified top-level builder plus the native typed rerank/image
  option path

## Remaining follow-up

- decide later whether the remaining low-level `together` compatibility lookup should be removed
  entirely after downstream migration
- decide later whether the unified wrapper should move from registry layer into a provider-owned
  TogetherAI package
