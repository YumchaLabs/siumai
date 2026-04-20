# OpenAI-Compatible Package Surface Alignment - Design

Last updated: 2026-04-20

## Problem

Compared with `repo-ref/ai/packages/openai-compatible/src/index.ts`, Siumai already had the main
shared OpenAI-compatible runtime, but the package boundary still drifted in a few important ways:

- generic package-level typed option structs for chat, completion, and embedding were missing on
  the provider-owned/public Rust facade
- generic package-level error helpers such as `OpenAICompatibleErrorData` and
  `ProviderErrorStructure<T>` had no comparable Rust names
- the shared `providerOptions.openaiCompatible` helper lane was incomplete because only chat had a
  first-class typed request-ext path on the public facade
- the audited upstream package also exports `OpenAICompatibleImageModelId`, while Siumai only had
  chat/completion/embedding model-id aliases on the public compat surface
- this package-surface work had no dedicated workstream, which made future audits harder to track

The result was not a single runtime bug. It was a package-shape and data-structure gap:

- public comparison against `repo-ref/ai/packages/openai-compatible/src/index.ts` stayed noisy
- generic compat options had to fall back to raw JSON more often than necessary
- the Rust public story for image-capable compat providers was honest at runtime but incomplete at
  the package-data-structure layer

## Goals

- Audit `repo-ref/ai/packages/openai-compatible/src/index.ts` as a package boundary, not just as a
  shared transport/runtime implementation.
- Expose the audited generic typed surface on the provider-owned/public Rust facade:
  `OpenAiCompatibleLanguageModel{Chat,Completion}Options`,
  `OpenAiCompatibleEmbeddingModelOptions`,
  `OpenAiCompatible{Chat,Completion,Embedding,Image}ModelId`,
  `OpenAiCompatibleErrorData`, and `ProviderErrorStructure<T>`.
- Make the shared `providerOptions.openaiCompatible` helper lane work across chat, completion, and
  embedding requests.
- Keep the Rust public story honest about the parts that still differ from the TypeScript package.

## Non-goals

- Do not fabricate TypeScript-only exports such as `createOpenAICompatible`,
  `OpenAICompatibleProvider`, `OpenAICompatibleProviderSettings`, or `VERSION` on the Rust
  facade.
- Do not invent a Rust `OpenAiCompatibleImageModel` wrapper type just to mirror the TypeScript
  callable class; the stable Rust image surface already lives through `OpenAiCompatibleClient`,
  registry `image_model(...)`, and `siumai::image::*`.
- Do not widen this workstream into provider-specific compat packages such as DeepInfra, Fireworks,
  or OpenRouter beyond the shared generic surface they reuse.

## Chosen design

### 1. Match the audited generic package names directly

The provider-owned/public Rust compat facade now carries the main generic names expected from the
audited AI SDK package:

- `OpenAiCompatibleLanguageModelChatOptions`
- `OpenAiCompatibleLanguageModelCompletionOptions`
- `OpenAiCompatibleEmbeddingModelOptions`
- deprecated migration aliases for the same option structs
- `OpenAiCompatibleErrorData`
- `ProviderErrorStructure<T>`
- `OpenAiCompatible{Chat,Completion,Embedding,Image}ModelId`
- `MetadataExtractor`

This keeps `provider_ext::openai_compatible::{options::*, *}` much easier to compare one-to-one
against `repo-ref/ai/packages/openai-compatible/src/index.ts`.

### 2. Carry the shared `openaiCompatible` helper lane across the real request families

The audited generic helper lane now exists on the stable Rust surface:

- `OpenAiCompatibleChatRequestExt::with_openai_compatible_options(...)`
- `OpenAiCompatibleCompletionRequestExt::with_openai_compatible_options(...)`
- `OpenAiCompatibleEmbeddingRequestExt::with_openai_compatible_options(...)`

Those helpers merge onto the shared `provider_options_map["openaiCompatible"]` namespace instead of
overwriting sibling raw fields. The merge logic is also centralized so the three request families
cannot drift in behavior silently.

### 3. Expose image model ids without fabricating a TypeScript callable class

The audited upstream package exports `OpenAICompatibleImageModelId = string`. Rust now mirrors that
as `OpenAiCompatibleImageModelId = String` because the runtime already owns real image execution:

- `OpenAiCompatibleClient` implements image generation/edit/variation capability
- compat presets such as Fireworks, DeepInfra, and TogetherAI already route real image work on the
  Rust side
- registry `image_model(...)` and `siumai::image::*` already provide the stable callable contract

What Rust still intentionally does **not** mirror is the TypeScript `OpenAICompatibleImageModel`
class itself. The runtime capability exists; the callable class export does not need a second Rust
wrapper to be honest.

### 4. Keep provider-construction exports deferred

The upstream package also exports `createOpenAICompatible`, `OpenAICompatibleProvider`, and
`OpenAICompatibleProviderSettings`.

Rust already represents that construction/config story through:

- `OpenAiCompatibleBuilder`
- `OpenAiCompatibleConfig`
- `OpenAiCompatibleClient`

This workstream keeps that existing Rust-first contract instead of inventing fake callable provider
types just for naming parity.

## Current implemented parity in this workstream

This workstream currently closes the following shared compat package-surface gaps:

- `provider_ext::openai_compatible::{options::*, *}` now exposes the audited generic typed option,
  error, extractor, and model-id names
- the shared `openaiCompatible` request-helper lane now exists across chat, completion, and
  embedding requests
- the generic compat image surface now also exposes `OpenAiCompatibleImageModelId`
- the merge behavior for typed generic compat request helpers is now centralized instead of being
  duplicated per request family

## Validation

The current slice is locked by:

- provider-option tests in
  `siumai-provider-openai-compatible/src/provider_options/openai_compatible.rs`
- provider-local request-ext and error-surface tests in
  `siumai-provider-openai-compatible/src/providers/openai_compatible/{ext/request_options.rs,mod.rs}`
- public surface compile guards in `siumai/tests/public_surface_imports_test.rs`
- top-level public-path parity coverage on the real compat wrapper paths in
  `siumai/tests/provider_public_path_parity_test.rs`

## Remaining follow-up

- Re-audit whether upstream `@ai-sdk/openai-compatible` adds more public generic data structures
  that deserve direct Rust mirrors.
- Re-evaluate later whether Rust should add package-level builder convenience aliases for provider
  construction, without pretending to be a TypeScript callable provider.
- If the upstream package ever exports a public generic image-option type, revisit whether the Rust
  compat facade should grow a comparable typed helper lane for image requests.
