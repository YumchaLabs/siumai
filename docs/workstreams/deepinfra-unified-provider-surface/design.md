# DeepInfra Unified Provider Surface - Design

Last updated: 2026-04-10

## Problem

AI SDK models DeepInfra as one provider surface:

- `chatModel()`
- `completionModel()`
- `embeddingModel()`
- `imageModel()`

with a split runtime behind that surface:

- text/completion/embedding use OpenAI-compatible `/openai/*`
- image generation uses provider-owned `/inference/{model}`
- image editing uses provider-owned `/openai/images/edits`

Siumai had drifted into an awkward hybrid state:

- `deepinfra` existed as an OpenAI-compatible preset, but not as a first-class built-in provider
- registry/native metadata did not own the DeepInfra image capability story
- OpenAI-compatible adapter registration could overwrite native registry metadata on hybrid
  providers
- DeepInfra-specific usage totals were not corrected when the provider reported reasoning tokens
  larger than `completion_tokens`
- stable/provider catalog layers still treated DeepInfra as `Custom("deepinfra")`

## Goals

- Make `deepinfra` a first-class built-in provider id.
- Expose one public provider surface for chat/completion/embedding plus provider-owned image
  generation/edit.
- Reuse the shared OpenAI-compatible runtime for the text families instead of building a second
  DeepInfra text stack.
- Keep native provider metadata authoritative when the compat adapter is attached.
- Correct the known DeepInfra usage-accounting drift on OpenAI-compatible responses/streams.
- Align public provider typing, registry catalog output, and docs/tests with that provider story.

## Non-goals

- Do not build a brand-new provider-owned DeepInfra text crate right now.
- Do not pretend DeepInfra supports provider-owned image variations when the audited AI SDK surface
  does not expose them.
- Do not close the whole first-class-provider audit in the same pass; Vertex MaaS remains a
  follow-up candidate.

## Chosen design

### Architectural rule

This workstream now treats one point as fixed by the audited AI SDK reference:

- AI SDK does expose a shared image interface through `ProviderV4.imageModel(...)`,
  `ImageModelV4`, and `generateImage(...)`
- that shared layer standardizes call shape and result aggregation, not transport/runtime
  ownership
- DeepInfra image execution must therefore stay provider-owned even though its text families still
  reuse the shared OpenAI-compatible runtime

That matches the upstream split more closely than trying to force DeepInfra image generation/edit
back into a generic compat image executor.

### 1. Canonical id stays `deepinfra`

`deepinfra` is now a built-in native provider id in registry metadata, factory selection, provider
builder helpers, public `ProviderType`, and provider catalog output.

The OpenAI-compatible preset still exists because the text families are implemented through the
shared compat runtime, but it no longer acts like the canonical whole-provider story by itself.

### 2. Registry-level aggregation instead of a duplicate provider crate

The main provider composition happens in
`siumai-registry/src/registry/factories/deepinfra.rs`.

That factory builds:

- an OpenAI-compatible client rooted at `{base}/openai` for chat/completion/embedding
- a provider-owned image client rooted at `{base}/inference` plus `{base}/openai/images/edits`

The unified client exposes both lanes behind one `deepinfra` provider id.

### 3. Native metadata owns the image capability, compat preset owns the text defaults

DeepInfra is a hybrid provider.

The compat preset remains the source of shared text-family defaults such as:

- chat default model
- embedding default model
- image-family fallback model ids used by the shared compat default table

But the native registry metadata owns the provider-level capability declaration, especially
`image_generation`.

That separation lets the registry describe DeepInfra correctly without teaching the generic compat
runtime that every compat preset suddenly has provider-owned image routes.

### 4. Compat adapter registration must merge, not overwrite

Hybrid providers such as TogetherAI and DeepInfra need both:

- native registry/provider metadata
- OpenAI-compatible adapter/runtime information

`register_openai_compatible_from_config(...)` now merges compat adapter info into an existing
native record instead of replacing native metadata fields such as provider name, capabilities, and
root base URL.

### 5. Usage normalization is provider-aware in shared OpenAI-compatible parsing

DeepInfra can report:

- `completion_tokens`
- `reasoning_tokens`

where `reasoning_tokens > completion_tokens`, which makes the raw totals internally inconsistent.

The shared OpenAI-compatible usage parser now routes through
`parse_provider_openai_usage_value(provider_id, value)` so DeepInfra can normalize those totals
before the stable `Usage` model is built.

### 6. Public typing/cross-layer docs are aligned

`ProviderType::DeepInfra`, provider catalog output, workstream docs, contract tests, public-path
tests, and public-surface compile tests now all describe the same first-class provider story.

## Behavioral changes

- `Provider::deepinfra()` and `Siumai::builder().deepinfra()` now build a unified client instead of
  falling back to a generic compat-only interpretation.
- registry-native DeepInfra metadata now advertises chat/completion/streaming/tools/vision/
  embedding/image-generation.
- DeepInfra image generation uses `{root}/inference/{model}`.
- DeepInfra image editing uses `{root}/openai/images/edits`.
- DeepInfra usage normalization now repairs inconsistent reasoning/completion totals before they
  enter the stable `Usage` layer.
- stable/provider catalog layers now treat DeepInfra as `ProviderType::DeepInfra`.
- the public facade now also exposes curated
  `provider_ext::deepinfra::{chat, completion, embedding, image, model_sets}` constants, and the
  provider catalog reuses that same audited model subset instead of listing only family defaults.

## Validation strategy

- registry contract tests for unified capabilities plus image-generation/edit routing
- public-path parity tests across `Siumai::builder()`, `Provider::deepinfra()`, and registry
  handles
- public-surface compile tests for the promoted `provider_ext::deepinfra` model-constant modules
- shared OpenAI-compatible usage parsing tests for DeepInfra reasoning totals
- provider catalog and `ProviderType` tests for the new first-class provider typing

## Remaining follow-up

- Decide later whether DeepInfra should eventually gain a provider-owned typed wrapper crate of its
  own or stay as a registry-layer hybrid wrapper.
- Keep TypeScript-only package exports such as `DeepInfraProviderSettings`, `DeepInfraErrorData`,
  and `VERSION` intentionally deferred on the Rust side unless a shared Rust provider-package
  pattern emerges first.
- Continue the AI SDK provider audit with Vertex MaaS and any other remaining first-class
  candidates.
