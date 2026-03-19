# Fearless Refactor V4 - Provider Package Alignment

Last updated: 2026-03-11

## Purpose

This document defines how Siumai should align **provider-owned packages** and
`provider_ext::<provider>` surfaces during the V4 refactor.

The goal is not to mechanically clone the Vercel AI SDK package list.
The goal is to keep a **Rust-first public contract** while borrowing the AI SDK's
good layering rules:

- provider-owned construction
- thin convenience builders
- typed provider escape hatches
- shared runtime internals hidden behind provider boundaries

## Decision Rules

### 1. Keep `builder` as convenience, not architecture

`Siumai::builder()` and `Provider::*()` should stay because they are useful,
ergonomic, and already familiar to existing users.

But builders must remain:

- thin
- convergent onto config-first construction
- unable to unlock provider-only capabilities that config-first cannot reach

### 2. `provider_ext::<provider>` is the preferred provider boundary

When a provider has stable vendor-specific semantics, users should find them under
`provider_ext::<provider>` rather than through raw `provider_options_map`.

Preferred exported sub-surfaces:

- `Client` / `Config`
- `options`
- `metadata` when response-specific fields are meaningful
- `tools` / `hosted_tools` / `provider_tools` when the provider owns tool semantics
- `resources` when the provider exposes non-family APIs
- `ext` only as an escape hatch, not the primary story

### 3. Do not promote every OpenAI-compatible preset into a dedicated package

OpenAI-compatible vendor presets should stay **preset-first** unless at least one
of these becomes true:

1. the vendor owns non-trivial request/response semantics beyond generic compat mapping
2. the vendor has stable typed metadata worth exposing publicly
3. the vendor has provider-owned resources/tools that are not just OpenAI-compat aliases
4. the vendor needs provider-owned auth/base-url/runtime rules that would otherwise leak through generic APIs

If none of those are true, the correct public story is:

- stable family API first
- config-first via `provider_ext::openai_compatible`
- vendor preset builder convenience second

## Alignment Tiers

### Tier A - Full provider-owned package

These providers should continue to own a rich `provider_ext::<provider>` surface.

- `openai`
- `anthropic`
- `gemini`
- `google_vertex`
- `minimaxi`
- `ollama`
- `deepseek`
- `xai`
- `groq`

Characteristics:

- provider-owned `Client` / `Config`
- typed `options`
- typed `metadata` where meaningful
- public-path parity coverage
- registry/config/builder convergence onto the same provider-owned client story

### Tier B - Focused provider-owned package

These providers should own a package, but only for the subset of capabilities they
actually differentiate today.

- `cohere`
- `togetherai`
- `bedrock`

Characteristics:

- provider-owned `Client` / `Config`
- typed request helpers only where there is real product value
- no requirement to invent missing metadata/resources just for symmetry

### Tier C - OpenAI-compatible vendor preset

These vendors should remain primarily under the shared compat family until they
earn promotion.

- `openrouter`
- `perplexity`
- `siliconflow`
- `together`
- `fireworks`
- other compat presets with no provider-owned public resources yet

Characteristics:

- public entry may exist as preset builders
- runtime can still expose typed request helpers when high-value
- avoid creating a separate provider package just to mirror vendor branding

For typed vendor views in this tier, apply `compat-vendor-view-contract.md`.

## Current Assessment

Public facade audit status:

- current provider-owned top-level packages now have explicit compile guards over the
  public surface they intentionally expose, instead of relying only on lower-crate tests
- rich native wrappers (`openai`, `anthropic`, `gemini`/`google`) now pin representative
  `tools` / `hosted_tools` / compatibility `provider_tools`, provider-owned `resources`,
  non-unified `ext` modules, and exposed helper accessors on the facade crate itself
- focused wrappers (`cohere`, `togetherai`, `bedrock`, `groq`, `xai`, `deepseek`,
  `ollama`, `minimaxi`) now also pin the narrower helper/tool/escape-hatch surface they
  intentionally own, without forcing artificial symmetry across metadata/resources
- `google_vertex` remains a focused wrapper rather than a full-spectrum native package, but its
  unique public helpers now include both Vertex-owned request/tool surfaces and a narrow typed
  metadata facade bound to the `vertex` namespace

## Native / wrapper providers already in good shape

### OpenAI / Anthropic / Gemini

These are the reference shape for a rich provider package:

- provider-owned `Client` / `Config`
- typed `options`
- typed `metadata`
- tool surfaces
- hosted tool builders
- provider-specific resources

### xAI / Groq / DeepSeek / Ollama / MiniMaxi

These providers already follow the intended V4 direction closely enough:

- provider-owned wrapper clients exist
- typed request helpers exist
- typed metadata exists
- public-path parity exists
- builders converge instead of forking behavior

The remaining work here is mostly **depth completion**, not **surface redesign**.

For providers in this group that intentionally keep a narrower native capability surface
than the shared runtime underneath, apply `focused-wrapper-contract.md`.

### Google Vertex

Vertex already has a provider-owned client/config story plus typed options and tool
surfaces, but it is still less complete than OpenAI / Anthropic / Gemini in package shape.

Audit result:

- `provider_ext::google_vertex` already exposes the parts users actually need today:
  `GoogleVertexClient` / `GoogleVertexConfig`, typed Imagen and embedding request helpers,
  Vertex-owned tool and hosted-tool entry points, plus provider-owned typed metadata helpers
  (`VertexMetadata`, `VertexChatResponseExt`, `VertexContentPartExt`)
- public-path parity already covers chat, chat stream, embedding, image generation, and
  image edit construction paths
- the visible response metadata now has a clearly separate Vertex-owned contract on the public
  surface: response/content-part extraction binds strictly to `provider_metadata["vertex"]`
  without falling back to shared Google/Gemini alias semantics, while custom stream events stay
  raw and namespace-scoped
- there is no obvious provider-owned resource surface yet that is mature enough to justify a
  `resources` module matching OpenAI or Anthropic

Recommendation:

- keep it Tier A
- do not force fake symmetry
- keep the new typed `metadata` module narrow and namespace-strict (`vertex` only)
- do not add `resources` yet just to make the module look complete

## Focused providers that should stay focused

### Cohere

Current story is healthy as a rerank-focused package.

Recommendation:

- keep `provider_ext::cohere` narrow
- do not inflate it into an artificial full-spectrum provider package
- prioritize rerank examples, contract tests, and request helper polish over new namespaces
- keep Cohere embedding on the shared OpenAI-compatible path for now; if users need Cohere embeddings, the public story should stay `Provider::openai().compatible("cohere")` or config-first `OpenAiCompatibleClient`, not a premature promotion of `provider_ext::cohere`

### TogetherAI

Current provider-owned package is rerank-focused and should stay that way.

Important distinction:

- `togetherai` native package is not the same thing as the OpenAI-compatible `together` preset

Recommendation:

- keep `provider_ext::togetherai` as a focused native package
- keep audio alignment for the OpenAI-compatible `together` preset under compat runtime rules
- do not merge the two stories into one confusing surface

### Bedrock

Bedrock owns construction/runtime/auth concerns strongly enough to stay provider-owned,
but it does not need to mimic OpenAI's surface area.

Recommendation:

- keep provider-owned config/client/builder
- keep request helpers for the capabilities we actually support
- avoid speculative metadata/resources until there is a stable public need

## Compat presets that should not be over-promoted yet

### SiliconFlow / Together / Fireworks

These presets now have meaningful audio behavior and public-path regression coverage,
but they still do not justify dedicated top-level provider packages yet.

Audit result:

- `siliconflow`, `together`, and `fireworks` now all have explicit capability and routing guards
  on the compat path
- public-path parity now locks Together TTS, SiliconFlow STT, and Fireworks transcription-only STT
  across builder/provider/config-first construction
- audio extras boundaries are now explicit too: the compat runtime exposes `TranscriptionExtras`
  when a preset advertises transcription, but `audio_translate` remains intentionally unsupported
  for `siliconflow`, `together`, and `fireworks`; public-path negative tests now lock that boundary
  across builder/provider/config-first and registry entry points before transport
- none of the three currently exposes a clearly provider-owned typed metadata or resource story that
  would justify promotion into a separate top-level provider package

Recommendation:

- keep them under the OpenAI-compatible preset story for now
- continue adding parity tests and capability guards
- do not mistake compat audio-family accessors for proof of a provider-owned extras contract;
  only promote if a vendor grows stable translation/streaming/metadata semantics that are stronger
  than the shared compat default story
- only promote one of them if typed metadata/resources/auth semantics become large enough

### OpenRouter / Perplexity

These already justify typed request or metadata helpers, but not necessarily a fully
separate provider package with its own client taxonomy.

Audit result:

- `provider_ext::openrouter` already works as a typed vendor view with request helpers, alignment
  tests, registry parity for provider-scoped reasoning defaults, and a runnable example, but it
  intentionally does not introduce a separate provider-owned client/config taxonomy
- `provider_ext::perplexity` already works as a typed vendor view with request helpers, typed
  response metadata, registry public-path parity, alignment tests, and a runnable example, again
  without needing a dedicated provider-owned client/config package
- public-surface compile guards now explicitly cover both `provider_ext::openrouter` and
  `provider_ext::perplexity`, which is the right level of stability for these vendor views

Recommendation:

- keep the current `provider_ext::{openrouter,perplexity}` vendor modules
- treat them as **typed vendor views over the compat runtime**, not as proof that every preset needs a full package
- keep registry-level vendor defaults flowing through `RegistryOptions` / `ProviderBuildOverrides` / shared `BuildContext` rather than introducing vendor-owned registry handle layers just to recover parity
- prefer adding compile guards, mapping tests, and examples before considering any package promotion
- use `compat-vendor-view-contract.md` as the default checklist for future OpenRouter / Perplexity surface work

## What We Should Build Next

### Priority 1 - Normalize provider package taxonomy

For every provider-owned package, make the following explicit in docs and exports:

- what is stable family API
- what is typed provider extension API
- what is only an escape hatch
- whether the package is full-spectrum, focused, or preset-only

### Priority 2 - Add a documented promotion policy for compat presets

Before creating any new `provider_ext::<vendor>` package, require a short written case:

- what semantics are vendor-owned
- why generic compat is insufficient
- what typed surface is worth stabilizing
- what tests/public examples will own the new surface

### Priority 3 - Keep audio aligned without over-creating packages

Audio is the current temptation point because vendor differences are real.

The correct policy is:

- keep Stable speech/transcription family contracts coherent
- keep provider-owned audio where the vendor really owns the contract (for example xAI TTS)
- keep compat audio for vendors that are still best modeled as OpenAI-compatible presets
- add explicit rejection tests when a provider only supports half the audio story

## Immediate TODO

- [x] Add a small "package tier" note into provider docs/examples where ambiguity still exists.
- [x] Promote `provider_ext::google_vertex` into a narrow provider-owned metadata surface bound only to `provider_metadata["vertex"]`, while keeping stream custom events and resource surfaces deferred.
- [x] Keep `provider_ext::bedrock` request-helper-focused unless Bedrock-specific response metadata grows into a stable typed public contract.
- [x] Keep `siliconflow` / `together` / `fireworks` on the compat path unless a promotion case is written.
- [x] Avoid creating new top-level provider packages for compat presets just to match the AI SDK package count.

## Summary

The refactor should optimize for **clear ownership**, not package quantity.

That means:

- keep builders
- keep provider-owned packages where semantics are real
- keep focused packages focused
- keep compat presets thin until they earn promotion
- keep `mistral` on the shared compat path: embedding parity is now covered through `Siumai::builder().openai().mistral()`, `Provider::openai().mistral()`, and config-first `OpenAiCompatibleClient`, without creating a standalone provider package just for construction symmetry
- prefer typed Rust-first provider boundaries over public API cloning
