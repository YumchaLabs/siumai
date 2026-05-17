# TogetherAI Unified Provider Surface - TODO

Last updated: 2026-04-20

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Canonical provider identity

- [x] Audit the AI SDK TogetherAI provider surface.
- [x] Document the historical split between `together` compat and the older `togetherai`
  rerank-led native surface, then supersede it with the unified public provider story.
- [x] Make `togetherai` the canonical public provider id.
- [x] Retire the old public `together` compat alias from builder/discovery surfaces.
- [x] Keep only low-level migration compatibility for legacy `together` lookup paths where needed.
- [x] Move the default TogetherAI model to the primary chat model.
- [x] Keep the public compat shortcut explicit as `togetherai_openai_compatible()` without letting
  the unified builder collide with the native `togetherai` provider id.

## Track B - Runtime/factory architecture

- [x] Reuse the shared OpenAI-compatible runtime for TogetherAI
  chat/completion/embedding plus Siumai extension speech/transcription.
- [x] Keep canonical TogetherAI image generation/edit on a provider-owned runtime instead of the
  generic compat image lane.
- [x] Keep native rerank on `siumai-provider-togetherai`.
- [x] Aggregate those three lanes inside the registry factory.
- [x] Expose a unified built client that surfaces text, image, and rerank capabilities together.
- [x] Align the provider-owned image request semantics with AI SDK TogetherAI.
  - generation and edit now both call `/images/generations`
  - edit now uses `image_url`, rejects `mask`, and warns on extra images
  - canonical `TogetherAiImageOptions` now flow under `providerOptions.togetherai`
- [x] Expand provider-catalog/default-model metadata so TogetherAI's unified family defaults are
  visible on the public registry/provider-info surface.
- [x] Feed the audited curated TogetherAI model subset into `provider_catalog.supported_models`
  instead of listing only defaults.
- [-] Implement a second native TogetherAI chat/completion runtime.
  - rejected for now because the shared compat runtime already matches the AI SDK provider shape
    more cheaply

## Track C - Public facade and regression coverage

- [x] Change `Provider::togetherai()` to the unified builder path.
- [x] Keep `provider_ext::togetherai` as the native typed rerank entry point.
- [x] Add registry contract tests for unified capabilities and override precedence.
- [x] Add public-path parity tests for chat and rerank behavior.
- [x] Add public-surface compile/runtime guards for the unified builder semantics.
- [x] Add regression coverage for provider-owned image generation/edit semantics, typed image
  options, default image fallback, and mask rejection.
- [x] Expose curated TogetherAI model constants and AI SDK-style option aliases on the public Rust
  typed surface for side-by-side package export checks.

## Track D - Docs and migration notes

- [x] Record the design in `docs/workstreams`.
- [x] Update structural-alignment docs that previously called the split an open gap.
- [x] Update changelog `Unreleased` sections instead of writing release notes.

## Track E - Follow-up audit

- [~] Re-audit the remaining AI SDK provider wrappers that still look like first-class candidates.
  - DeepInfra is now closed
  - Vertex MaaS is now closed
  - Cohere is now closed
- [~] Mirror the audited non-callable TogetherAI package exports where they are stable data
  structures rather than TS-only callable provider settings.
  - [x] `TogetherAIErrorData` is now exposed on `provider_ext::togetherai`
  - [x] `TogetherAIImageModelOptions` / `TogetherAIRerankingModelOptions` plus the deprecated
        package-compat aliases are now exposed on `provider_ext::togetherai`
  - [-] `TogetherAIProviderSettings` and `VERSION` remain intentionally deferred because the Rust
    provider facade uses `Config` / `Builder` / `Provider::togetherai()` as the stable
    constructor/settings story
- [ ] Decide whether the remaining low-level `together` compatibility lookup should be deleted
  entirely after downstream migration.
- [ ] Decide whether the unified TogetherAI wrapper should eventually move into a provider-owned
  package instead of living in `siumai-registry`.
