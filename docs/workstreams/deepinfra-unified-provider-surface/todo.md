# DeepInfra Unified Provider Surface - TODO

Last updated: 2026-04-10

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Canonical provider identity

- [x] Audit the AI SDK DeepInfra provider surface.
- [x] Make `deepinfra` a first-class built-in provider id instead of only a compat preset.
- [x] Register native provider metadata for DeepInfra, including provider-owned image capability.
- [x] Align the public default chat model with the audited AI SDK DeepInfra default.

## Track B - Runtime/factory architecture

- [x] Reuse the shared OpenAI-compatible runtime for chat/completion/embedding.
- [x] Keep DeepInfra image generation/edit on provider-owned routes.
- [x] Aggregate those lanes inside a registry-layer unified factory/client.
- [x] Preserve native registry metadata when attaching the OpenAI-compatible adapter.
- [-] Create a separate provider-owned DeepInfra text crate.
  - rejected for now because the shared compat runtime already matches the AI SDK DeepInfra text
    surface with much lower maintenance cost

## Track C - Data structure and semantic alignment

- [x] Correct DeepInfra OpenAI-compatible usage normalization when reasoning tokens exceed the
  reported completion total.
- [x] Promote `ProviderType::DeepInfra` so DeepInfra no longer falls back to
  `Custom("deepinfra")` in the stable/provider catalog layers.
- [x] Align provider catalog output with DeepInfra native metadata plus an audited curated
  text/completion/embedding/image model subset.

## Track D - Public facade and regression coverage

- [x] Expose `Provider::deepinfra()` / `Siumai::builder().deepinfra()` on the unified builder path.
- [x] Add registry contract tests for unified capabilities and provider-owned image routes.
- [x] Add public-path parity tests for chat, image generation, and image edit behavior.
- [x] Add public-surface compile/runtime guards for the unified builder semantics.
  - compile coverage now also pins `provider_ext::deepinfra::{chat, completion, embedding, image,
    model_sets}` so the promoted public namespace cannot silently drift away from the curated
    DeepInfra subset

## Track E - Docs and follow-up

- [x] Record the design in `docs/workstreams`.
- [x] Update structural-alignment docs that previously treated DeepInfra as an open provider audit.
- [x] Update changelog `Unreleased` sections instead of writing release notes.
- [-] Mirror TypeScript-only package exports such as `DeepInfraProviderSettings`,
  `DeepInfraErrorData`, or `VERSION` one-for-one on the Rust side.
  - deferred intentionally because the Rust public contract already uses unified builder/config
    surfaces instead of TS-style package settings/error-data aliases, and the current hybrid
    DeepInfra wrapper does not justify inventing a one-off Rust pattern here
- [~] Continue the broader AI SDK first-class provider audit after DeepInfra.
  - Vertex MaaS remains open
