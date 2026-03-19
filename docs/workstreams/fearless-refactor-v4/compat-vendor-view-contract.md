# Fearless Refactor V4 - Compat Vendor View Contract

Last updated: 2026-03-17

## Purpose

This document defines the contract for a **compat vendor view**.

A compat vendor view is a vendor-specific typed layer that:

- stays on the shared `openai_compatible` runtime
- may expose typed request helpers under `provider_ext::<vendor>`
- may expose typed response metadata only when there is stable provider evidence
- must not be described as a provider-owned `Config` / `Client` package unless it earns promotion later

Current examples:

- `openrouter`
- `perplexity`

For package-tier decisions, see `provider-package-alignment.md`.
For the public entry ordering, see `public-api-story.md`.

## Core rule

If a vendor remains on the shared compat runtime, the public contract must be
**shared by ownership, explicit by capability source, honest by metadata, and
light by docs/tests**.

That means:

1. runtime ownership stays with `OpenAiCompatibleClient` and compat registry config
2. vendor capability declarations come from the compat preset registry, not a fake native package
3. typed vendor helpers are additive ergonomics, not proof of provider-owned package symmetry
4. typed metadata exists only where stable provider evidence already exists
5. docs and tests keep the vendor-view story narrow and accurate

## When to use this contract

Apply this contract when all of the following are true:

- the vendor still fits the shared OpenAI-compatible transport/protocol runtime
- typed vendor-specific request helpers provide real user value
- some vendor-specific metadata may be worth exposing, but only for clearly evidenced fields
- a provider-owned `Config` / `Client` taxonomy would currently be more ceremony than value

Do **not** use this contract for:

- full provider-owned packages with their own runtime/auth/resource story
- focused wrapper packages that already expose provider-owned `Config` / `Client`
- preset-only aliases that do not yet justify any typed vendor extension surface

## Required invariants

### 1. Ownership invariant

Compat vendor views must remain visibly layered on the shared compat runtime.

Required behavior:

- config-first construction stays `OpenAiCompatibleClient`-based
- `Provider::openai().<vendor>()` shortcuts resolve shared compat defaults
- docs/examples do not imply a separate provider-owned client taxonomy when none exists

### 2. Capability invariant

Capability declarations must come from the compat preset registry/config story.

Required behavior:

- builder shortcuts, config-first construction, and registry handles resolve the same vendor capability split
- vendor-scoped defaults that matter publicly (for example OpenRouter reasoning) flow through `ProviderBuildOverrides` / shared `BuildContext` on the registry path instead of a fake native config layer
- registry-wide defaults for those same knobs use `RegistryOptions` / `RegistryBuilder` but still converge into that same shared `BuildContext` path
- capability decisions are encoded once in compat preset metadata instead of being duplicated in a fake native layer
- unsupported families stay unsupported even if the shared runtime supports them for other vendors

Current examples:

- `openrouter`: `tools + embedding + reasoning`, but no audio
- `perplexity`: `tools`, but no separate embedding or audio promotion

### 3. Typed extension invariant

Typed vendor helpers should live under `provider_ext::<vendor>`.

Required behavior:

- request helpers are vendor-owned in naming, even if implementation reuses shared compat mapping
- only stable/high-value vendor knobs get typed helpers
- raw `with_provider_option(...)` escape hatches remain available for long-tail vendor parameters

### 4. Metadata invariant

Typed metadata should exist only when backed by real provider evidence.

Required behavior:

- expose typed metadata only for fields with stable public meaning
- reuse shared compat extraction helpers when metadata is derived from the shared runtime
- keep typed metadata bound to the vendor-owned root (`provider_metadata["vendor"]`) instead of silently falling back to a generic compat namespace
- keep non-streaming and streaming `StreamEnd` extraction rules aligned when typed metadata is exported

Current examples:

- `perplexity` has typed hosted-search metadata worth exposing
- `openrouter` can expose an alias-based typed metadata view for stable `sources` / `logprobs`, but it should stay vendor-namespaced and must not imply a promoted provider-owned package

### 5. Documentation invariant

Docs, examples, and matrices must describe these surfaces as **compat vendor views**.

Required documentation effects:

- package-tier docs keep vendor views separate from provider-owned packages
- examples stay on the compat narrative unless promotion is intentional
- no doc should imply that typed vendor helpers automatically require a dedicated client/config package

### 6. Test invariant

Tests must protect the vendor-view boundary at the right layers.

Required behavior:

- preset guards lock capability split and builder-default convergence
- registry handle tests lock provider-scoped build overrides and final request-body merge behavior
- registry-scoped vendor defaults are verified on the final transport boundary, not only in config-local builder tests
- registry-global defaults are verified on that same final transport boundary when they are part of the public story
- top-level public-path parity exists only for the families the compat vendor view actually advertises
- typed metadata tests cover both non-streaming and streaming end-state when metadata is exported publicly

## Checklist

When promoting or auditing a compat vendor view, verify all items below.

### Surface

- [ ] shared compat runtime remains the canonical execution path
- [ ] builder shortcuts converge to shared compat config defaults
- [ ] typed vendor helpers add value without creating fake package symmetry

### Capability boundary

- [ ] compat preset metadata declares the final capability split
- [ ] builder/config/registry paths agree on the same capability split
- [ ] registry-scoped vendor defaults reuse `ProviderBuildOverrides` / `BuildContext`
- [ ] registry-global vendor defaults reuse `RegistryOptions` / `BuildContext`
- [ ] unsupported families are not over-promoted in docs or helpers

### Typed extensions

- [ ] request helpers cover only stable/high-value vendor knobs
- [ ] raw provider-option fallback still works for escape hatches
- [ ] public naming stays vendor-owned and Rust-first

### Metadata

- [ ] typed metadata exists only where provider evidence is strong enough
- [ ] shared extraction helper is reused where possible
- [ ] non-streaming and streaming end-state stay aligned when metadata is exported

### Tests

- [ ] preset guards lock capability split
- [ ] registry-handle tests lock provider-specific override precedence
- [ ] public-path parity covers any advertised top-level family path
- [ ] metadata tests cover both 200-response and `StreamEnd` when applicable

### Docs

- [ ] public API docs describe the surface as a compat vendor view
- [ ] provider package alignment points to this contract
- [ ] TODO/milestone notes mention the boundary explicitly
- [ ] examples stay within the compat vendor-view story

## Preferred test pattern

Use the lightest test that proves the public claim.

Recommended order:

1. preset guards for capability split and builder-default convergence
2. registry-handle request capture for provider-scoped override precedence
3. top-level public-path parity only for the family paths the vendor actually advertises
4. typed metadata tests for both non-streaming and streaming end-state only when that metadata is publicly exported

Current anchor files:

- `siumai/tests/openai_compatible_preset_guards_test.rs`
- `siumai/tests/provider_public_path_parity_test.rs`
- `siumai-registry/src/registry/factories/contract_tests.rs`

## Non-goals

This contract does **not** require:

- inventing provider-owned `Config` / `Client` types for every compat vendor
- promoting every preset with a few custom knobs into a standalone package
- duplicating capability declarations in both compat registry config and a fake native layer
- flattening vendor-specific metadata into the Stable family surface too early

## Promotion heuristic

If a compat vendor view starts accumulating:

- provider-owned auth/base-url/runtime rules
- stable non-compat resources or tools
- multiple capability families with provider-owned semantics
- broader typed metadata that no longer fits the shared compat ownership story

then it may be time to promote it into a fuller provider package or focused wrapper.

Until then, keep it as a compat vendor view and keep the boundary explicit.
