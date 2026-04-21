# Shared Type Surface Alignment - Design

Last updated: 2026-04-21

## Problem

Compared with the shared AI package surface in `repo-ref/ai/packages/ai/src/types/index.ts`,
Siumai already had most of the runtime carriers, but the public shared type layer still drifted in
several important ways:

- the public Rust surface did not expose audited shared names such as `JSONValue`,
  `CallWarning`, `ProviderMetadata`, `ImageModelProviderMetadata`,
  `LanguageModelRequestMetadata`, `LanguageModelResponseMetadata`,
  `ImageModelResponseMetadata`, `SpeechModelResponseMetadata`,
  `TranscriptionModelResponseMetadata`, `EmbeddingModelUsage`, and `ImageModelUsage`
- the stable runtime carrier `ResponseMetadata` had no `headers` slot, so the public Rust surface
  could not honestly mirror the upstream response-metadata contract
- the shared warning surface still missed the AI SDK `deprecated` category
- the stable Rust prelude did not re-export these shared data structures, which kept public audits
  noisy even when the internal runtime had most of the needed information
- the current `Usage` type is a compatibility-oriented superset of provider V4 usage, but the AI
  package also exposes a higher-level `LanguageModelUsage` projection that Siumai did not model
  explicitly

This was not a single runtime bug. It was a shared contract gap:

- public comparison against `repo-ref/ai/packages/ai/src/types/*` stayed harder than necessary
- metadata and usage helpers had to reuse nearby internal carriers with mismatched names or shapes
- downstream Rust code had no stable, audited entry point for the AI SDK-style shared data layer

## Goals

- Audit the shared AI package surface as a public contract, not only as internal runtime data.
- Expose the stable shared names that already have honest Rust equivalents.
- Add missing shared metadata carriers where the runtime can provide them without pretending that
  all wiring is already complete.
- Keep provider-owned typed options and typed provider metadata in provider crates.
- Make future audits against `repo-ref/ai/packages/ai/src/types/*` cheaper and more mechanical.

## Non-goals

- Do not mirror every TypeScript prompt/helper type mechanically.
- Do not introduce `RequestOptions`, `TimeoutConfiguration`, or
  `LanguageModelCallOptions` in this workstream without a separate design.
- Do not fabricate richer runtime capture than Siumai currently has; for example, speech response
  `body` stays optional because the shared carrier is now present before the runtime is fully
  wired.

## Chosen design

### 1. Add one dedicated shared-type module in `siumai-spec`

`siumai-spec/src/types/ai_sdk.rs` becomes the public home for shared AI SDK-style names that are
not provider-owned:

- aliases:
  - `JSONValue`
  - `CallWarning`
  - `ProviderMetadata`
  - `ProviderOptions`
  - `Context`
  - `ImageModelProviderMetadata`
- usage/data structures:
  - `ToolCall`
  - `ToolResult`
  - `LanguageModelUsage`
  - `LanguageModelInputTokenDetails`
  - `LanguageModelOutputTokenDetails`
  - `EmbeddingModelUsage`
  - `ImageModelUsage`
  - `LanguageModelRequestMetadata`
  - `LanguageModelResponseMetadata`
  - `ImageModelResponseMetadata`
  - `SpeechModelResponseMetadata`
  - `TranscriptionModelResponseMetadata`

This keeps the shared audit surface local instead of scattering these names across unrelated
runtime modules.

### 2. Use honest conversion helpers instead of fake direct aliases

Some names can be direct aliases (`JSONValue`, `CallWarning`, `ProviderMetadata`), but the more
important shared carriers need explicit Rust structs:

- `LanguageModelUsage` is now a projection from `Usage`, because the AI package shape is not the
  same as provider V4 usage
- `ProviderOptions` and `Context` are honest aliases onto the existing open JSON-object carriers
- `ToolCall` and `ToolResult` are passive Rust data structures that mirror the provider-utils
  helper shape without pretending Siumai already has one exact runtime source for every typed tool
  helper result
- `EmbeddingModelUsage` stays the audited one-field shape `{ tokens }`
- `ImageModelUsage` mirrors the AI package `ImageModelV4Usage` token totals
- request/response metadata structs expose the AI package fields directly and convert from the
  existing runtime carriers

This avoids the trap of “matching the name but not the data structure”.

### 3. Extend stable runtime carriers where the public contract requires it

Two shared runtime carriers needed real semantic widening:

- `ResponseMetadata` now includes optional `headers`
- `Warning` now includes the `Deprecated { setting, message }` category

Those are not cosmetic changes. They are stable contract slots used directly by the audited AI SDK
surface.

### 4. Keep runtime honesty for partially wired metadata

Some public shared structs are now available before every runtime path is fully upgraded:

- `SpeechModelResponseMetadata.body` remains optional and current HTTP-only conversions leave it as
  `None`
- `LanguageModelRequestMetadata.body` converts from the current serialized request body and falls
  back to raw string JSON values when parsing is not possible

This is intentional. The public data structures now exist, but they do not claim stronger runtime
capture than Siumai currently has.

### 5. Re-export the shared layer through the stable facade

The stable Rust facade now re-exports the shared names through:

- `siumai::types::*`
- `siumai::prelude::unified::*`

That keeps the shared surface visible in the same public places where users already import the
stable family/request/response contracts.

## Validation

This workstream is currently locked by:

- `cargo check -p siumai-spec --tests`
- `cargo nextest run -p siumai --test public_surface_imports_test`
- public compile guards in `siumai/tests/public_surface_imports_test.rs`
- local unit tests in `siumai-spec/src/types/{common,ai_sdk}.rs`

## Deferred follow-up

- Audit whether the AI package `LanguageModelUsage` helper functions deserve public Rust helper
  functions instead of only `From<Usage>` plus `merge`.
- Design `RequestOptions`, `TimeoutConfiguration`, and `LanguageModelCallOptions` separately.
- Revisit response-body capture for speech/transcription once the runtime has a stable place to
  preserve it across providers.
