# Anthropic Package Surface Alignment - Design

Last updated: 2026-04-15

## Problem

Compared with `repo-ref/ai/packages/anthropic/src/index.ts`, Siumai's Anthropic surface was
already close on runtime behavior, but some important public names still drifted at the
provider-owned/package boundary:

- Rust exposed `AnthropicOptions`, while the AI SDK package exports
  `AnthropicLanguageModelOptions`
- the upstream package also keeps the deprecated migration alias `AnthropicProviderOptions`
- Rust exposed typed response metadata as `AnthropicMetadata`, while the AI SDK package exports
  `AnthropicMessageMetadata`
- `AnthropicUsageIteration` already existed in the lower metadata layer, but it was not re-exported
  on the stable `siumai::provider_ext::anthropic` facade
- tool-level `providerOptions.anthropic` for function tools still lacked a public typed
  `AnthropicToolOptions` helper even though the Anthropic protocol/runtime already understood part
  of that shape
- `@ai-sdk/amazon-bedrock` also re-exports `AnthropicProviderOptions`, but the stable Rust Bedrock
  facade did not mirror that cross-package edge at all

None of these were runtime bugs, but they made package-surface comparison against the reference
repo noisy and kept a few important data structures farther from the audited AI SDK contract than
necessary.

## Goals

- Align the Anthropic provider-owned/public naming layer more closely with the audited AI SDK
  package.
- Keep existing Rust-first names working so current callers do not need a forced migration.
- Re-export the important metadata names on the stable facade, not only in lower protocol crates.
- Mirror the Bedrock cross-package `AnthropicProviderOptions` export where Rust feature boundaries
  allow it cleanly.

## Non-goals

- Do not force `siumai-provider-amazon-bedrock` to depend directly on `siumai-provider-anthropic`
  just to mirror a TypeScript package re-export.
- Do not rename the existing Rust-first implementation types internally.
- Do not treat tool-level `AnthropicToolOptions` as solved in this workstream; that is a separate
  provider-owned typed helper gap.

## Chosen design

### 1. Keep the Rust-first implementation names as the storage layer

The concrete implementation types stay unchanged:

- `AnthropicOptions`
- `AnthropicMetadata`

This avoids churn in provider/runtime code and keeps existing Rust callers source-compatible.

### 2. Add AI SDK-style alias names at the provider-owned boundary

The Anthropic provider/package surface now also exposes:

- `AnthropicLanguageModelOptions = AnthropicOptions`
- deprecated `AnthropicProviderOptions = AnthropicLanguageModelOptions`
- `AnthropicMessageMetadata = AnthropicMetadata`

These aliases are intentionally thin type aliases rather than duplicate structs.

### 3. Re-export the same names on the stable `siumai::provider_ext::anthropic` facade

The stable facade now re-exports:

- `AnthropicLanguageModelOptions`
- `AnthropicProviderOptions`
- `AnthropicMessageMetadata`
- `AnthropicUsageIteration`
- `AnthropicToolOptions`

That keeps the public Rust surface easier to diff against the audited `@ai-sdk/anthropic` index.

### 4. Close the tool-level typed helper gap honestly

The upstream package also exports `AnthropicToolOptions`, and this workstream now treats that as a
runtime-backed public surface rather than a naming-only shim.

Rust now exposes:

- `AnthropicToolAllowedCaller`
- `AnthropicToolOptions`
- `AnthropicToolExt` for `Tool` / `ToolFunction`

The runtime path was brought up to the same minimum semantic level:

- `with_anthropic_options(...)` now merges onto existing `providerOptions.anthropic` objects
  instead of overwriting sibling raw fields
- `with_anthropic_tool_options(...)` does the same on function tools
- Anthropic protocol tool conversion now forwards `eagerInputStreaming` in addition to the already
  supported `deferLoading` / `allowedCallers`
- the experimental Anthropic request bridge now restores `eagerInputStreaming` on tool provider
  options as well

### 5. Mirror Bedrock's cross-package Anthropic alias only on the facade

Upstream `@ai-sdk/amazon-bedrock` re-exports `AnthropicProviderOptions` from the Anthropic package.

Rust should not force the Bedrock provider crate to depend on the Anthropic provider crate just for
that naming edge, because Bedrock is feature-gated independently and already owns its Anthropic-on-
Bedrock runtime behavior directly.

So the chosen compromise is:

- keep the provider crates decoupled
- conditionally re-export `AnthropicProviderOptions` from `siumai::provider_ext::bedrock`
  only when both `bedrock` and `anthropic` features are enabled

This preserves a clean Cargo boundary while still making the stable facade closer to the audited AI
SDK package surface in the combined-feature case.

## Validation

This workstream is locked by:

- alias tests in `siumai-provider-anthropic/src/provider_options/anthropic/mod.rs`
- tool helper/merge tests in
  `siumai-provider-anthropic/src/providers/anthropic/ext/{request_options,tools}.rs`
- metadata alias tests in `siumai-protocol-anthropic/src/provider_metadata/anthropic.rs`
- protocol tool-conversion tests in
  `siumai-protocol-anthropic/src/standards/anthropic/utils/tools.rs`
- stable public compile guards in `siumai/tests/public_surface_imports_test.rs`

## Remaining follow-up

- Re-check if other Anthropic package-index exports need stable facade mirrors after the next AI SDK
  update.
