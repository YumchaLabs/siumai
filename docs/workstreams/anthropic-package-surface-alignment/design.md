# Anthropic Package Surface Alignment - Design

Last updated: 2026-04-22

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
- the package-level `forwardAnthropicContainerIdFromLastStep(...)` helper from the audited
  `@ai-sdk/anthropic` index had no provider-owned/stable Rust counterpart even though Siumai
  already surfaced the same Anthropic container metadata
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
- `AnthropicMessageMetadata`
- `find_anthropic_container_id_from_last_step(...)`
- `forward_anthropic_container_id_from_last_step(...)`

`AnthropicMessageMetadata` is now a dedicated typed struct instead of a thin alias to
`AnthropicMetadata`.

That split is intentional:

- `AnthropicMessageMetadata` mirrors the narrower AI SDK typed message-metadata contract
  (`usage`, `stopSequence`, `iterations`, `container`, `contextManagement`)
- `AnthropicMessageMetadata.container` now uses dedicated required-field message container/skill
  structs, so the narrow public surface no longer inherits the wider helper's optional
  `id` / `expiresAt` / skill-field looseness
- `AnthropicMetadata` remains the wider Rust helper that can derive convenience fields such as
  `service_tier`, `server_tool_use`, or reasoning replay metadata from the raw nested usage/content
  lane

This keeps the public typed package surface honest without deleting the broader Rust helper API.

The container-forward helpers are intentionally package-owned helpers rather than orchestrator-only
utilities:

- the provider package now exports `find_anthropic_container_id_from_last_step(...)` for callers
  that only need the latest container id
- it also exports `forward_anthropic_container_id_from_last_step(...)`, returning a stable
  `ProviderOptionsMap` override that mirrors the upstream prepare-step helper output
- the input stays generic over any history that can yield optional `ProviderMetadataMap`
  references, so callers are not forced onto one specific orchestrator step type

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

### 6. Follow-on: consolidate native structured output and task budget onto `output_config`

After the initial package-surface/name audit, a second Anthropic drift remained in the request
layer when compared with the current
`repo-ref/ai/packages/anthropic/src/anthropic-messages-language-model.ts`:

- native structured JSON output was still lowered to deprecated `output_format` instead of
  `output_config.format`
- `effort`, future `task_budget`, and native structured-output writes were assembled through
  separate overlay paths that could overwrite one another
- the provider-owned typed surface still lacked upstream `inferenceGeo`, narrowed `effort` to
  exclude `xhigh`, and treated adaptive thinking as a bare tag so `display` was lost
- the request bridge/normalizer still restored the old `output_format` path first, so exact
  same-protocol inspection drifted away from the request builder
- `container.skills` still used one flattened local struct, so the audited public split between
  Anthropic `skillId` and custom `providerReference` was lost on typed serde and bridge
  normalization
- stream-time structured-output mode inference used an older "tools present => jsonTool" rule
  instead of the same `supportsNativeStructuredOutput` gate used by the request body

The chosen follow-on design is:

- `structuredOutputMode = outputFormat` now lowers native JSON Schema only to
  `output_config.format`
- the legacy `output_format` lane stays only as a compatibility/fallback path for older
  `json_object`-style behavior where the audited AI SDK still uses it
- Anthropic request overlays now build one shared `output_config` object so `format`, `effort`,
  and `task_budget` can coexist without field loss
- the provider-owned typed Anthropic options now expose `taskBudget`, `inferenceGeo`,
  `AnthropicThinkingDisplay`, and the full audited effort enum including `xhigh`, including fluent
  builder/config helpers and beta-header inference where applicable
- bridge-side request normalization now prefers `output_config.format` and restores
  `output_config.task_budget` back onto `providerOptions.anthropic.taskBudget`, while still
  accepting legacy `output_format`
- `container.skills` now uses the upstream discriminated union shape end-to-end: typed provider
  options keep `anthropic -> skillId` and `custom -> providerReference`, request overlays lower
  custom provider references onto native Anthropic `skill_id`, and same-protocol request
  normalization restores the custom public shape instead of collapsing it to `skillId`
- adaptive thinking `display` now survives normalize -> overlay -> finalize request shaping
  instead of disappearing when Anthropic thinking is rebuilt
- streaming mode selection now uses the same native structured-output predicate as the request-body
  builder, so having tools no longer forces a false downgrade to the JSON-tool path on supported
  models

## Validation

This workstream is locked by:

- alias tests in `siumai-provider-anthropic/src/provider_options/anthropic/mod.rs`
- tool helper/merge tests in
  `siumai-provider-anthropic/src/providers/anthropic/ext/{request_options,tools}.rs`
- typed metadata surface tests in `siumai-protocol-anthropic/src/provider_metadata/anthropic.rs`
- non-stream fixture assertions in `siumai/tests/anthropic_messages_fixtures_alignment_test.rs`
  now also lock the audited `AnthropicMessageMetadata` web-search shape including explicit `null`
  `stopSequence` / `iterations` / `container` / `contextManagement`
- stream-end fixture assertions in
  `siumai/tests/anthropic_messages_stream_fixtures_alignment_test.rs` now lock that same typed
  shape on the public streaming finish surface
- protocol tool-conversion tests in
  `siumai-protocol-anthropic/src/standards/anthropic/utils/tools.rs`
- stable public compile guards in `siumai/tests/public_surface_imports_test.rs`
- package-helper tests in
  `siumai-provider-anthropic/src/providers/anthropic/prepare_step.rs`
- request-option/body-overlay tests in
  `siumai-protocol-anthropic/src/standards/anthropic/request_options.rs`
  now lock merged `output_config.{format,effort,task_budget}`
- transformer tests in
  `siumai-protocol-anthropic/src/standards/anthropic/transformers.rs`
  now pin native structured output on `output_config.format`
- provider-spec tests in `siumai-provider-anthropic/src/providers/anthropic/spec.rs`
  now lock task-budget beta-header inference
- typed-option serde tests in
  `siumai-provider-anthropic/src/provider_options/anthropic/mod.rs` now lock `xhigh` effort,
  adaptive thinking `display`, and `inferenceGeo`
- bridge/public-path tests in `siumai/src/experimental_bridge/request/tests.rs` and
  `siumai/tests/provider_public_path_parity_test.rs` now pin `output_config.format` round-trips
  plus the `container.skills` custom `providerReference` restoration path
- the `anthropic-code-execution-20250825.2` fixture assertion in
  `siumai/tests/anthropic_messages_fixtures_alignment_test.rs` now also locks typed container
  metadata instead of an outdated `container: null` expectation

## Remaining follow-up

- Re-check if other Anthropic package-index exports need stable facade mirrors after the next AI SDK
  update.
- Re-audit the next upstream Anthropic request changelog for additional `output_config` fields so
  Siumai keeps one merged native output-config lane instead of reintroducing parallel overlays.
- Re-audit whether upstream Anthropic adds further adaptive-thinking or inference policy enums so
  the Rust typed surface does not fall back behind the audited union shapes again.
