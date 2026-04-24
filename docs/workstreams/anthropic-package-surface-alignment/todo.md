# Anthropic Package Surface Alignment - TODO

Last updated: 2026-04-22

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Anthropic package export audit

- [x] Audit `repo-ref/ai/packages/anthropic/src/index.ts`.
- [x] Audit the Bedrock cross-package `AnthropicProviderOptions` re-export in
  `repo-ref/ai/packages/amazon-bedrock/src/index.ts`.

## Track B - Provider-owned/public alias parity

- [x] Add `AnthropicLanguageModelOptions` on the Anthropic provider-owned/public Rust surface.
- [x] Add `AnthropicProviderSettings` plus package `VERSION` on the Anthropic provider-owned/public
  Rust surface.
- [x] Add deprecated `AnthropicProviderOptions` migration coverage.
- [x] Add `AnthropicMessageMetadata` on the typed Anthropic metadata surface as a dedicated narrow
  struct instead of a thin alias to the wider Rust helper metadata.
- [x] Narrow `AnthropicMessageMetadata.container` / `skills` onto dedicated required-field message
  metadata structs instead of reusing the wider helper container shape.
- [x] Lock `AnthropicMessageMetadata` on audited non-stream and `StreamEnd` fixtures so the AI SDK
  null-preserving shape is covered beyond unit-only roundtrip tests.
- [x] Re-export `AnthropicUsageIteration` on the stable public facade.
- [x] Add typed `AnthropicToolOptions` plus a Rust helper for function-tool provider options.
- [x] Make Anthropic request/tool helper writes merge onto existing `providerOptions.anthropic`
  objects instead of overwriting sibling fields.
- [x] Forward `eagerInputStreaming` through the Anthropic protocol and bridge paths.

## Track C - Bedrock cross-package parity

- [x] Re-export `AnthropicProviderOptions` from `provider_ext::bedrock` when both `bedrock` and
  `anthropic` features are enabled.
- [-] Add a hard provider-crate dependency from Bedrock onto Anthropic just to mirror a TypeScript
  package edge.
  - rejected intentionally because the Cargo feature boundary should stay decoupled

## Track D - Docs and changelog

- [x] Add a dedicated Anthropic workstream under `docs/workstreams/`.
- [x] Update the structural-alignment matrix with the new Anthropic package-surface status.
- [x] Update `CHANGELOG.md` under `Unreleased`.
- [x] Export the package-level Anthropic container carry-forward helper equivalent to upstream
  `forwardAnthropicContainerIdFromLastStep(...)`.

## Track E - Remaining follow-up

- [ ] Re-audit the Anthropic package surface after the next upstream AI SDK provider-index update.

## Track F - Native structured output and task budget follow-on

- [x] Re-audit the current
  `repo-ref/ai/packages/anthropic/src/anthropic-messages-language-model.ts` request path after the
  upstream `output_config.format` migration.
- [x] Move native JSON Schema structured output from deprecated `output_format` to
  `output_config.format`.
- [x] Merge `output_config.format`, `output_config.effort`, and `output_config.task_budget`
  through one overlay path so Anthropic native output fields do not overwrite each other.
- [x] Add typed `taskBudget` support on the Anthropic provider-owned/public surface plus fluent
  builder/config helpers.
- [x] Add typed `inferenceGeo` support on the Anthropic provider-owned/public surface plus shared
  `SiumaiBuilder` helpers.
- [x] Restore the full audited `effort` enum by adding `xhigh` to the typed Anthropic surface.
- [x] Preserve adaptive thinking `display` through typed serde plus protocol normalize/overlay
  request shaping.
- [x] Enable the `task-budgets-2026-03-13` beta token when typed Anthropic task budget settings
  are present.
- [x] Keep request normalization / same-protocol bridge restoration compatible with both
  `output_config.format` and legacy `output_format`.
- [x] Align stream-time native structured-output mode selection with the final request-body
  semantics even when tools are present.
- [x] Restore the audited `container.skills` public split so typed/provider/bridge paths preserve
  `anthropic -> skillId` and `custom -> providerReference`.
