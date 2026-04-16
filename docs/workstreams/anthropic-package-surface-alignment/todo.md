# Anthropic Package Surface Alignment - TODO

Last updated: 2026-04-15

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
- [x] Add deprecated `AnthropicProviderOptions` migration coverage.
- [x] Add `AnthropicMessageMetadata` on the typed Anthropic metadata surface.
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

## Track E - Remaining follow-up

- [ ] Re-audit the Anthropic package surface after the next upstream AI SDK provider-index update.
