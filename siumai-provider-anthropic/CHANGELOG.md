# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.8](https://github.com/YumchaLabs/siumai/compare/siumai-provider-anthropic-v0.11.0-beta.7...siumai-provider-anthropic-v0.11.0-beta.8) - 2026-05-18

### Fixed

- align anthropic files capability surface

### Other

- *(release)* prepare v0.11.0-beta.8
- converge provider boundary architecture
- harden crate boundaries
- *(examples)* move extras example index
- *(examples)* tighten example guidance
- clean stale refactor docs

### Added

- Native Anthropic now exposes an AI SDK-style package settings wrapper:
  `AnthropicProviderSettings` covers the audited `apiKey` / `authToken` / `baseURL` / `headers` /
  `fetch` subset, converts into the provider-owned builder/config surfaces, and keeps upstream
  `generateId` / `name` explicitly deferred.
- Native Anthropic provider now also exposes a provider-owned `skills()` resource aligned with the
  AI SDK `AnthropicSkills` surface: multipart/base64 helpers upload to `POST /v1/skills`, the
  skill client automatically enables `skills-2025-10-02`, and canonical `providerReference` /
  `providerMetadata` are returned from the stable result shape.
- Native Anthropic request options now also expose typed task-budget support aligned with
  `anthropic-messages-options.ts`: `AnthropicTaskBudget`, `AnthropicTaskBudgetType`, and fluent
  builder/config helpers now own the `providerOptions.anthropic.taskBudget` surface directly.
- Native Anthropic typed options now also cover the remaining audited request enums more
  completely: `AnthropicInferenceGeo`, `AnthropicThinkingDisplay`, and `AnthropicEffort::Xhigh`
  are now first-class on the provider-owned/public surface.
- Native Anthropic package helpers now also expose
  `find_anthropic_container_id_from_last_step(...)` and
  `forward_anthropic_container_id_from_last_step(...)`, mirroring the upstream
  `forwardAnthropicContainerIdFromLastStep(...)` workflow for forwarding container ids across
  prepare-step history.

### Fixed

- Native Anthropic auth-token construction now maps `authToken` to `Authorization: Bearer ...`
  without forcing an empty `x-api-key`, matching the audited AI SDK alternate-auth path more
  closely.
- Native Anthropic provider metadata/spec wiring now advertises the Files API beta requirement for
  prompt-side provider-owned file references, keeping the public provider surface aligned with the
  audited request converter behavior.
- Native Anthropic spec/header inference now also matches the audited AI SDK task-budget contract:
  `taskBudget` enables the `task-budgets-2026-03-13` beta token, and structured-output capability
  checks now use the same native-output semantics as the final request-body builder instead of an
  older output-format-only heuristic.
- Native Anthropic request-option serde now preserves adaptive thinking `display` and the current
  upstream `xhigh` effort enum instead of narrowing those typed values away at the Rust facade
  boundary.
- Anthropic `skills()` upload now follows the audited AI SDK metadata path more closely: when the
  create response includes `latest_version`, the client also fetches
  `GET /v1/skills/{id}/versions/{version}` so version-scoped `name` / `description` win over the
  create-response copies.
- Anthropic thinking replay helper now maps response-side reasoning metadata onto next-turn `providerOptions.anthropic.signature` / `redactedData` on reasoning parts instead of writing legacy message-level `metadata.custom["anthropic_*"]` shims.
- Anthropic request-side typed options now align much more closely with
  `anthropic-messages-options.ts`: the public builder/config/facade surface covers typed
  `thinking`, `sendReasoning`, `disableParallelToolUse`, `cacheControl`, `metadata.userId`,
  `mcpServers`, `contextManagement`, `toolStreaming`, `effort`, `speed`, and `anthropicBeta`,
  and `container.skills` now uses a typed `AnthropicContainerSkillType` plus the audited
  discriminated union shape (`anthropic -> skillId`, `custom -> providerReference`) instead of
  flattening every entry onto one legacy struct.
- Anthropic enabled-thinking request shaping now also applies the upstream
  `max_tokens = maxOutputTokens + thinkingBudget` rule consistently, including legacy
  client-specific thinking configuration paths.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Native Anthropic provider extracted into its own crate as part of the workspace split.
- Files API helper and beta wiring, plus container metadata support.
- Message batches and `count_tokens` helpers.
- Computer-use provider tools (and related fixture alignment).

### Fixed

- Vercel-aligned JSON tool streaming behavior.
