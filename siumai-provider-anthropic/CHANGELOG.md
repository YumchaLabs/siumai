# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Native Anthropic provider now also exposes a provider-owned `skills()` resource aligned with the
  AI SDK `AnthropicSkills` surface: multipart/base64 helpers upload to `POST /v1/skills`, the
  skill client automatically enables `skills-2025-10-02`, and canonical `providerReference` /
  `providerMetadata` are returned from the stable result shape.

### Fixed

- Native Anthropic provider metadata/spec wiring now advertises the Files API beta requirement for
  prompt-side provider-owned file references, keeping the public provider surface aligned with the
  audited request converter behavior.
- Anthropic `skills()` upload now follows the audited AI SDK metadata path more closely: when the
  create response includes `latest_version`, the client also fetches
  `GET /v1/skills/{id}/versions/{version}` so version-scoped `name` / `description` win over the
  create-response copies.
- Anthropic thinking replay helper now maps response-side reasoning metadata onto next-turn `providerOptions.anthropic.signature` / `redactedData` on reasoning parts instead of writing legacy message-level `metadata.custom["anthropic_*"]` shims.
- Anthropic request-side typed options now align much more closely with
  `anthropic-messages-options.ts`: the public builder/config/facade surface covers typed
  `thinking`, `sendReasoning`, `disableParallelToolUse`, `cacheControl`, `metadata.userId`,
  `mcpServers`, `contextManagement`, `toolStreaming`, `effort`, `speed`, and `anthropicBeta`,
  and `container.skills` now uses a typed `AnthropicContainerSkillType` plus upstream `skillId`
  camelCase deserialization support.
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
