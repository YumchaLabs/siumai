# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Responses input warning parity middleware now tolerates the expanded reasoning-part stable shape
  after the request-side `providerOptions` rollout.
- Responses input warning parity middleware now reads reasoning request state from canonical
  `providerOptions.openai`, preserves AI SDK-style warning snapshots for malformed reasoning
  provider options, and stops treating `provider_metadata` as a request-side reasoning carrier.
- Best-effort remote cancel for streaming Responses now tracks structured
  `Part(ResponseMetadata)` events in addition to the legacy `openai:response-metadata` custom
  event, so HTTP and websocket wrappers still call `POST /responses/{id}/cancel` after the
  stream-part migration.
- OpenAI transcription and audio-translation clients now consume the canonical shared `audio`
  request input directly, and no longer read `file_path` inside provider client code for
  translation or streaming transcription paths.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Native OpenAI provider extracted into its own crate as part of the workspace split.
- Vercel-aligned Responses API request/response shaping and fixtures.

### Fixed

- Preserve transcription text on SSE EOF.
- Default moderation model selection when no model is provided.
- Parsing/validation improvements for Files API and Responses stream events.
