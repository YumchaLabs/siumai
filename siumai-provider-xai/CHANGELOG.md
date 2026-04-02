# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Aligned xAI Responses response and SSE semantics with the audited AI SDK boundary:
  reasoning metadata now stays under `providerMetadata.xai`, parsed responses no longer emit
  top-level provider metadata for plain text paths, and `file_search` tool response/stream outputs
  now normalize to the stable snake_case tool name plus camelCase result fields.

## [0.11.0-beta.5] - 2026-01-15

### Added

- xAI provider extracted into its own crate as part of the workspace split.

### Changed

- Fixture parity and Responses stream mapping aligned with Vercel AI SDK.
