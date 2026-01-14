# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.5] - Unreleased

### Added

- Native OpenAI provider extracted into its own crate as part of the workspace split.
- Vercel-aligned Responses API request/response shaping and fixtures.

### Fixed

- Preserve transcription text on SSE EOF.
- Default moderation model selection when no model is provided.
- Parsing/validation improvements for Files API and Responses stream events.
