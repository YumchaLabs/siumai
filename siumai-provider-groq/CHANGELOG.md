# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Groq transcription request coverage now follows the canonical shared `audio` input surface after
  the AI SDK-style STT request-shape alignment.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Groq provider extracted into its own crate as part of the workspace split.

### Changed

- Fixture parity and transport wiring aligned with the split architecture.
