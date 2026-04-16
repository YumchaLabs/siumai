# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Ollama now exposes a provider-owned curated `models` surface for `chat` plus `embedding`, and
  the runtime model-listing implementation has moved to `model_listing.rs` so `models.rs` can mean
  public model constants consistently again.

### Fixed

- Ollama message/tool-result conversion and streaming event serialization now compile against the
  AI-SDK-aligned `provider_options` fields and newer stable stream event variants.
- Ollama streaming now also preserves first-chunk parse-failure lifecycle ordering: invalid JSONL
  chunks emit `stream-start` before the parse error instead of skipping the stream lifecycle start.
- Ollama default-model reporting and provider-catalog output now reuse the provider-owned curated
  model source instead of a second handwritten list.

## [0.11.0-beta.5] - 2026-01-15

### Fixed

- Chat tool and image request shaping aligned with Ollama expectations.

### Changed

- Provider extracted into its own crate as part of the workspace split.
