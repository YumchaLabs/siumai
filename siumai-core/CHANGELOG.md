# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.5] - Unreleased

### Added

- Provider-agnostic core extracted from the facade crate as part of the workspace split.
- Injectable HTTP transport (custom `fetch` parity), including streaming use-cases.
- Typed V3 stream parts and cross-protocol transcoding foundations used by gateway/proxy layers.

### Fixed

- Stricter SSE JSON parsing to reduce silent drift.
