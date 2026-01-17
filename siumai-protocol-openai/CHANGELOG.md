# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.5] - 2026-01-15

### Added

- OpenAI(-like) protocol mapping split out into a dedicated crate.
- OpenAI Responses SSE stream serialization helpers (gateway/proxy use-cases).
- Protocol-level JSON response encoders for transcoding.

### Fixed

- Vercel-aligned parsing/serialization for Responses API stream parts and fixtures.
