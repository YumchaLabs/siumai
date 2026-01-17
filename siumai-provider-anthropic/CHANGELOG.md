# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.5] - 2026-01-15

### Added

- Native Anthropic provider extracted into its own crate as part of the workspace split.
- Files API helper and beta wiring, plus container metadata support.
- Message batches and `count_tokens` helpers.
- Computer-use provider tools (and related fixture alignment).

### Fixed

- Vercel-aligned JSON tool streaming behavior.
