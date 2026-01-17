# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.5] - 2026-01-15

### Added

- Registry extracted as a standalone crate (factories + handles), aligning with the workspace split.
- Wiring for provider crates to be constructed via factories instead of hard-coded built-ins.

### Changed

- Modularized provider/factory modules to reduce cross-layer coupling.
