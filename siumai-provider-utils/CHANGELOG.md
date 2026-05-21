# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added the new `siumai-provider-utils` crate as the canonical home for AI SDK-style
  provider/protocol helper behavior.
- Initial helper coverage includes builder defaults, chat request normalization, data/base64
  helpers, downloads, error-message extraction, headers, IDs, JSON instruction/parse helpers, MIME
  detection, optional-value helpers, provider options/references, reasoning mapping, runtime
  metadata, serial jobs, settings, URL composition, UTF-8 decoding, runtime validation helpers, and
  `standards::ToolNameMapping`.
