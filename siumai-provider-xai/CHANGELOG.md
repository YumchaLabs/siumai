# Changelog

## Unreleased

### Breaking

- Replaced `XaiClient`, `XaiBuilder`, `XaiConfig`, and capability discovery with the synchronous,
  model-independent `XaiProvider` and `XaiProviderBuilder` runtime.
- Made the Responses API the default language path. Chat Completions remains available through
  `XaiProvider::chat_completions`.
- Removed raw provider-option passthrough, compatibility-config conversions, legacy public type
  aliases, and the crate-local `xai` relay feature.
- Moved video generation behind the explicit `experimental` namespace and stopped exposing
  ephemeral signed asset URLs.

### Added

- Added typed Chat Completions and Responses options, including xAI prompt-cache affinity,
  Responses hosted search/tools, reasoning, logprobs, response chaining, and an explicitly legacy
  Chat live-search compatibility boundary.
- Added provider-owned files, image generation and editing, text-to-speech, and experimental video
  resources backed by one shared authenticated transport.
- Added synchronous model handles and concrete Responses, Chat Completions, image, and speech
  Registry registrations, with Responses as the recommended language route.
- Refreshed open model-ID hints for Grok 4.5, Grok 4.3, Grok 4.20, Grok Build, Grok Imagine image,
  and Grok Imagine video families while keeping unknown future IDs callable.
