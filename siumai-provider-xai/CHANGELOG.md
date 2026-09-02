# Changelog

## Unreleased

## [0.11.0-beta.11](https://github.com/YumchaLabs/siumai/compare/siumai-provider-xai-v0.11.0-beta.10...siumai-provider-xai-v0.11.0-beta.11) - 2026-09-02

### Added

- *(facade)* expose direct transport configuration
- *(providers)* add Gemini multimodal embedding and Cohere transcription

### Fixed

- *(review)* harden facade convergence contracts

### Other

- *(facade)* teach the typed Siumai hierarchy
- *(facade)* document unified family calls (U6)
- close trusted CONNECT milestone
- close direct transport infrastructure milestone
- *(providers)* finish HTTP transport settings migration

## [0.11.0-beta.10](https://github.com/YumchaLabs/siumai/compare/siumai-provider-xai-v0.11.0-beta.9...siumai-provider-xai-v0.11.0-beta.10) - 2026-08-13

### Added

- *(siumai)* expose OpenAI Responses WebSocket surface
- *(provider)* [**breaking**] deepen xAI and Groq media surfaces
- *(provider)* [**breaking**] expand Deepgram and ElevenLabs audio families
- *(provider)* [**breaking**] deepen MiniMax portable and voice surfaces
- *(provider)* [**breaking**] deepen Kimi and ARK surfaces
- *(provider)* [**breaking**] expand flagship and Chinese surfaces
- *(openai)* [**breaking**] add portable families and native resources
- *(gemini)* [**breaking**] add provider-owned product surfaces
- *(gemini)* [**breaking**] add faithful Interactions language
- *(core)* [**breaking**] enforce typed stream settlement
- *(core)* [**breaking**] enforce canonical language replay semantics
- *(minimax)* [**breaking**] canonicalize provider identity and protocols
- *(openai)* add provider-faithful Responses and Realtime runtime

### Fixed

- *(openai)* canonicalize partial responses terminals

### Other

- narrow Anthropic Skills support claims
- publish flagship provider journeys
- close validation ownership migration
- *(language)* [**breaking**] unify terminal outcomes and settlement
- *(core)* [**breaking**] remove runtime model policy
- *(core)* [**breaking**] bind provider options to configured instances
- *(openai)* make Responses settlement dialect-owned
- *(openai)* move prompt cache intent to content annotations
- *(gemini)* [**breaking**] establish product-level provider
- *(providers)* [**breaking**] extract Moonshot and Volcengine owners
- *(openai)* [**breaking**] rename Responses protocol module
- [**breaking**] rebuild Siumai around provider-faithful family contracts
- establish Siumai Next architecture baseline

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
