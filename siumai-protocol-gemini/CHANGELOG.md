# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.7](https://github.com/YumchaLabs/siumai/compare/siumai-protocol-gemini-v0.11.0-beta.6...siumai-protocol-gemini-v0.11.0-beta.7) - 2026-05-05

### Added

- add schema-less structured output helpers
- *(google)* align package settings and streaming surfaces
- *(types)* align shared ai sdk type surface
- *(provider)* align package surfaces across providers
- refactor
- *(alignment)* align image and media input contracts
- *(streaming)* align gemini stable parts and extras consumers

### Fixed

- align provider streaming bridges
- *(ci)* align response fixtures and clippy checks
- *(ci)* satisfy clippy feature matrix
- *(text)* retain provider request metadata
- *(completion)* preserve response bodies
- *(image)* include direct response envelopes
- *(chat)* preserve provider response bodies
- *(google)* preserve embedding response body
- *(google)* align image provider metadata
- *(files)* align provider metadata
- *(gemini)* align finish provider metadata shape
- *(gemini)* preserve usage token details
- *(provider)* align tool denial fallback text
- *(gemini)* preserve raw finish reasons
- *(streaming)* allow lossy gemini fallback for part events
- *(parity)* align perplexity metadata and gemini reasoning streams

### Other

- add beta 7 migration guidance
- prepare beta release notes
- rename typed stream overlay
- update stream examples for typed events
- remove legacy chat stream events
- stop emitting legacy stream deltas
- make gemini stream serializer typed-only
- emit typed gemini stream parts

### Fixed

- Gemini embedding response transformation now preserves an AI SDK-style response envelope with
  the raw response body on direct transformer usage.
- Gemini / Vertex reasoning streams now emit AI SDK-style custom `reasoning-start`,
  `reasoning-delta`, and `reasoning-end` events alongside stable runtime reasoning parts,
  preserving `providerMetadata.{google|vertex}.thoughtSignature` on the public stream surface.
- Gemini GenerateContent stream serialization now suppresses duplicate reasoning deltas when a
  bridge feeds the serializer both the typed `Part` lane and the mirrored Gemini custom
  `reasoning-delta` event for the same chunk.
- Gemini prompt/response conversion now honors the newer `FilePartSource` split consistently:
  image/file request conversion rejects provider-managed file references explicitly instead of
  mixing them with `MediaSource`, and response parsing now rebuilds image/file parts through
  `FilePartSource::{base64,url}` so `google` feature builds are green again after the
  provider-reference refactor.
- Gemini JSON response grounding metadata synthesis now accepts document-style unified `source` parts instead of assuming URL-only sources.
- Gemini tool-result conversion now handles explicit `image-data` content with preserved media types and stringifies unsupported explicit file/url/id variants predictably.
- Gemini JSON response usage replay now derives prompt/cache/text/reasoning totals from the normalized AI SDK-style usage model and preserves `trafficType` instead of dropping it at the type boundary.
- Gemini streaming serialization now accepts the new runtime `ChatStreamEvent::Part(ChatStreamPart)` semantic channel by routing it through the typed stream-part compatibility bridge, so reasoning/source/finish-style V4 parts no longer require provider-scoped custom events first.
- Gemini streaming parsing now emits first-class runtime stream parts for reasoning/source/provider-executed tool semantics and for `emit_v3_tool_call_parts=true` function-call paths instead of tunneling those AI SDK-stable semantics through `gemini:*` custom events.
- Gemini usage parsing/serialization now treats `candidatesTokenCount + thoughtsTokenCount` as total completion usage and preserves `cachedContentTokenCount` / `trafficType` across SSE round-trips without overwriting raw usage fields when stable totals are absent.
- Gemini request shaping now matches the upstream `google|vertex` namespace precedence fix more
  closely: request provider options and `thoughtSignature` replay use the runtime namespace first
  and fall back to the canonical sibling key only when needed.
- Gemini image request shaping now consumes canonical top-level `aspectRatio` / `seed` more
  consistently: Gemini image generation maps them into `generationConfig.imageConfig.aspectRatio`
  and `generationConfig.seed`, while Imagen request shaping now also honors provider-owned
  `aspectRatio` overrides with AI SDK-aligned precedence.
- Gemini GenerateContent SSE now also preserves AI SDK-style first-chunk parse-failure lifecycle
  ordering: invalid JSON on the first SSE payload emits `stream-start` before the parse error
  instead of skipping the stream lifecycle start entirely.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Gemini / GenerateContent protocol mapping split out into a dedicated crate.
- GenerateContent SSE stream serialization helpers.

### Fixed

- Vercel-aligned tool call parsing/serialization and official endpoint shaping.
