# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.7](https://github.com/YumchaLabs/siumai/compare/siumai-protocol-openai-v0.11.0-beta.6...siumai-protocol-openai-v0.11.0-beta.7) - 2026-05-05

### Added

- *(openai-compatible)* add alibaba qwen options parity
- add AI SDK rerank result views
- add schema-less structured output helpers
- *(ai-sdk)* align shared structural surfaces and builder helpers
- *(prompt)* align shared prompt and data content surfaces
- *(types)* align shared ai sdk type surface
- *(speech)* align shared tts request options
- refactor
- *(transcription)* require media type for audio inputs
- *(audio)* align transcription input shape with ai sdk
- *(alignment)* align image and media input contracts
- *(openai-compatible)* support query params and structured outputs policy
- *(openai-compatible)* align request settings surface
- *(core)* align ai sdk v4 stream parts and openai metadata surfaces
- *(media)* align provider-owned image and video surfaces with ai sdk

### Fixed

- align provider streaming bridges
- *(ci)* align response fixtures and clippy checks
- *(ci)* satisfy clippy feature matrix
- *(text)* retain provider request metadata
- *(completion)* preserve response bodies
- *(openai)* include image response envelope
- *(image)* include direct response envelopes
- *(openai)* preserve responses api body
- *(openai)* preserve embedding response body
- *(openai)* preserve rerank response body
- *(text)* preserve raw response envelopes
- *(rerank)* preserve raw response envelopes
- *(files)* align provider metadata
- *(openai)* align image provider metadata
- *(openai)* emit code interpreter call on code done
- *(openai)* align tool search stream lifecycle
- *(openai)* omit apply patch stream provider flag
- *(openai)* preserve apply patch operation shape
- *(openai)* align non-stream hosted tool outputs
- *(openai)* align shell provider execution flag
- *(openai)* align mcp stream lifecycle
- *(openai)* align local shell stream input
- *(openai)* align web search stream results
- *(openai)* escape streamed tool input deltas
- *(openai)* align file search stream results
- *(openai)* align computer use stream parts
- *(openai)* stream image generation partial results
- *(openai)* handle reasoning summary done events
- *(openai)* synthesize hosted dynamic response items
- *(openai)* replay apply patch calls
- *(openai)* preserve tool search response items
- *(openai)* align responses provider tools
- *(provider)* align openai-compatible tool preparation
- *(provider)* align openai compatible chat conversion
- *(deepseek)* align chat message conversion
- *(xai)* default missing responses usage to zero
- *(xai)* surface responses cost metadata
- *(xai)* request streaming usage by default
- *(openai-compatible)* align provider chat settings
- *(openai-compatible)* align alibaba chat settings
- *(openai-compatible)* align perplexity chat settings
- *(openai-compatible)* align perplexity response format
- *(openai-compatible)* inject mistral json instruction
- *(openai-compatible)* enable qwen structured outputs by default
- *(togetherai)* align image requests with AI SDK
- *(xai)* align reasoning defaults with AI SDK
- *(openai-compatible)* align deepseek thinking options
- *(openai-compatible)* add alibaba prompt cache control
- *(openai-compatible)* align vendor finish and qwen usage
- *(openai)* replay responses incomplete reasons
- *(openai-compatible)* align provider usage semantics
- align file upload provider option defaults
- *(openai-compat)* align deepinfra paths and stream terminal parity
- *(openai-compatible)* align image provider options and warnings
- *(openai-compatible)* map known chat provider options
- *(openai-compatible)* align structured outputs default policy

### Other

- add beta 7 migration guidance
- prepare beta release notes
- rename typed stream overlay
- update stream examples for typed events
- remove legacy chat stream events
- stop emitting legacy stream deltas
- make openai responses stream typed-only
- *(transcription)* move stt provider knobs to provider options
- *(openai)* remove final responses request metadata fallback

### Fixed

- OpenAI-compatible rerank response transformation now preserves an AI SDK-style response envelope
  with the raw response body before adapter normalization, so direct transformer usage keeps the
  same debugging metadata as the HTTP executor path.
- OpenAI and OpenAI-compatible embedding response transformation now also preserves an AI
  SDK-style response envelope with the raw response body on direct transformer usage.
- OpenAI-compatible Alibaba/Qwen chat request shaping now applies AI SDK-style prompt cache
  markers from message/part `providerOptions.alibaba|qwen.cacheControl` as Alibaba
  `cache_control` content-part fields, while keeping request-level `cacheControl` out of the
  top-level body.
- OpenAI Responses request conversion now maps prompt-side provider-owned user `image` / `file`
  references directly onto `input_image.file_id` / `input_file.file_id`, and request
  normalization converts incoming wire `file_id` items back into canonical stable
  `providerReference` parts without needing bridge-emitted `fileIdPrefixes`.
- OpenAI Responses and OpenAI-compatible usage parsing/serialization now converge on the shared AI SDK-style `inputTokens` / `outputTokens` / `raw` model, preserving provider-native `raw` usage plus `input_tokens_details.cached_tokens` and `output_tokens_details.reasoning_tokens` during JSON and SSE replay.
- OpenAI Responses request conversion now forwards `tool-approval-response.reason` on MCP approval items instead of dropping it.
- OpenAI Image request shaping now supports adapter-owned provider-options lookup hooks, so OpenAI-compatible image generation/edit/variation can merge provider-owned request fields from deprecated `openai-compatible`, canonical `openaiCompatible`, and provider-owned keys instead of hardcoding `providerOptions.openai|azure`.
- OpenAI and OpenAI-compatible image specs now surface AI SDK-style unsupported `aspectRatio` / `seed`
  warnings across generation, edit, and variation requests instead of only covering the generation
  seed case.
- OpenAI image variation multipart shaping now consumes the shared typed variation image input and
  explicitly rejects URL-backed variation inputs on the multipart path instead of assuming raw
  bytes only.
- OpenAI-compatible chat request/response shaping now also follows the audited AI SDK raw/camelCase
  provider-key contract more closely: provider-owned passthrough options merge raw + camelCase
  keys with camelCase taking precedence, non-stream and stream-finish provider metadata keep the
  resolved request-side namespace key with an explicit provider root, and Gemini-compatible
  `extra_content.google.thought_signature` now survives on compat tool calls as
  `providerMetadata.{provider}.thoughtSignature`.
- OpenAI typed metadata helpers now also expose keyed accessors for the shared provider-root
  contract (`*_metadata_with_key(...)`), bringing the OpenAI helper surface in line with the
  existing keyed Anthropic/Azure/DeepSeek patterns.
- OpenAI Responses request conversion now uses stable message/part/tool-result `providerOptions` as the canonical request-time lane for item ids, reasoning payloads, and image detail; assistant tool-call ids no longer read request-side `providerMetadata.openai`.
- OpenAI Responses request conversion now matches the stricter canonical provider boundary more closely: reasoning and compaction items no longer read request-side `provider_metadata`, encrypted reasoning without `itemId` is forwarded as a first-class reasoning item, tool-result approval-id skipping reads only output `providerOptions.openai`, image detail reads only part/tool-result `providerOptions`, and assistant tool-call ids now also stay on canonical `providerOptions`.
- OpenAI Responses request normalization now writes request-side `itemId`, `reasoningEncryptedContent`, and `imageDetail` back into canonical `providerOptions.openai` slots instead of response-style `provider_metadata.openai`, and `input_image` normalization now restores AI-SDK-shaped user image file parts rather than collapsing them into the older image-only shape.
- OpenAI Responses tool-result conversion now preserves explicit `file-data` / `file-url` / `file-id` / `image-data` / `image-url` / `image-file-id` shapes instead of collapsing them through a coarse image/file union.
- OpenAI Responses response decoding now also writes stable `raw_finish_reason` from
  `incomplete_details.reason`, so downstream response/extras layers can observe the provider-native
  raw finish cause without reparsing the raw payload.
- OpenAI-compatible chat response decoding now also preserves provider-native legacy
  `raw_finish_reason = "function_call"` while still normalizing the stable finish reason to
  `tool_calls`.
- OpenAI Responses request fixtures now lock native `file_id` roundtrips for tool-result `image-file-id` / `file-id`, and provider-keyed `ToolResultFileId` inputs now have regression coverage proving OpenAI-native ids win when projecting to Responses input items.
- OpenAI Responses fixture baselines now match the stable canonical model instead of older compatibility shapes: tool-result attachments no longer use the removed generic `file` variant, unsupported settings are asserted via `unsupported { feature }`, and exact response roundtrips now pin `Usage.inputTokens` / `Usage.outputTokens` / `Usage.raw`.
- OpenAI Responses exact request/response alignment now also preserves the audited stable tool
  structure more closely: provider-executed tool calls/results keep stable `dynamic` plus
  tool-result `input`, and hosted dynamic `local_shell` / `shell` / `apply_patch` items now
  serialize back to native Responses tool item types on the response bridge path instead of
  degrading to generic function/custom tool calls.
- OpenAI Responses SSE serialization now accepts the new runtime `ChatStreamEvent::Part(ChatStreamPart)` semantic channel directly by routing it through the typed stream-part compatibility bridge instead of requiring provider-scoped custom events first, and it now normalizes those parts before locking serializer state so direct part replay no longer self-deadlocks.
- OpenAI Responses SSE parsing now emits first-class runtime stream parts for `stream-start`, `response-metadata`, non-tool `text-*`, `reasoning-*`, `source`, successful `finish`, and provider-hosted tool / MCP / approval semantics, and document sources can now reserialize back to Responses annotations via `providerMetadata.openai.fileId` even when the stable source shape no longer carries a top-level document URL.
- OpenAI Responses same-protocol replay of provider-hosted tool / MCP / approval items now uses a dedicated runtime replay carrier for `rawItem` / `outputIndex`, so parser output, bridge output, and SSE serialization no longer depend on loose provider-scoped custom JSON extras for those hints.
- OpenAI Responses failed/unknown finish replay now preserves `null` usage totals end-to-end instead of materializing zero counts inside buffered terminal events.
- OpenAI audio multipart shaping now consumes the canonical shared transcription audio input,
  including base64-backed request payloads, instead of depending on the removed stable
  `audio_data | file_path` split.
- OpenAI STT multipart shaping now also requires and forwards the stable transcription
  `mediaType`, matching the AI SDK required-input contract instead of treating MIME attachment as
  optional.

## [0.11.0-beta.5] - 2026-01-15

### Added

- OpenAI(-like) protocol mapping split out into a dedicated crate.
- OpenAI Responses SSE stream serialization helpers (gateway/proxy use-cases).
- Protocol-level JSON response encoders for transcoding.

### Fixed

- Vercel-aligned parsing/serialization for Responses API stream parts and fixtures.
