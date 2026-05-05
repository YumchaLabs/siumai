# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.7](https://github.com/YumchaLabs/siumai/compare/siumai-spec-v0.11.0-beta.6...siumai-spec-v0.11.0-beta.7) - 2026-05-05

### Added

- add AI SDK V4 prompt projection
- add AI SDK V4 generate result overlays
- add AI SDK V4 call options overlay
- add AI SDK V4 model tool projections
- add AI SDK V4 tool choice projection
- align provider tool ownership shape
- add AI SDK generated data helper parity
- add AI SDK download error shape
- add AI SDK generate image prompt shape
- add AI SDK no such provider error shape
- add AI SDK telemetry options shape
- add AI SDK UI stream alias parity
- add AI SDK UI message helper parity
- add AI SDK UI part aliases
- add AI SDK UI chat passive shapes
- add AI SDK provider error carriers
- add AI SDK passive error carriers
- add AI SDK UI client option shapes
- align AI SDK UI stream options and safe validation
- add AI SDK generate video result view
- add AI SDK UI message stream chunks
- add AI SDK generate object event views
- add AI SDK rerank result views
- add AI SDK media result views
- add AI SDK language model stream views
- add AI SDK generate text step control views
- add AI SDK generate text callback event views
- add AI SDK text stream part views
- add AI SDK generate text result views
- add AI SDK generate text content union
- add AI SDK text content output views
- add AI SDK tool failure output views
- add AI SDK generated file reasoning views
- add AI SDK tool approval output views
- align AI SDK tool call result views
- add schema-less structured output helpers
- add no object generated error
- expose provider utils schema and id helpers
- expose ai sdk json schema alias
- expose ai sdk source shape
- expose ai sdk embedding alias
- expose ai sdk usage helpers
- *(ai-sdk)* align shared structural surfaces and builder helpers
- *(prompt)* align shared prompt and data content surfaces
- *(video)* align result materialization with ai sdk
- *(prompt)* support tool result file url media type
- *(prompt)* add prompt part metadata builders
- *(prompt)* align tool result provider option helpers
- *(prompt)* add single-provider option helpers
- *(prompt)* add provider option builders
- *(prompt)* add tool approval response builders
- *(spec)* expose provider-defined tool provider options
- *(tooling)* align provider utils runtime helpers
- *(types)* add provider utils shared type aliases
- *(types)* add shared prompt model message surface
- *(types)* add prompt call settings compatibility
- *(video)* add ai sdk response metadata views
- *(types)* add shared request options surface
- *(types)* add shared language model call options
- *(types)* align shared ai sdk type surface
- *(audio)* preserve request metadata on audio results
- *(speech)* align shared tts request options
- *(audio)* align speech and transcription helper results
- *(media)* align helper empty-result semantics
- *(video)* refine empty-result error semantics
- *(video)* align task-based video model surface
- refactor
- *(transcription)* require media type for audio inputs
- *(audio)* align transcription input shape with ai sdk
- *(alignment)* align image and media input contracts
- *(core)* align ai sdk v4 stream parts and openai metadata surfaces
- *(media)* align provider-owned image and video surfaces with ai sdk

### Fixed

- *(ci)* align response fixtures and clippy checks
- *(ci)* satisfy clippy feature matrix
- *(text)* retain provider request metadata
- *(completion)* preserve response bodies
- *(text)* preserve raw response envelopes
- *(audio)* preserve raw JSON response bodies
- *(rerank)* preserve raw response envelopes
- *(openai)* align responses provider tools
- *(completion)* preserve raw finish reasons
- tighten AI SDK V4 provider metadata
- tighten AI SDK V4 prompt provider options
- validate AI SDK V4 custom kinds
- align AI SDK call warning shape
- canonicalize AI SDK V4 tool result outputs
- tighten AI SDK V4 tool object overlays
- tighten AI SDK V4 generated content overlays
- align AI SDK V4 numeric usage overlays
- align V4 function tool input examples
- preserve V4 approval response provider options
- align AI SDK tool repair error shapes
- align AI SDK tool view discriminators
- align finish reason serde values
- align tool choice serde shape
- *(types)* enforce prompt discriminator invariants
- *(openai)* align provider references and usage parsing
- *(streaming)* restore textual delta compatibility

### Other

- add beta 7 migration guidance
- prepare beta release notes
- update stream examples for typed events
- remove legacy chat stream events
- stop emitting legacy stream deltas
- start typed-only stream migration
- *(ai-sdk)* cover response body metadata surfaces
- *(transcription)* move stt provider knobs to provider options
- *(minimaxi)* move video vendor fields behind provider options

### Added

- Add shared `ProviderMetadataMap` helpers aligned with AI SDK `ProviderMetadata`:
  `provider_metadata_object`, `provider_metadata_value`, `provider_metadata_from_object`, and
  `merge_provider_metadata` now centralize the stable `provider_id -> object` contract used by
  response-side provider metadata.
- Add shared Groq provider-tool helpers aligned with AI SDK `browserSearch()`:
  `tools::groq::browser_search()` and `provider_defined_tool("groq.browser_search")` now build the
  stable `Tool::ProviderDefined` shape with the canonical `browser_search` default name.
- Add canonical `providerOptions` to shared file uploads through `FileUploadRequest`, bringing the
  stable file-upload request shape closer to AI SDK `FilesV4UploadFileCallOptions`.
- Add a stable AI SDK-aligned completion family: `CompletionRequest` now keeps a structured
  prompt plus `tools`, `tool_choice`, `common_params`, `response_format`, `provider_options_map`,
  and per-request `HttpConfig`, while `CompletionResponse` carries generated text, finish reason,
  usage, warnings, response metadata, and provider metadata.
- Add the Vercel-aligned `Warning::Compatibility` variant and helper constructor.
- Add the AI SDK-style `Warning::Unsupported { feature, details }` variant and normalize helper constructors to emit that shared shape while keeping legacy unsupported warning variants for compatibility.
- `Usage` now makes AI SDK-style `inputTokens` / `outputTokens` / `raw` the canonical stable layer, while legacy `prompt/completion/total` counts remain available through compatibility accessors/serde plus normalized helper APIs for migration-safe provider/protocol code.
- Extend unified `source` parts with optional `mediaType`, `filename`, and `providerMetadata` fields for document-style sources.
- Refactor unified `source` parts into a stricter AI SDK-style URL/document union via `SourcePart`, while preserving `sourceType`-based wire serialization and compatibility decoding.
- Extend unified `tool-approval-request` / `tool-approval-response` parts with request `providerMetadata` and response `reason`.
- Add a first-class AI SDK-style UI-message type layer:
  `UiMessage`, `UiMessagePart`, `UiDataPart`, and `UiToolPart` now model the static
  `UIMessage` / `convertToModelMessages` boundary in the shared type crate.
- Extend unified `tool-approval-response` parts with optional `providerExecuted`, preserving the
  AI SDK approval-response routing hint on the stable content surface.
- Add first-class `providerOptions` to `ChatMessage`, request-capable `ContentPart` variants, and tool-result output/content shapes, including builder/helper APIs for stable mutation.
- Add first-class V4 `custom` / `reasoning-file` content parts plus explicit tool-result content variants (`file-data`, `file-url`, `file-id`, `image-data`, `image-url`, `image-file-id`) and a stable provider-keyed `ToolResultFileId`.
- Add first-class prompt-side provider-owned file/image references through shared
  `ProviderReference` plus `FilePartSource`, including `ContentPart::{image,file}_provider_reference(...)`
  and `ChatMessageBuilder::{with_image,with_file}_provider_reference(...)`.
- Add first-class runtime `ChatStreamPart` semantics and `ChatStreamEvent::Part`, so the stable streaming surface can carry AI SDK V4 stream-part concepts such as `source`, `response-metadata`, `stream-start warnings`, `finish`, `custom`, `file`, and `reasoning-file`.
- Add runtime-only `ChatStreamReplay` plus `ChatStreamEvent::PartWithReplay`, so protocol serializers can carry same-protocol replay hints such as OpenAI Responses `rawItem` / `outputIndex` without widening `ChatStreamPart` or overloading generic `providerMetadata`.
- Add first-class `ProviderType::{Azure, Cohere, TogetherAi, Bedrock}` variants so built-in native
  provider identity no longer needs to degrade those audited AI SDK-style providers to
  `Custom(...)`.
- Add first-class `ProviderType::{Mistral, Fireworks, Perplexity}` variants so the next
  AI SDK-packaged OpenAI-compatible provider ids no longer need to degrade to `Custom(...)` at the
  stable typing layer.
- Extend the shared image request family with top-level `aspectRatio` across generation/edit/variation
  plus shared `seed` on edit/variation, including builder/helper methods for the new canonical
  image call-option fields.
- Refactor `ImageVariationRequest` to carry a typed `ImageEditInput` file/url source image instead
  of a raw byte-only field, bringing the shared variation surface closer to AI SDK
  `ImageModelV4File`.
- Extend typed `ImageEditInput` file/url inputs with first-class per-input `providerOptions`,
  further aligning the shared image file shape with AI SDK `ImageModelV4File`.
- Extend typed `VideoGenerationInput` file/url inputs with first-class per-input
  `providerOptions`, aligning the shared video file shape more closely with AI SDK
  `VideoModelV4File`.
- Add canonical `AudioInputData` plus helper constructors/accessors, and refactor shared
  `SttRequest` / `AudioTranslationRequest` typing onto the AI SDK-style
  `audio + mediaType + providerOptions` surface instead of the older `audio_data | file_path`
  split.
- Add `ProviderType::DeepInfra`, so stable/provider catalog layers can model DeepInfra as a
  first-class provider instead of falling back to `Custom("deepinfra")`.
- Add `ProviderType::Vertex` and `ProviderType::AnthropicVertex`, extending the stable provider
  identity layer for the broader Google Vertex wrapper family alongside `VertexMaas`.
- `ChatResponse` now exposes optional stable `raw_finish_reason`, giving provider/protocol/extras
  layers one canonical slot for AI SDK-style raw finish metadata.

### Fixed

- `ChatResponse::reasoning()` and `ChatMessage::reasoning()` now ignore empty/whitespace reasoning
  parts, so metadata-only placeholders such as Anthropic `redacted_thinking` blocks no longer
  surface as fake empty reasoning strings through the stable helper API.
- Shared response-side provider metadata now reuses the same `ProviderMetadataMap` root across
  `ChatResponse`, `CompletionResponse`, stable content parts, stream metadata, and skill-upload
  results instead of each lane carrying a different nested-map convention. UI `providerMetadata`
  intentionally remains on `ProviderOptionsMap` because AI SDK `convertToModelMessages()` treats
  it as request-time `providerOptions`.
- Tool-result content parsing now accepts the newer AI SDK provider-reference aliases
  `file-reference` / `image-file-reference` plus `providerReference` payload keys alongside the
  existing `file-id` / `image-file-id` compatibility shape.
- Prompt-side `ContentPart::Audio.mediaType` and `ContentPart::File.mediaType` now serialize on
  the canonical AI SDK-style `mediaType` field while still accepting the legacy `media_type`
  alias during decode.
- `Usage` now tracks whether legacy totals are actually known, so builders, merges, serde, and normalized helpers stop materializing unknown compatibility totals as `0` when only partial AI SDK-style usage data is available.
- `Usage` no longer exposes legacy `prompt/completion/total` counts as public storage fields, so new code must use builders/constructors and compatibility accessors instead of struct literals or direct field reads.
- `Usage.merge()` now follows AI SDK `addLanguageModelUsage()` semantics more closely: token/accounting fields are aggregated, but provider-native `raw` usage is cleared on merge instead of being recursively combined into a synthetic payload.
- `ProviderOptionsMap` serde now normalizes provider ids during JSON decode and re-emits the canonical `openaiCompatible` wire key during encode, so JSON request fixtures and builder-authored requests share the same lookup behavior.
- `ResponseMetadata` now serializes the stable AI SDK field names `modelId` / `timestamp` while continuing to accept legacy `model` / `created` aliases during decode.
- Shared `SttRequest` / `AudioTranslationRequest` now require `mediaType` directly on the stable
  struct and constructor surface, so `from_audio(...)` / `from_base64(...)` align with the AI SDK
  required transcription input contract instead of leaving media type optional.

## [0.11.0-beta.6] - 2026-03-02

### Added

- Support runtime-only per-request HTTP overrides (headers + timeout) used by the facade family call options.

