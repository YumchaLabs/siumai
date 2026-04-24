# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Native Bedrock provider now also exposes package-level `AmazonBedrockProviderSettings` plus
  `VERSION` on the provider-owned/public Rust surface. The new settings carrier keeps provider
  construction model-agnostic (`into_builder()`, `into_builder_for_model(...)`,
  `into_config_for_model(...)`), and the underlying builder/config surfaces now also expose honest
  header helpers (`headers(...)`, `header(...)`, `with_headers(...)`, `with_header(...)`) so the
  audited supported subset of AI SDK provider settings no longer requires raw HTTP-config
  mutation.

### Fixed

- Native Bedrock prompt conversion now rejects prompt-side provider-owned user file/image
  references explicitly on the request path, matching the audited AI SDK Bedrock limitation
  instead of leaving those inputs to ambiguous fallback handling.
- Native Bedrock prompt-conversion regression coverage now also locks URL-backed user `file`
  parts as unsupported on the request path, matching the upstream AI SDK Bedrock converter's
  no-URL policy for Converse documents/images.
- Native Bedrock chat responses and streamed terminal `ChatResponse` values now preserve the raw
  Bedrock `stopReason` on stable `raw_finish_reason` instead of dropping it after normalization.
- Native Bedrock Converse JSON streaming now emits an AI SDK-style first-chunk preamble on the
  stable stream lane: `stream-start`, stable `stream-start`, stable `response-metadata`,
  runtime-opt-in `raw`, then the Bedrock content/tool deltas.
- Native Bedrock Converse JSON streaming now also preserves first-chunk parse-failure lifecycle
  ordering more completely: invalid JSON chunks emit the stable preamble before optional runtime
  `raw` and the parse error, later valid chunks do not duplicate the preamble, and Bedrock
  provider error envelopes now surface stable `error` parts.
- Native Bedrock streamed terminal `ChatResponse` values now also preserve the default model,
  request warnings, and Bedrock `stopSequence` metadata on the stream path instead of dropping
  them during JSON-stream accumulation.
- Native Bedrock Converse JSON streaming now also emits stable AI SDK-style body semantics instead
  of staying legacy-delta-only: text, reasoning, tool-input, tool-call, and terminal finish
  parts are emitted on the main stream lane while the older `ContentDelta` / `ThinkingDelta` /
  `ToolCallDelta` / `StreamEnd` compatibility events remain available.
- Native Bedrock finish parts and non-stream chat responses now preserve more of the audited AI
  SDK payload shape: usage keeps cache-aware `inputTokens` totals plus raw usage, finish/provider
  metadata now carries `trace`, `performanceConfig`, `serviceTier`, `cacheWriteInputTokens`,
  `cacheDetails`, `stopSequence`, and `isJsonResponseFromTool`, streamed/non-stream responses now
  retain reasoning provider metadata plus default-model identity and request warnings, and
  Mistral-model tool call ids are normalized the same way as upstream AI SDK Bedrock.
- Native Bedrock request shaping now also preserves much more of the audited AI SDK option
  surface: `BedrockChatOptions` now includes typed `reasoningConfig`, `anthropicBeta`, and
  `serviceTier` alongside unknown top-level passthrough fields; Anthropic Bedrock requests now
  derive `additionalModelResponseFieldPaths`, `thinking`, `anthropic_beta`, and native
  `output_config.format` JSON-schema routing where upstream Bedrock uses it; and
  `maxReasoningEffort` now maps onto the same provider-specific Bedrock fields as upstream
  (`output_config.effort`, flat `reasoning_effort`, or nested `reasoningConfig`).
- Native Bedrock prompt conversion now also follows the upstream AI SDK Bedrock converter much
  more closely on the request path: message-level `cachePoint` survives on leading
  system/developer plus user/tool/assistant blocks, user `file` parts map to Bedrock
  `document` / `image` blocks with typed `citations` support and upstream filename stripping,
  assistant reasoning blocks replay `signature` / `redactedData` from canonical
  `providerOptions.bedrock`, tool-result `content` now supports `text` plus `image-data`, and
  request-side Mistral tool ids are normalized for both assistant tool calls and tool results.
- Bedrock now also exposes a typed reasoning replay helper on the public provider surface:
  `BedrockContentPartExt::bedrock_reasoning_metadata()` reads typed `signature` /
  `redactedData` from `ContentPart::Reasoning.provider_metadata["bedrock"]`, and
  `assistant_message_with_reasoning_metadata(...)` carries those fields back into next-turn
  request-side `providerOptions.bedrock`.

### Changed

- The provider-owned typed option surface now also exposes AI SDK-style
  `AmazonBedrockLanguageModelOptions` / `AmazonBedrockRerankingModelOptions` plus deprecated
  `BedrockProviderOptions` / `BedrockRerankingOptions` aliases for public migration parity.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Amazon Bedrock fixture parity updates aligned with the Vercel reference.
