# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add a first-class completion family surface: `traits::CompletionCapability`,
  `completion::{CompletionModelV3, CompletionModel, CompletionStream, CompletionStreamHandle}`,
  and `LlmClient::as_completion_capability()` now let family-first code execute completion models
  without falling back to chat-specific abstractions.
- Add a shared skill-upload surface aligned with AI SDK `SkillsV4`: canonical
  `SkillFileContent` / `SkillUploadFile` / `SkillUploadRequest` / `SkillUploadResult` now live on
  the shared type layer, `traits::SkillsCapability` is public, and `LlmClient` exposes
  `as_skills_capability()` for provider-owned skill resources such as OpenAI and Anthropic.
- Add a shared AI SDK-style UI-message helper surface:
  `ui::{validate_ui_messages, convert_to_model_messages, convert_to_chat_request}` now converts
  stable `UiMessage` values into canonical `ChatMessage` / `ChatRequest` structures, including
  assistant `step-start` block splitting, data-part callbacks, and incomplete-tool filtering.
- OpenAI-compatible route selection now also understands `RequestType::Completion`, so compat
  providers can resolve `/completions` through the shared adapter path instead of hard-coding
  chat-only routes.
- `OpenAiCompatibleConfig` now has an explicit `auth_required` flag so generic
  OpenAI-compatible providers can honestly model AI SDK's optional `apiKey` setting for local or
  private gateways without weakening built-in hosted-provider auth validation.

### Fixed

- Core OpenAI-compatible image response transformation now accepts provider wire
  `data[].b64_json` / `data[].url` responses directly and returns an AI SDK-style response
  envelope instead of requiring callers to pre-normalize image payloads into Siumai's unified
  struct.
- Shared stream factories now treat `ChatStreamPart::TextDelta` and
  `ChatStreamPart::ReasoningDelta` as the canonical runtime text/reasoning stream model.
- OpenAI-compatible Perplexity typed metadata now also preserves hosted-search
  `usage.reasoningTokens` instead of dropping that field while normalizing the response into the
  stable provider-rooted metadata surface.
- Shared stream wrappers now keep semantic-only streams semantic-only, so `StreamFactory` tail
  injection and `SimulateStreamingMiddleware` no longer append duplicate text after typed text
  parts.
- Shared provider metadata aggregation is now aligned around the spec-level
  `ProviderMetadataMap`: stream aggregation, OpenAI-compatible terminal metadata, and typed
  metadata accessors now all keep explicit provider-rooted object payloads instead of
  lane-specific nested `HashMap` conventions.
- OpenAI Responses cross-protocol bridging now prefers the stable semantic part lane: parseable
  legacy/custom v3 stream-part payloads are promoted into `ChatStreamEvent::Part` /
  `PartWithReplay`, and `LanguageModelV3StreamPart::to_runtime_part()` is public so adapters can
  project the typed overlay back into the runtime contract without reusing provider-prefixed
  custom events.
- Injectable HTTP transport parity now also covers non-stream `GET` / binary `GET` executor
  paths: `execute_get_request` and `execute_get_binary` now route through
  `HttpTransport::execute_get`, preserving per-request headers and retry/401-refresh behavior
  instead of bypassing custom transport wiring on provider-owned GET resources.
- OpenAI Chat request conversion now maps prompt-side provider-owned `image` / `file`
  references onto native `file_id` payloads, while OpenAI-compatible request conversion rejects
  those unsupported provider references explicitly instead of silently degrading them.
- OpenAI-compatible chat streaming now honors runtime raw-chunk emission on the stable part lane:
  `OpenAiCompatibleEventConverter::with_include_raw_chunks(true)` emits `raw` immediately after
  `stream-start` and before `response-metadata` on the first parsed chunk, matching the upstream
  AI SDK ordering more closely.
- OpenAI-compatible chat streaming now also preserves `stream-start` ordering on first-chunk parse
  failures: invalid JSON SSE payloads emit the stable `StreamStart` event/part before optional
  `raw` output and the final parse error, keeping the lifecycle closer to AI SDK's
  `start() -> raw? -> error` behavior.
- Stable response/stream finish propagation is now closer to AI SDK: `ChatResponse` carries
  `raw_finish_reason`, shared OpenAI-compatible chat decoding fills it from provider finish
  reasons, and `StreamProcessor` preserves raw finish reasons from terminal responses or stable
  `ChatStreamPart::Finish` parts when building final responses.
- OpenAI-compatible chat request/response shaping now also follows the audited AI SDK raw/camelCase
  provider-key contract more closely: provider-owned passthrough options merge raw + camelCase
  keys with camelCase taking precedence, compat response/finish metadata keep the resolved
  request-side namespace key with an explicit provider root, and Gemini-compatible
  `extra_content.google.thought_signature` now survives on finalized compat tool calls as
  `providerMetadata.{provider}.thoughtSignature`.
- `ClientWrapper` now forwards completion operations, and `ProviderCapabilities` tracks a native
  `completion` flag plus `supports("completion")`, so completion-capable clients can flow through
  generic runtime/registry adapters without losing capability checks.
- Provider-aware OpenAI-compatible usage parsing now supports DeepInfra-specific normalization:
  inconsistent `completion_tokens` / `reasoning_tokens` totals are corrected before the stable
  `Usage` model is built, and completion/stream decoding paths now route through that provider-aware
  parser instead of the generic usage parser.
- Stable provider typing now includes `ProviderType::DeepInfra`, and provider-aware retry/default
  validator helpers no longer downgrade DeepInfra to `Custom("deepinfra")`.
- Stable provider typing now also includes `ProviderType::Azure`, so provider-aware retry/default
  validator helpers no longer downgrade Azure OpenAI to `Custom("azure")`; the validator keeps
  Azure deployment ids permissive instead of pretending they are fixed OpenAI model names.
- Family routing variants now also collapse back to their canonical provider identity at the
  stable type layer: `openai-chat`, `openai-responses`, and `azure-chat` no longer read as fake
  custom providers when provider-aware retry/default-validator helpers inspect raw provider ids.
- Stable provider typing now also includes `ProviderType::Vertex` and
  `ProviderType::AnthropicVertex`, so provider-aware retry/default validator helpers no longer
  downgrade the Google Vertex native wrappers to `Custom(...)`.
- Provider-aware retry/default-validator helpers now also understand Vertex MaaS explicitly:
  Google-family backoff, default-model suggestions, stop-sequence limits, and model-family checks
  now treat `vertex-maas` as a first-class provider instead of a generic compat alias.
- Provider-aware retry/default-validator helpers now also understand Cohere, TogetherAI, and
  Bedrock explicitly instead of degrading those built-in providers to `Custom(...)`; Cohere now
  gets native model-family checks/default suggestions, TogetherAI now follows the OpenAI-compatible
  retry family explicitly, and Bedrock keeps a deliberate generic retry fallback.
- Provider-aware retry/default-validator helpers now also understand `mistral`, `fireworks`, and
  `perplexity` explicitly instead of degrading those AI SDK-packaged compat providers to
  `Custom(...)`; all three now follow the OpenAI-compatible retry family, and `mistral` also has a
  first-class embedding-default suggestion aligned with `mistral-embed`.
- OpenAI-compatible capability inference now also distinguishes chat-surface from
  completion-surface providers: `mistral` and `perplexity` no longer inherit completion capability
  just because they expose chat, while `fireworks` keeps explicit completion support.
- OpenAI-compatible chat/stream response decoding now normalizes usage through the shared AI SDK-style `inputTokens` / `outputTokens` / `raw` model instead of only preserving legacy totals and partial detail fields.
- The historical `LanguageModelV3StreamPart` typed overlay is now a V4-capable superset with first-class `custom` and `reasoning-file` parts, and OpenAI-compatible `AsText` fallback can now degrade those parts into explicit text instead of silently dropping them.
- The upgraded typed stream-part overlay now also exposes public `LanguageModelV4*` aliases so new code can use AI SDK-aligned names without losing compatibility with the historical `LanguageModelV3*` surface.
- `LanguageModelV3StreamPart` can now convert to and from the new spec-level `ChatStreamEvent::Part(ChatStreamPart)` runtime semantic channel, and `EventBuilder` exposes a first-class `add_part(...)` helper.
- `EventBuilder` now also exposes `add_part_with_replay(...)`, and the shared stream processor / encoder helpers treat runtime replay-bearing part events the same as ordinary semantic part events.
- Anthropic `reasoning-*` typed custom-event conversion now preserves stable `id` and `providerMetadata`, so protocol serializers can replay AI SDK-style reasoning signatures and redacted-thinking metadata from the semantic part surface instead of dropping to ad hoc custom payloads.
- `StreamProcessor` now preserves terminal response envelope fields from `StreamEnd`, including `id`, `model`, `audio`, `system_fingerprint`, `service_tier`, and `warnings`.
- Final stream aggregation now falls back to `StreamStart` metadata when no terminal response is available and retains terminal multimodal parts that were not rebuilt from deltas, such as sources and provider-decorated tool calls.
- `StreamProcessor` now consumes structured runtime stream parts directly, preserving streamed warnings, finish reasons, provider metadata, sources, custom content, generated files, reasoning files, tool approval requests, and completed tool result parts instead of dropping them at the transport boundary.
- OpenAI Responses stream bridging now rebuilds provider-hosted tool / MCP replay as stable part events with a runtime replay carrier instead of synthesizing provider-scoped custom payloads with embedded `rawItem` JSON.
- OpenAI-compatible stream decoding now keeps observed terminal chunk fields such as `system_fingerprint` and `service_tier` on `StreamEnd`, including the EOF fallback path when no explicit finish chunk is emitted.
- OpenAI-compatible EOF / `[DONE]` fallback now also emits the missing stable semantic suffix on
  the direct `Part` lane: active `text-*` / `reasoning-*` parts are closed, unfinished tool-call
  lifecycles are finalized, and a stable `finish` part is emitted before the legacy `StreamEnd`.
- OpenAI-compatible streaming now matches AI SDK model-router metadata timing more closely: placeholder Azure `prompt_filter_results` preludes with empty `id` / `model` and `created = 0` no longer synthesize early `response-metadata` or `1970-01-01` timestamps before the first real metadata chunk arrives.
- OpenAI-compatible response metadata extraction is now owned by provider adapters instead of a shared compat-layer whitelist, aligning more closely with AI SDK `openai-compatible` `metadataExtractor` semantics: known providers explicitly opt into `sources` / `logprobs` / prediction-token extraction, Perplexity keeps its provider-specific hosted-search metadata, and generic compat providers no longer infer those fields by default.
- The compat adapter layer now exposes a public `ResponseMetadataExtractor` hook plus `MetadataExtractingAdapter`, allowing config/builders to inject AI SDK-style OpenAI-compatible response metadata extraction without replacing the full provider adapter.
- OpenAI-compatible request settings now also include AI SDK-style `includeUsage` / `queryParams` / `transformRequestBody` concepts: compat chat streams omit `stream_options.include_usage` unless the provider config opts in explicitly, `OpenAiCompatibleRequestSettings` now also carries deterministic provider-level URL query params into compat route generation, and callers can install a public `RequestBodyTransformer` hook that runs after provider normalization on the final request body.
- The compat request-settings lane now also exposes an explicit provider-level `supportsStructuredOutputs` policy for chat-completions shaping aligned with AI SDK: JSON Schema `response_format` requests now default to wire `json_object` unless callers explicitly enable structured outputs with `supportsStructuredOutputs = true`.
- OpenAI-compatible chat request normalization now also interprets the audited AI SDK known compat provider options from canonical `openaiCompatible` and provider-owned request keys: `user`, `reasoningEffort`, `textVerbosity`, and `strictJsonSchema` are mapped onto stable wire fields instead of being forwarded as raw camelCase extras.
- Shared URL joining now preserves existing query strings when appending paths, so provider specs such as OpenAI-compatible audio can safely compose `/audio/*` endpoints on top of query-bearing base URLs.
- OpenAI-compatible SSE serialization now reuses the shared OpenAI chat-usage writer, preserving usage detail fields and provider-unknown totals instead of flattening replayed `usage` chunks to synthetic zero-valued legacy integers.
- OpenAI-compatible replay of typed V3 `finish` parts now preserves unknown usage totals as `null` when no stable prompt/completion totals are available.
- Shared URL joining now preserves existing query strings and fragments when appending paths, so
  query-bearing endpoint composition such as OpenAI Files `/files?...` no longer corrupts the
  request path.
- OpenAI-compatible stream metadata extraction now falls back to the request model when early
  provider chunks omit `model`, fixing Azure model-router placeholder chunks that previously
  dropped stable `stream-start.model`.
- OpenAI-compatible chat/non-stream response decoding now maps `message.annotations` / `delta.annotations` URL citations into stable `source` parts, and URL `source` stream parts now serialize back into chat-completions `annotations`.
- `systemMessageMode=remove` warnings now use the explicit `compatibility` warning shape instead of a generic warning string.
- OpenAI-compatible and OpenAI Chat request conversion now read canonical message/part `providerOptions` for request-only behavior such as extra params, assistant reasoning replay hints, and image detail; request-side `provider_metadata` / `message.metadata.custom` are no longer treated as input on those main paths.
- Audio helper APIs such as `transcribe_file(...)` and `translate_file(...)` now materialize local
  files into canonical shared audio inputs before execution, and the audio executor no longer owns
  request-shape `file_path` fallback behavior.
- Audio convenience helpers now also follow the required `mediaType` contract of the shared
  transcription request types: `transcribe(...)` / `translate(...)` take explicit media types, and
  file-based helpers fail fast when a local path does not imply a recognizable audio media type.
- OpenAI-compatible config defaults now treat built-in `openrouter` and `perplexity` presets as
  structured-output-capable, so config-first public paths preserve JSON Schema response formats by
  default instead of falling back to generic `json_object`.
- The shared HTTP image executor now materializes `data:` / `http(s):` URL-backed edit and
  variation inputs before synchronous multipart/inline transformers run, so OpenAI,
  OpenAI-compatible, and Vertex image paths no longer reject typed URL inputs that can be
  centralized in the executor layer.

## [0.11.0-beta.6] - 2026-03-02

### Added

- Apply per-request `HttpConfig` overrides (headers + timeout) at the HTTP executor layer, including streaming requests.
- Add convenience helpers on `dyn LlmClient` for full chat requests (`chat_request`, `chat_stream_request`).

## [0.11.0-beta.5] - 2026-01-15

### Added

- Provider-agnostic core extracted from the facade crate as part of the workspace split.
- Injectable HTTP transport (custom `fetch` parity), including streaming use-cases.
- Typed V3 stream parts and cross-protocol transcoding foundations used by gateway/proxy layers.

### Fixed

- Stricter SSE JSON parsing to reduce silent drift.
