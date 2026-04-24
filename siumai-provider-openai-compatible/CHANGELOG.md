# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add compat-backed AI SDK-style `MistralProviderSettings` for the audited package-level
  `apiKey` / `baseURL` / `headers` / `fetch` construction subset, with upstream `generateId`
  explicitly deferred until the shared runtime owns a stable ID hook.
- Add compat-backed AI SDK-style `PerplexityProviderSettings` for the audited package-level
  `apiKey` / `baseURL` / `headers` / `fetch` construction subset.
- Add compat-backed AI SDK-style `DeepInfraProviderSettings` for the audited package-level
  `apiKey` / `baseURL` / `headers` / `fetch` construction subset.
- Add compat-backed AI SDK-style `MoonshotAIProviderSettings` for the audited package-level
  `apiKey` / `baseURL` / `headers` / `fetch` construction subset.
- Add compat-backed AI SDK-style `FireworksProviderSettings` for the audited package-level
  `apiKey` / `baseURL` / `headers` / `fetch` construction subset.
- Add compat-backed AI SDK-style `GoogleVertexMaasProviderSettings` for the audited package-level
  `project` / `location` / `baseURL` / `headers` / `fetch` construction subset, plus a Rust
  token-provider analogue for the Node auth wrapper.
- Add generic AI SDK-style `OpenAICompatibleProviderSettings` for the audited
  `@ai-sdk/openai-compatible` package settings surface, including `name`, required `baseURL`,
  optional `apiKey`, `headers`, `queryParams`, `fetch`, `includeUsage`,
  `supportsStructuredOutputs`, `transformRequestBody`, and `metadataExtractor`.

- Add AI SDK-aligned completion-family execution on the real OpenAI-compatible `/completions`
  route for both non-stream and SSE flows, including completion response usage/metadata/provider
  metadata propagation and optional streamed `include_usage`.
- Add AI SDK-style completion prompt materialization for structured prompts: first `system`
  prelude, text-only `user` / `assistant` turns, rejected `tool` / assistant `tool-call`
  messages, trailing `assistant:` prefix, and default `\nuser:` stop behavior.
- Add canonical `togetherai` compat preset metadata so the registry and explicit compat builder
  surfaces can expose the AI SDK-style TogetherAI text/audio boundary without duplicating the
  shared OpenAI-compatible runtime.
- Add curated DeepInfra `chat` / `completion` / `embedding` / `image` model constants on the
  compat provider surface, so promoted DeepInfra facade/catalog layers can reuse one audited model
  subset instead of duplicating string lists.
- Add curated Vertex MaaS `chat` / `completion` / `embedding` model constants on the compat
  provider surface, so promoted Vertex MaaS facade/catalog layers can reuse one audited model
  subset instead of hardcoding a parallel string list.

### Fixed

- Shared OpenAI-compatible auth/header validation now allows requests to proceed when an explicit
  `Authorization` header is already present, even if `api_key` is empty. This unblocks
  Google-style Bearer-token flows such as Vertex MaaS without requiring fake placeholder API keys.
- Shared OpenAI-compatible builders can now carry an async token provider directly, so
  Google-style Bearer-token flows can be configured through package settings instead of replacing
  the shared compat runtime.
- Generic OpenAI-compatible settings now use a plain synthetic provider config instead of reusing
  built-in provider presets when `name` collides with ids such as `groq`, and that generic path can
  build unauthenticated local/private gateway clients just like AI SDK's optional `apiKey` setting.
- Shared OpenAI-compatible tool warnings now support an allowlist of provider-defined tool ids, so
  provider-owned runtimes can tunnel supported server-side tools through Chat Completions without
  emitting duplicate generic unsupported warnings.
- Public OpenAI-compatible TogetherAI discovery/builder surfaces now prefer the canonical
  `togetherai` id: `.openai().togetherai_openai_compatible()` replaces the older
  `.openai().together()` shortcut, `list_provider_ids()` and supported-provider listings no longer
  advertise the legacy alias, while lower-level compatibility can still resolve `together`
  internally during migration.
- Shared compat default-model helpers now also expose public speech/transcription getters, so
  unified providers such as canonical `togetherai` can reuse one default-model registry for
  embedding/image/speech/transcription catalog output instead of reaching into private config
  helpers.

- OpenAI-compatible completion usage parsing is now provider-aware for DeepInfra as well: both
  non-stream and SSE completion flows normalize DeepInfra usage totals so reasoning tokens are not
  dropped when the upstream response underreports `completion_tokens`.
- OpenAI-compatible completion streaming terminal responses now preserve raw provider finish
  reasons on stable `ChatResponse.raw_finish_reason` instead of dropping them at stream end.
- Completion request shaping now merges audited provider options from deprecated
  `providerOptions['openai-compatible']`, canonical `providerOptions.openaiCompatible`, and
  provider-owned keys, including `logitBias -> logit_bias`, while surfacing explicit unsupported
  warnings for `topK`, `tools`, `toolChoice`, and structured `responseFormat`.
- Completion responses now also emit the AI SDK-style deprecation warning for legacy
  `providerOptions['openai-compatible']` so the completion family matches the audited compat chat
  warning lane.
- Completion response metadata now preserves raw completion `choices[0].logprobs` on the compat
  provider namespace instead of incorrectly reusing chat-completions logprob extraction.
- OpenAI-compatible completion streaming now also honors runtime-only `includeRawChunks` on the
  audited `/completions` SSE path: stable `stream-start`, `raw`, `response-metadata`, `text-*`,
  and terminal `finish` parts are emitted while preserving legacy `ContentDelta` / `StreamEnd`.
- OpenAI-compatible content-part metadata helpers now cover stable `reasoning-file` and `custom`
  parts instead of assuming the older content-part subset only.
- The built-in Perplexity compat preset now exposes AI SDK-shaped typed provider metadata more
  closely: `providerMetadata.perplexity` uses canonical `images.imageUrl|originUrl`,
  `usage.citationTokens|numSearchQueries`, and `cost.*` fields, while the Rust typed helper keeps
  parsing older snake_case compatibility payloads as aliases.
- OpenRouter and Perplexity typed request helpers now merge object-shaped provider options instead
  of replacing the whole provider namespace, so `with_openrouter_options(...)` /
  `with_perplexity_options(...)` behave consistently with the existing Mistral/Fireworks helper
  surface.
- OpenAI-compatible public provider surfaces now expose an AI SDK-style response metadata
  extractor hook: `ResponseMetadataExtractor`, `OpenAiCompatibleConfig::with_metadata_extractor`,
  and `OpenAiCompatibleBuilder::with_metadata_extractor` can extend built-in provider metadata
  extraction without requiring a custom `ProviderAdapter`.
- OpenAI-compatible public config/builder surfaces now also expose AI SDK-style request settings:
  `with_include_usage(...)` controls whether compat chat streams send
  `stream_options.include_usage`, default compat requests now omit that field until explicitly
  enabled, and `RequestBodyTransformer` / `with_request_body_transformer(...)` mirror AI SDK
  `transformRequestBody` on the final normalized chat payload.
- OpenAI-compatible public config/builder/runtime surfaces now also expose AI SDK-style
  `queryParams` and provider-level `supportsStructuredOutputs` concepts: compat route generation
  now appends deterministic provider query params across chat / embeddings / image / audio /
  rerank / model-listing paths, compat chat now defaults to downgrading JSON Schema outputs to
  `response_format = { "type": "json_object" }` while emitting a stable
  `unsupported { feature: "responseFormat" }` warning middleware on the chat response path, and
  callers can opt back into wire-level `json_schema` by setting
  `supports_structured_outputs(true)`.
- OpenAI-compatible chat runtime shaping now also honors AI SDK-style known compat request options
  from canonical `providerOptions.openaiCompatible` and provider-owned keys: `user`,
  `reasoningEffort`, `textVerbosity`, and `strictJsonSchema` now map to wire `user`,
  `reasoning_effort`, `verbosity`, and `response_format.json_schema.strict` instead of leaking as
  raw camelCase request fields.
- OpenAI-compatible image runtime shaping now also follows the AI SDK image provider-options lane:
  compat image generation/edit/variation merge provider-owned fields from deprecated
  `openai-compatible`, canonical `openaiCompatible`, and provider-owned keys instead of only
  `providerOptions.openai|azure`, and image generation now surfaces stable
  `unsupported { feature: "seed" }` warnings instead of silently dropping `seed`.
- OpenAI-compatible transcription public paths now follow the tightened shared AI SDK-aligned
  contract as well: stable STT requests require `mediaType`, and compat transcription tests/builders
  now compile against the canonical `from_audio(audio, mediaType)` surface.
- Built-in `openrouter` and `perplexity` compat presets now default `supportsStructuredOutputs` to
  enabled on their public builder/config/registry paths, matching the schema-preserving structured
  output behavior expected by their provider-owned parity tests.
- OpenAI-compatible chat runtime now also installs AI SDK-style provider-defined tool warnings on
  the default response path: provider-defined tools remain filtered out of Chat Completions
  requests, and successful chat responses now emit `unsupported { feature: "provider-defined tool
  <id>" }` warnings without extra user-installed middleware.
- OpenAI-compatible chat runtime now also installs the AI SDK deprecation warning for legacy
  `providerOptions['openai-compatible']`: the deprecated key still works for audited compat chat
  options, and successful chat responses now emit `other { message: "The 'openai-compatible' key
  in providerOptions is deprecated. Use 'openaiCompatible' instead." }` on the default response
  path.
- The built-in Together/TogetherAI compat defaults are now aligned on the same family-default map,
  capability set, and primary chat model, so canonical `togetherai` can reuse the shared compat
  registry path for completion/embedding/image/speech/transcription just like the AI SDK provider.
- The compat family-default table now also includes `mistral -> mistral-embed`, so default-model
  lookup, registry catalog output, and first-class provider-identity promotion for the AI SDK
  `mistral` package no longer lose its embedding default on the shared compat runtime.
- The compat family-default table now also includes the audited Fireworks image default
  (`accounts/fireworks/models/flux-1-dev-fp8`), so unified Fireworks registry/facade wiring can
  fall back from the text default model to the correct AI SDK image default on provider-owned
  image paths.
- OpenAI-compatible provider/client capability inference now also preserves the audited package
  boundary for AI SDK compat providers: `mistral` and `perplexity` keep chat-only public
  capabilities, while `fireworks` continues to expose completion on the shared compat runtime.
- OpenAI-compatible provider-option normalization now also covers the audited Fireworks chat
  request-shaping quirks: `thinking.budgetTokens` is normalized to `thinking.budget_tokens`,
  `reasoningHistory` is normalized to `reasoning_history`, and Fireworks reasoning-effort levels
  are down-mapped from `minimal|xhigh` to the wire-supported `low|high`. The public compat
  surface now also exposes matching typed Fireworks language-model options
  (`FireworksChatOptions` / `FireworksLanguageModelOptions`) plus `FireworksChatRequestExt`, so
  the audited AI SDK option shape no longer depends on raw `providerOptions.fireworks` JSON.
  Fireworks now also has an audited model-constant namespace on the compat public surface
  (`fireworks::{chat, completion, embedding, image}`), and the same curated AI SDK subset can be
  reused by higher layers instead of maintaining separate string lists.
- OpenAI-compatible Mistral/Perplexity parity is now closer to the audited AI SDK packages:
  provider-owned typed Mistral language-model options
  (`MistralChatOptions` / `MistralLanguageModelOptions` plus `MistralChatRequestExt`) are now
  public, compat runtime shaping now normalizes Mistral `safePrompt`, document limits,
  `structuredOutputs`, `parallelToolCalls`, and `reasoningEffort` instead of leaking camelCase
  request fields, built-in `mistral` config/runtime defaults now preserve JSON Schema structured
  outputs by default like the AI SDK package, and provider-owned curated Mistral/Perplexity model
  constants now back compat defaults plus the public provider catalog instead of stale hardcoded
  strings such as the old Perplexity legacy default id.

## [0.11.0-beta.5] - 2026-01-15

### Added

- OpenAI-compatible vendor presets and adapter registry extracted into a dedicated crate.

### Changed

- Version and dependency alignment with the split workspace layout.
