# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0-beta.8](https://github.com/YumchaLabs/siumai/compare/siumai-registry-v0.11.0-beta.7...siumai-registry-v0.11.0-beta.8) - 2026-05-18

### Added

- add google vertex xai provider boundary

### Fixed

- align anthropic files capability surface
- require explicit compat completion capabilities
- repair release test regressions

### Other

- *(clippy)* align provider feature gates
- lock xai package files boundary
- clarify togetherai audio extension boundary
- lock groq provider-owned speech boundary
- lock registry compat handle boundaries
- *(clippy)* clean release lint failures
- *(release)* prepare v0.11.0-beta.8
- mark registry family adapters removed
- align ProviderFactory compatibility audit
- isolate provider composite clients
- isolate language extension handle adapters
- remove vision builder capability hint
- remove provider-type builder alias
- remove registry speech handle tts alias
- remove static-header json executor helper
- remove dedicated vision compatibility surface
- migrate openai compatible override shortcuts
- migrate core provider override shortcuts
- migrate openai compatible registry overrides
- migrate vertex registry overrides
- migrate minimaxi registry overrides
- migrate groq registry overrides
- migrate anthropic registry overrides
- migrate bedrock registry overrides
- migrate xai registry overrides
- migrate ollama registry overrides
- migrate vertex maas registry overrides
- migrate public path registry overrides
- centralize registry option defaults
- add provider build override shortcuts
- move builder default models into registry metadata
- route facade parity registries through helpers
- guard public path registry helper boundary
- route openai compatible registries through helper
- route azure registry factory options through helper
- converge provider boundary architecture
- centralize builtin registry factory resolution
- move provider model aliases out of core
- harden crate boundaries
- converge fearless architecture boundaries
- *(examples)* move extras example index
- *(examples)* tighten example guidance
- clean stale refactor docs

### Added

- Add completion-family registry plumbing: `ProviderFactory::completion_model_family*`,
  `ProviderRegistryHandle::completion_model(...)`, `CompletionModelHandle`, and dedicated
  completion-model cache entries now let registry-first code resolve completion models without
  falling back to ad hoc chat/client paths.
- Add skill-upload registry/unified plumbing: `Siumai` and `LanguageModelHandle` now bridge the
  shared `SkillsCapability`, so AI SDK-style skill uploads can flow through unified/registry
  callers on the audited OpenAI and Anthropic paths instead of only through provider-specific
  clients.
- Built-in OpenAI-compatible factory wiring now advertises completion capability and can construct
  completion-family models through the same registry/provider-id path as chat and embeddings.
- Built-in native OpenAI and Azure factories now also advertise completion capability and can
  construct completion-family models through the direct provider path instead of falling back to
  generic chat/client routes.
- Built-in TogetherAI registry wiring now mirrors the AI SDK single-provider surface on canonical
  `togetherai`: chat/completion/embedding/speech/transcription resolve through the shared
  OpenAI-compatible runtime, image and rerank stay provider-owned, and the unified language-model
  client now exposes text-family, image, and rerank capabilities from one provider id.
- Built-in DeepInfra registry wiring now mirrors the AI SDK single-provider surface on canonical
  `deepinfra`: chat/completion/embedding resolve through the shared OpenAI-compatible runtime,
  image generation/edit stay provider-owned, the unified language-model client now exposes both
  text-family and image capabilities from one provider id, and provider-catalog output now reuses
  the audited curated DeepInfra text/completion/embedding/image model subset instead of listing
  only defaults.
- Built-in Vertex MaaS registry wiring now mirrors the AI SDK single-provider surface on canonical
  `vertex-maas`: chat/completion/embedding resolve through the shared OpenAI-compatible runtime,
  `project + location` can derive the Vertex `/endpoints/openapi` base URL, the unified
  language-model client now exposes the audited MaaS text-family capabilities from one provider id,
  and provider-catalog output now reuses the audited curated MaaS model subset instead of
  hardcoding a second string list.
- Built-in Cohere registry wiring now mirrors the AI SDK single-provider surface on canonical
  `cohere`: chat/embedding/rerank all resolve through the native `/v2` Cohere runtime, and the
  unified language-model client now exposes the audited Cohere capability set from one provider id.
- Built-in Fireworks registry wiring now mirrors the AI SDK single-provider surface on canonical
  `fireworks`: chat/completion/embedding/transcription resolve through the shared OpenAI-compatible
  runtime, image generation/edit stay provider-owned, and the unified language-model client now
  exposes both the audited Fireworks text-family and image capabilities from one provider id.

### Fixed

- Completion handles now enforce provider capability checks for both non-stream and streaming
  execution, while still bridging legacy `LlmClient` completion-capable providers into the stable
  completion family surface.
- Focused-provider catalog output is now closer to the provider-owned public surface: DeepSeek,
  Ollama, and MiniMaxi reuse provider-owned curated model sources/defaults instead of maintaining
  separate handwritten arrays, and MiniMaxi catalog output now also includes the image-family
  defaults exposed by the public provider wrapper.
- TogetherAI registry metadata/defaults now use the chat-facing unified provider story instead of a
  rerank-only interpretation: provider metadata advertises the full AI SDK-style capability set,
  the default model now points at the chat default, and public/registry contract tests pin the
  unified-builder override path plus native rerank fallback behavior.
- TogetherAI unified image routing now also matches the audited AI SDK package more closely:
  canonical `togetherai` image generation and edit both use provider-owned
  `POST /images/generations`, edit inputs map to `image_url`, mask edits are rejected before
  transport, and image default-model fallback stays separate from the chat default.
- TogetherAI's explicit compat escape hatch no longer collides with the native provider id on the
  public builder surface: `.openai().togetherai_openai_compatible()` is now the canonical shortcut
  on both `Provider` and `Siumai::builder()`, while the unified-builder path keeps the hidden
  `together` alias only internally so `openai`-only builds do not accidentally trip native
  TogetherAI feature gates.
- TogetherAI catalog/default-model data now reflects the full unified provider story instead of a
  chat+rerank fragment: the registry catalog lists embedding/image/speech/transcription/rerank
  family defaults for canonical `togetherai`, and the shared compat default-model helpers now
  expose public speech/transcription getters for unified-provider consumers.
- Hybrid OpenAI-compatible/native provider registration now merges adapter state into existing
  native registry records instead of overwriting native metadata, which keeps provider-owned
  names/capabilities/base URLs authoritative for TogetherAI and DeepInfra.
- Cohere registry metadata/defaults now use the native unified provider story instead of the old
  rerank-only interpretation: provider metadata advertises chat/streaming/tools/embedding/rerank,
  provider catalog output lists curated chat/embed/rerank models from the provider-owned Cohere
  model set instead of a duplicated hardcoded string list, and the canonical native path no longer
  injects an ambiguous provider-wide default model.
- Azure registry catalog output now also treats canonical `azure` as a first-class native provider
  instead of reporting it as `Custom(...)`, while intentionally keeping deployment ids
  model-list-light on the public provider catalog surface.
- Provider-catalog family lookups now also collapse OpenAI/Azure routing variants back to their
  canonical provider identity, so `openai-chat`, `openai-responses`, and `azure-chat` no longer
  appear as fake custom providers on registry-side provider-info queries.
- Default built-in registry helper wiring now also registers DeepInfra directly, so feature-minimal
  builtins builds no longer omit the canonical `deepinfra` provider id.
- Provider catalog output now also treats `vertex`, `anthropic-vertex`, and `vertex-maas` as
  first-class native providers with curated Google Vertex model lists instead of generic custom
  provider entries. The `vertex` / `anthropic-vertex` lists now also reuse provider-owned curated
  constant sets instead of keeping a second handwritten copy inside the catalog layer.
- Provider catalog output now also treats `azure`, `cohere`, `togetherai`, and `bedrock` as
  first-class provider types instead of reporting those built-in native providers as `Custom(...)`.
- Provider catalog output now also treats `mistral`, `fireworks`, and `perplexity` as first-class
  provider types instead of reporting those AI SDK-packaged compat providers as `Custom(...)`,
  while still reusing the shared OpenAI-compatible runtime and listing curated default models on
  the catalog surface.
- OpenAI-compatible registry factories now also preserve the audited provider-package completion
  split: `mistral` and `perplexity` reject completion-family handles, while `fireworks` continues
  to advertise and build completion-family models.
- Fireworks registry metadata/defaults now also use the unified provider story instead of the old
  identity-only promotion: native metadata advertises provider-owned image capability, provider
  catalog output lists curated Fireworks image models, and the unified factory now keeps text and
  image default-model fallbacks separated so the chat default does not leak into image execution.
- The compatibility unified builder no longer injects provider-wide default models for `vertex`
  or `anthropic-vertex`; both Google Vertex wrappers now require explicit model ids, matching the
  AI SDK provider/model split instead of guessing a chat or image default.

## [0.11.0-beta.5] - 2026-01-15

### Added

- Registry extracted as a standalone crate (factories + handles), aligning with the workspace split.
- Wiring for provider crates to be constructed via factories instead of hard-coded built-ins.

### Changed

- Modularized provider/factory modules to reduce cross-layer coupling.
