# Fearless Registry Facade Construction Boundary - TODO

Last updated: 2026-05-16

## Current Slice

- [x] Define the registry/facade construction boundary in a dedicated workstream.
- [x] Add a registry-owned helper for built-in provider factory selection.
- [x] Reuse the helper from default registry creation.
- [x] Reuse the helper from compatibility `SiumaiBuilder` construction.
- [x] Update public facade tests to use the helper instead of concrete built-in factory structs.
- [x] Add source guard coverage for the new boundary.
- [x] Run focused formatting, check, and nextest validation.
- [x] Start the `provider_public_path_parity_test.rs` cleanup by extracting OpenAI built-in
      registry test helpers and routing OpenAI public-path registry setup through
      `builtin_provider_factory`.
- [x] Repeat the public-path registry helper cleanup for Gemini, keeping the built-in factory
      lookup registry-owned while preserving custom transport and wiremock request parity tests.
- [x] Lift the shared public-path test helpers to the file scope so future built-in provider
      migrations only need to supply provider ids and provider-specific base URLs.
- [x] Apply the shared built-in registry helper to Cohere public-path tests, covering chat,
      embedding, rerank, and provider-specific override registry construction.
- [x] Apply the shared built-in registry helper to TogetherAI public-path tests, covering rerank,
      image, chat, completion, stream, and provider-specific override registry construction.
- [x] Apply the shared built-in registry helper to DeepInfra public-path tests, covering chat,
      image generation, image edit, completion, and streaming completion parity construction.
- [x] Apply the shared built-in registry helper to DeepSeek public-path tests, covering
      `create_provider_registry` and `RegistryBuilder` construction while preserving reasoning
      default and provider-specific override coverage.
- [x] Apply the shared built-in registry helper to Groq public-path tests, covering
      `create_provider_registry`, provider-specific override construction, metadata preservation,
      chat, streaming chat, speech, and transcription parity paths.
- [x] Apply the shared built-in registry helper to Ollama public-path tests, covering
      `create_provider_registry`, chat, streaming chat, embedding, and provider-specific override
      construction; preserve native registry embedding extension semantics for Ollama-specific
      request options.
- [x] Apply the shared built-in registry helper to XAI public-path tests, covering
      `create_provider_registry`, `RegistryBuilder`, provider-specific build overrides, metadata
      preservation, chat, streaming chat, image, speech, transcription, and video query parity
      paths.
- [x] Apply the shared built-in registry helper to MiniMaxi public-path tests, covering
      `create_provider_registry`, wiremock-backed provider-specific build overrides, custom
      transport overrides, chat, streaming chat, file management, image, speech, music, video, and
      video query parity paths.
- [x] Apply the shared built-in registry helper to Bedrock public-path tests, covering
      `create_provider_registry`, provider-specific build overrides, chat, streaming chat,
      embedding, image, and rerank parity paths.
- [x] Apply the shared built-in registry helper to Anthropic public-path tests, covering
      `create_provider_registry`, `RegistryBuilder`, provider-specific build overrides, chat,
      streaming chat, typed options, metadata, reasoning, and unsupported audio/non-text parity
      paths.
- [x] Apply the shared built-in registry helper to Google Vertex public-path tests, covering
      `create_provider_registry`, `RegistryBuilder`, Vertex, Anthropic Vertex, Vertex MaaS,
      provider-specific build overrides, chat, streaming chat, image, image edit, video, embedding,
      typed metadata, structured output, and unsupported non-text parity paths.
- [x] Add a registry-owned Azure factory option helper and route Azure public-path registry setup
      through registry-owned helpers, including default built-in construction, deployment-based URL
      mode, provider metadata-key selection, embedding, image, speech, transcription, chat, and
      streaming chat parity paths.
- [x] Add a registry-owned OpenAI-compatible provider factory helper and route dynamic provider-id
      public-path registry setup through it, covering TogetherAI fallback-style audio/image/
      embedding lanes plus SiliconFlow, Jina, VoyageAI, Fireworks, Mistral, OpenRouter,
      Perplexity, and Infini registry override paths.
- [x] Move compatibility `SiumaiBuilder` default-model selection out of the builder's
      provider-specific `match` and into registry/provider-owned metadata helpers. Native
      providers now declare default-model or explicit-model-required policy in
      `native_provider_metadata`, the built-in catalog reuses that metadata for
      `ProviderRecord::default_model`, and `SiumaiBuilder` delegates model-less construction to
      `registry::helpers::builtin_provider_default_model(...)`.
- [x] Improve `ProviderBuildOverrides` ergonomics by adding registry-owned constructors for common
      API-key/base-URL/custom-fetch combinations and `RegistryBuilder` provider-level shortcut
      methods, then migrate focused facade tests away from hand-rolled `provider_build_overrides`
      maps.
- [x] Make `RegistryOptions::default()` the single source of registry option defaults, update
      `create_provider_registry(..., None)` to reuse it, and migrate internal helpers plus small
      tests away from full-field default option literals.
- [x] Continue the public-path raw options cleanup by migrating Azure and DeepSeek
      `provider_public_path_parity_test.rs` registry setup through `RegistryBuilder`
      provider-level shortcuts and adding guard coverage for those modules.
- [x] Continue the public-path raw options cleanup by migrating Vertex MaaS registry setup through
      `RegistryBuilder` provider-level shortcuts, including base-url + HTTP-config + custom-fetch
      composition, and extending guard coverage to prevent raw `RegistryOptions` plumbing from
      returning to that module.
- [x] Continue the public-path raw options cleanup by migrating Ollama registry setup through
      `RegistryBuilder` provider-level shortcuts, covering the default registry path and the
      provider-specific override path with global transport/base-url defaults.
- [x] Continue the public-path raw options cleanup by migrating XAI registry setup through
      `RegistryBuilder` provider-level shortcuts, covering custom transport overrides, global
      reasoning defaults, and the wiremock-backed video query registry path.
- [x] Continue the public-path raw options cleanup by migrating Bedrock registry setup through
      `RegistryBuilder` provider-level shortcuts, covering the default registry path and the
      provider-specific override path with global transport/base-url defaults.
- [x] Continue the public-path raw options cleanup by migrating Anthropic registry setup through
      `RegistryBuilder` provider-level shortcuts, covering the default registry path and the
      provider-specific override path with global transport/base-url defaults.
- [x] Continue the public-path raw options cleanup by migrating Groq registry setup through
      `RegistryBuilder` provider-level shortcuts, including metadata override tests that previously
      hand-rolled `ProviderBuildOverrides` inside the builder chain.
- [x] Continue the public-path raw options cleanup by migrating MiniMaxi registry setup through
      `RegistryBuilder` provider-level shortcuts, covering custom transport overrides,
      wiremock-backed base URL overrides, and video query registry construction.

## Follow-up Candidates

- [ ] Decide whether `siumai::registry::factories` should become experimental-only in a future
      breaking cleanup.
- [ ] Continue auditing `provider_public_path_parity_test.rs` for registry setup duplication that
      can move into shared helper functions without weakening provider-specific override coverage.
- [ ] Continue migrating larger provider-specific public-path modules from manual
      `RegistryOptions { provider_build_overrides: ... }` setup to `RegistryBuilder`
      provider-level shortcuts where the test does not intentionally cover raw options plumbing.

## Done Criteria

- Built-in factory selection has one registry-owned implementation.
- Facade tests that only need built-in providers do not import concrete factory types.
- Custom registry support remains intact.
- Focused checks pass.

## Validation Log

- `cargo fmt --package siumai-registry`
- `cargo check -p siumai-registry --tests --features openai --no-default-features`
- `cargo check -p siumai-registry --tests --features openai,anthropic,google,google-vertex,azure,groq,deepseek,ollama,cohere,togetherai,minimaxi,bedrock,xai --no-default-features`
- `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --features openai --no-default-features --no-fail-fast compatibility_builder_uses_registry_owned_default_model_resolution focused_public_facade_tests_use_registry_owned_builtin_factory_resolution`
- `cargo fmt --package siumai-registry --package siumai --check`
- `cargo check -p siumai-registry --tests --features openai --no-default-features`
- `cargo nextest run -p siumai-registry --features openai --no-default-features --no-fail-fast provider_build_overrides_constructors_match_fluent_chain registry_builder_merges_provider_specific_shortcuts focused_public_facade_tests_use_provider_build_override_shortcuts`
- `cargo check -p siumai --tests --features openai,google,google-vertex,deepinfra --no-default-features`
- `cargo fmt --package siumai-registry --package siumai --check`
- `cargo check -p siumai-registry --tests --features openai --no-default-features`
- `cargo nextest run -p siumai-registry --features openai --no-default-features --no-fail-fast registry_options_default_keeps_registry_creation_defaults registry_options_default_is_create_provider_registry_default_source lru_cache_eviction ttl_expiration language_model_inherits_registry_interceptors`
- `cargo check -p siumai --tests --features openai --no-default-features`
- `cargo fmt --package siumai-registry --package siumai --check`
- `cargo check -p siumai-registry --tests --features openai --no-default-features`
- `cargo nextest run -p siumai-registry --features openai --no-default-features --no-fail-fast migrated_public_path_modules_use_registry_builder_shortcuts focused_public_facade_tests_use_provider_build_override_shortcuts`
- `cargo check -p siumai --tests --features azure,deepseek --no-default-features`
- `cargo fmt --package siumai-registry --package siumai --check`
- `cargo check -p siumai-registry --tests --features openai --no-default-features`
- `cargo nextest run -p siumai-registry --features openai --no-default-features --no-fail-fast registry_builder_merges_provider_specific_shortcuts migrated_public_path_modules_use_registry_builder_shortcuts focused_public_facade_tests_use_provider_build_override_shortcuts`
- `cargo check -p siumai --tests --features google-vertex --no-default-features`
- `cargo nextest run -p siumai --test provider_public_path_parity_test --features google-vertex --no-default-features --no-fail-fast vertex_maas_public_path`
- `cargo check -p siumai --tests --features azure,deepseek,google-vertex --no-default-features`
- `cargo fmt --package siumai-registry --package siumai --check`
- `cargo nextest run -p siumai-registry --features openai --no-default-features --no-fail-fast migrated_public_path_modules_use_registry_builder_shortcuts`
- `cargo check -p siumai --tests --features ollama --no-default-features`
- `cargo nextest run -p siumai --test provider_public_path_parity_test --features ollama --no-default-features --no-fail-fast ollama_public_path`
- `cargo fmt --package siumai-registry --package siumai --check`
- `cargo nextest run -p siumai-registry --features openai --no-default-features --no-fail-fast migrated_public_path_modules_use_registry_builder_shortcuts`
- `cargo check -p siumai --tests --features xai --no-default-features`
- `cargo nextest run -p siumai --test provider_public_path_parity_test --features xai --no-default-features --no-fail-fast xai_public_path`
- `cargo fmt --package siumai-registry --package siumai --check`
- `cargo nextest run -p siumai-registry --features openai --no-default-features --no-fail-fast migrated_public_path_modules_use_registry_builder_shortcuts`
- `cargo check -p siumai --tests --features bedrock --no-default-features`
- `cargo nextest run -p siumai --test provider_public_path_parity_test --features bedrock --no-default-features --no-fail-fast bedrock_public_path`
- `cargo fmt --package siumai-registry --package siumai --check`
- `cargo nextest run -p siumai-registry --features openai --no-default-features --no-fail-fast migrated_public_path_modules_use_registry_builder_shortcuts`
- `cargo check -p siumai --tests --features anthropic --no-default-features`
- `cargo nextest run -p siumai --test provider_public_path_parity_test --features anthropic --no-default-features --no-fail-fast anthropic_public_path`
- `cargo fmt --package siumai-registry --package siumai --check`
- `cargo nextest run -p siumai-registry --features openai --no-default-features --no-fail-fast migrated_public_path_modules_use_registry_builder_shortcuts`
- `cargo check -p siumai --tests --features groq --no-default-features`
- `cargo nextest run -p siumai --test provider_public_path_parity_test --features groq --no-default-features --no-fail-fast groq_public_path`
- `cargo fmt --package siumai-registry --package siumai --check`
- `cargo nextest run -p siumai-registry --features openai --no-default-features --no-fail-fast migrated_public_path_modules_use_registry_builder_shortcuts`
- `cargo check -p siumai --tests --features minimaxi --no-default-features`
- `cargo nextest run -p siumai --test provider_public_path_parity_test --features minimaxi --no-default-features --no-fail-fast minimaxi_public_path`
