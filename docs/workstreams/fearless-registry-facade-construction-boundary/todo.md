# Fearless Registry Facade Construction Boundary - TODO

Last updated: 2026-05-14

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

## Follow-up Candidates

- [ ] Decide whether `siumai::registry::factories` should become experimental-only in a future
      breaking cleanup.
- [ ] Continue auditing `provider_public_path_parity_test.rs` provider-by-provider for repeated
      local registry builder snippets; keep Azure custom URL-config paths separate from generic
      built-in helper migration.
- [ ] Move provider-specific default model selection out of compatibility `SiumaiBuilder` and into
      registry/provider metadata once the construction helper is stable.
- [ ] Revisit `ProviderBuildOverrides` ergonomics for common test/custom-transport setup.

## Done Criteria

- Built-in factory selection has one registry-owned implementation.
- Facade tests that only need built-in providers do not import concrete factory types.
- Custom registry support remains intact.
- Focused checks pass.
