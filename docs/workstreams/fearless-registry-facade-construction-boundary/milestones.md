# Fearless Registry Facade Construction Boundary - Milestones

Last updated: 2026-05-16

## M1 - Workstream Framing

Status: Complete

Document why concrete built-in provider factories should not be the normal public facade path.

Exit criteria:

- Design, TODO, and milestones exist.
- `docs/README.md` links the workstream.

## M2 - Built-in Factory Resolver

Status: Complete

Add a registry helper that maps provider ids to built-in factory instances and returns
`Arc<dyn ProviderFactory>`.

Exit criteria:

- Default registry creation uses the helper.
- Compatibility `SiumaiBuilder` uses the helper for provider selection.
- Feature-disabled providers return explicit `UnsupportedOperation` errors.

## M3 - Public Facade Cleanup

Status: Complete

Move focused public facade tests away from direct concrete built-in factory construction.

Exit criteria:

- Public helper and metadata boundary tests use `builtin_provider_factory(...)`.
- Public import smoke tests keep registry abstractions visible without encouraging concrete built-in
  factory types.
- A source guard prevents the cleaned public tests from regressing.

## M4 - Validation

Status: Complete

Run focused formatting, check, nextest, and diff hygiene validation.

Exit criteria:

- Formatting is clean for touched crates.
- `siumai-registry` and `siumai` focused checks pass with affected feature sets.
- Focused nextest runs pass.
- `git diff --check` has no whitespace errors.

## M5 - Provider Public-Path Sweep

Status: Complete

Apply the shared built-in registry helpers provider-by-provider inside
`provider_public_path_parity_test.rs` so facade parity coverage stops encoding concrete built-in
factory names and hand-written registry construction plumbing.

Exit criteria:

- OpenAI, Azure, Gemini, Cohere, TogetherAI, DeepInfra, DeepSeek, Groq, Ollama, XAI, MiniMaxi,
  Bedrock, Anthropic, and Google Vertex public-path registry setup use registry-owned built-in
  helper routing; Azure URL-mode variants use the registry-owned Azure option helper; dynamic
  OpenAI-compatible provider-id variants use the registry-owned OpenAI-compatible helper.
- OpenAI, Gemini, Cohere, TogetherAI, Azure, DeepSeek, Vertex MaaS, Ollama, XAI, Bedrock,
  Anthropic, Groq, MiniMaxi, Google Vertex, and dynamic OpenAI-compatible public-path registry
  setup use `RegistryBuilder` provider-level shortcuts rather than hand-written `RegistryOptions {
  provider_build_overrides: ... }` maps or generic `.with_provider_build_overrides(...)`
  wrappers in migrated modules.
- Remaining direct concrete factory call sites are either provider contract tests, advanced
  low-level integrations, or explicitly tracked follow-up providers.

## M6 - Builder Default-Model Policy

Status: Complete

Move model-less compatibility `SiumaiBuilder` default selection out of the builder implementation
and into registry/provider-owned metadata.

Exit criteria:

- `SiumaiBuilder` delegates default-model resolution to
  `registry::helpers::builtin_provider_default_model(...)`.
- Native provider metadata declares either a default model or an explicit-model-required policy.
- Built-in provider catalog default-model records reuse the same native metadata instead of
  hand-written per-provider patches.
- A source guard prevents provider-specific default model constants from returning to
  `provider/build.rs`.

## M7 - Provider Build Override Ergonomics

Status: Complete

Move common provider-specific API-key/base-URL/custom-fetch override composition into registry-owned
constructors and `RegistryBuilder` shortcut methods.

Exit criteria:

- `ProviderBuildOverrides` has intent-named constructors for common test and custom transport setup.
- `RegistryBuilder` exposes provider-level shortcut methods that merge with existing provider
  overrides instead of requiring callers to hand-roll `HashMap` plumbing.
- Focused facade tests use the shortcuts for OpenAI, Gemini, Vertex, and DeepInfra request parity
  setup.
- A source guard prevents the focused facade tests from regressing to manual
  `provider_build_overrides` maps.

## M8 - Registry Options Default Source

Status: Complete

Make `RegistryOptions::default()` the single source of default registry construction behavior.

Exit criteria:

- `RegistryOptions` has a manual `Default` implementation that preserves `':'` separator and
  enabled automatic middleware semantics.
- `create_provider_registry(..., None)` reuses `RegistryOptions::default()` instead of maintaining
  a second default tuple.
- Internal helpers and small tests use `None` or `..Default::default()` instead of spelling every
  default field.
- Source guard coverage prevents the raw default tuple from returning to `create_provider_registry`.

## Closeout

Status: Complete

All milestones for this construction-boundary lane are complete as of 2026-05-16. Remaining ideas
around making concrete factory exports experimental-only or further de-duplicating public-path test
helpers are deferred follow-ups, not open exit criteria for this workstream.
