# Fearless Registry Facade Construction Boundary - Milestones

Last updated: 2026-05-14

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

Status: In Progress

Apply the shared built-in registry helpers provider-by-provider inside
`provider_public_path_parity_test.rs` so facade parity coverage stops encoding concrete built-in
factory names.

Exit criteria:

- OpenAI, Gemini, Cohere, TogetherAI, DeepInfra, DeepSeek, Groq, Ollama, XAI, MiniMaxi, and
  Bedrock public-path registry setup use registry-owned built-in helper routing.
- Remaining direct concrete factory call sites are either provider contract tests, advanced
  low-level integrations, or explicitly tracked follow-up providers.
