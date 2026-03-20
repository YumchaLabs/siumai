# Fearless Refactor V4 - Validation Matrix

Last updated: 2026-03-20

## Purpose

The V4 refactor is no longer blocked on large architectural moves.

The main risk now is **feature drift across package boundaries**:

- a provider package compiles on its own but not through `siumai`
- a facade feature compiles, but the provider-owned package surface regresses
- examples or public-surface tests drift behind the real provider matrix

This document defines the validation lanes that close those gaps.

## Validation tiers

### Tier 0 - Local refactor loop

Use these during normal implementation:

- `cargo fmt --all --check`
- `./scripts/test-fast.sh`
- `./scripts/test-smoke.sh`
- targeted `cargo nextest run -p siumai --test ...`

Goal:

- catch obvious compile and contract regressions quickly
- keep no-network parity tests as the default signal
- avoid paying the `--all-features --workspace` cost for every edit

### Tier 1 - Pull request gates

PR validation must prove that the public package layout still compiles in feature-minimal mode.

Required lanes:

1. `test`
   - core safety net
   - minimal facade compile/test path (`siumai` with `openai`)

2. `pr-provider-smoke`
   - single-feature facade compile matrix for each first-class provider surface
   - current set:
     - `openai`
     - `azure`
     - `anthropic`
     - `google`
     - `google-vertex`
     - `ollama`
     - `xai`
     - `groq`
     - `minimaxi`
     - `deepseek`
     - `cohere`
     - `togetherai`
     - `bedrock`

3. `pr-provider-contracts`
   - provider-scoped, no-network `cargo nextest` lanes for the top-level `siumai` facade
   - validates one representative contract bundle per provider without requiring `all-features`
   - current profiles:
     - `openai-native`
     - `openai-compat`
     - `azure`
     - `anthropic`
     - `google`
     - `google-vertex`
     - `ollama`
     - `xai`
     - `groq`
     - `minimaxi`
     - `deepseek`
     - `cohere`
     - `togetherai`
     - `bedrock`

4. `pr-cross-feature-contracts`
   - no-network `cargo nextest` lanes for non-default multi-feature combinations
   - current profiles:
      - `openai-websocket`
      - `google-gcp`
      - `openai-json-repair`
   - the `openai-json-repair` lane also locks structured-output refusal/content-filter semantics,
     so best-effort JSON repair cannot silently reinterpret plain refusal text as a successful JSON
     string result
   - protects feature interactions that compile-only smoke does not fully exercise

5. `provider-package-build-matrix`
   - compile each provider package directly with its own feature gate
   - protects provider-owned public surfaces that the facade does not compile exhaustively
   - also covers compatibility packages such as:
     - `siumai-provider-openai-compatible`
     - `siumai-provider-anthropic-compatible`

6. `pr-facade-guardrails`
   - compile all `siumai` examples under `all-providers`
   - run `public_surface_imports_test`

Acceptance rule:

- a provider migration is not considered operationally complete until it is present in:
  - the facade feature smoke lane
  - the provider contract lane
  - the direct provider-package build lane
- release readiness also requires the cross-feature contract lane to stay green for the current
  non-default coupling set (`openai-websocket`, `google,gcp`, `openai,json-repair`)

### Tier 2 - Mainline / merge validation

These lanes are allowed to be heavier:

1. `cargo nextest run --profile ci --all-features --workspace`
2. facade feature build matrix
3. extras feature build matrix
4. docs build
5. formatting and clippy

The facade feature build matrix must include:

- all first-class single-provider features
- cross-feature combinations that exercise non-default wiring:
  - `openai,openai-websocket`
  - `google,gcp`
  - `openai,json-repair`
  - `all-providers`

### Tier 3 - Release readiness

Before calling the V4 line release-ready:

- all Tier 1 and Tier 2 lanes must be green
- major examples must compile on their recommended construction paths
- at least one no-network parity or fixture test must exist for every newly aligned provider story
- when a fix touches provider-side default propagation, auth routing, or transport selection, add one
  targeted live smoke with real credentials on the affected provider path; for native OpenAI chat
  this now explicitly means verifying both explicit-request-model and builder/config-default-model
  `chat_request(...)` against a real model (most recently `gpt-5.2`) behind the configured proxy
- migration docs must point users to the preferred path:
  - registry-first
  - config-first
  - builder convenience last

Live integration tests remain valuable, but they are **not** the primary architectural gate.

## Package-boundary mapping

Use the following mapping when deciding where to add validation:

| Surface | Primary validation | Secondary validation |
|---|---|---|
| `siumai-provider-*` provider-owned surface | `provider-package-build-matrix` | provider-local tests / examples |
| `siumai` facade single-provider feature | `pr-provider-smoke` / `pr-provider-contracts` / `feature-build-matrix` | top-level no-network tests |
| `siumai` public examples and exports | `pr-facade-guardrails` | docs review |
| `siumai-extras` optional integration layers | `extras-build-matrix` | focused crate tests |

## Current focus

The current post-refactor validation focus is:

1. keep every provider package independently buildable
2. keep `siumai` feature-minimal builds honest
3. make the example tree reflect the real package tiers
4. prevent “default feature passes, non-default provider breaks” regressions

## Non-goals

This matrix is not trying to:

- make every provider participate in every Stable family
- require live-network tests for routine refactor confidence
- force one giant `all-features` job to be the only source of truth

The goal is a layered gate system where **package boundaries, feature boundaries, and public
examples all have explicit owners**.

## Recent targeted validations

- `registry-options-test-debt`
  - Historical test-debt fix:
    - `siumai/tests/registry_openai_compat_ignored.rs` now uses the current `RegistryOptions` shape instead of the pre-override field set.
    - `siumai/tests/middleware_override_test.rs` now centralizes test-only `RegistryOptions` construction and explicitly declares mock factory chat capabilities, matching the current registry `language_model(...)` gate.
  - Targeted regression:
    - `cargo test --target-dir F:\SourceCodes\Rust\siumai\.codex-target -p siumai --test middleware_override_test --quiet`
    - `cargo test --target-dir F:\SourceCodes\Rust\siumai\.codex-target -p siumai --test registry_openai_compat_ignored --quiet`

- `extensibility-example-build-debt`
  - Historical example-debt fix:
    - the `06-extensibility` examples now match the current `HttpChatExecutor` contract by passing `Some(...)` request/response transformers and setting `defer_transformer_selection` explicitly, instead of relying on the older non-optional transformer fields.
  - Validation:
    - `cargo test --target-dir F:\SourceCodes\Rust\siumai\.codex-target -p siumai --no-run --quiet`

- `openai-public-path-contract-sweep`
  - Wider no-network contract sweep:
    - `cargo nextest run --target-dir F:\SourceCodes\Rust\siumai\.codex-target -p siumai --features openai --test provider_public_path_parity_test --test openai_embedding_public_helper_request_parity_test --test streaming_tests`
  - Result:
    - 144 tests passed, 1 skipped
    - covers OpenAI native public-path parity, selected OpenAI-compatible public-path parity enrolled under the `openai` facade feature, embedding helper request preservation, interceptor request assertions, and streaming cancel/retry invariants

- `openai-live-smoke-gpt-5.2`
  - Root-cause fix:
    - OpenAI native streaming `/responses` requests no longer inject Chat Completions-only `stream_options.include_usage`; live `gpt-5.2` validation surfaced that OpenAI rejects `stream_options.include_usage` on the Responses route.
  - Package-local regression coverage:
    - `cargo test --target-dir F:\SourceCodes\Rust\siumai\.codex-target -p siumai-protocol-openai --features openai-responses --quiet`
    - `cargo test --target-dir F:\SourceCodes\Rust\siumai\.codex-target -p siumai-provider-openai --lib --quiet`
  - Targeted live smoke:
    - `cargo test --target-dir F:\SourceCodes\Rust\siumai\.codex-target -p siumai --test openai_live_smoke_test --features openai -- --ignored --nocapture`
  - Covered surface:
    - registry-first `openai:gpt-5.2` non-stream + stream text generation
    - registry-first `openai:text-embedding-3-small` batch embedding helper

- `anthropic-google-vertex-contract-sweep`
  - Wider no-network contract sweep:
    - `cargo nextest run --target-dir F:\SourceCodes\Rust\siumai\.codex-target -p siumai --features anthropic,google,google-vertex --test provider_public_path_parity_test --test anthropic_messages_fixtures_alignment_test --test anthropic_messages_stream_fixtures_alignment_test --test google_generative_ai_fixtures_alignment_test --test google_generative_ai_stream_fixtures_alignment_test --test google_vertex_typed_metadata_boundary_test --test gemini_embedding_batch_helper_parity_test --test vertex_embedding_batch_helper_parity_test`
  - Result:
    - 238 tests passed, 0 skipped
    - covers Anthropic public-path parity plus fixture-alignment invariants, Gemini/Google public-path and streaming fixture parity, Vertex typed-metadata boundary coverage, and Gemini/Vertex embedding batch-helper request preservation

- `embedding-request-aware-bridge`
  - Build matrix pass:
    - `cargo check --target-dir C:\Temp\siumai-target -p siumai-core --lib --quiet`
    - `cargo check --target-dir C:\Temp\siumai-target -p siumai-provider-openai --features openai --quiet`
    - `cargo check --target-dir C:\Temp\siumai-target -p siumai-provider-openai-compatible --features openai-standard --quiet`
    - `cargo check --target-dir C:\Temp\siumai-target -p siumai-provider-gemini --features google --quiet`
    - `cargo check --target-dir C:\Temp\siumai-target -p siumai-provider-google-vertex --features google-vertex --quiet`
    - `cargo check --target-dir C:\Temp\siumai-target -p siumai-provider-ollama --features ollama --quiet`
    - `cargo check --target-dir C:\Temp\siumai-target -p siumai-provider-azure --features azure --quiet`
    - `cargo check --target-dir C:\Temp\siumai-target -p siumai-registry --quiet`
  - Focused runtime regression:
    - `cargo test --target-dir C:\Temp\siumai-target -p siumai-registry --lib embedding_model_handle_family_trait_preserves_request_config_on_bridge_path --quiet`
  - Public helper regression:
    - `cargo test --target-dir C:\Temp\siumai-target -p siumai --test openai_embedding_public_helper_request_parity_test --features openai --quiet`
  - Wrapper + batch follow-up regressions:
    - `cargo test --target-dir F:\SourceCodes\Rust\siumai\.codex-target -p siumai-registry --lib embedding_model_handle_embed_many_uses_native_family_batch_path_when_available --quiet`
    - `cargo test --target-dir F:\SourceCodes\Rust\siumai\.codex-target -p siumai --test azure_embedding_request_extensions_parity_test --features azure --quiet`
    - `cargo check --target-dir F:\SourceCodes\Rust\siumai\.codex-target -p siumai-provider-gemini --features google --quiet`
    - `cargo test --target-dir F:\SourceCodes\Rust\siumai\.codex-target -p siumai --test gemini_embedding_batch_helper_parity_test --features google --quiet`
    - `cargo check --target-dir F:\SourceCodes\Rust\siumai\.codex-target -p siumai-provider-google-vertex --features google-vertex --quiet`
    - `cargo test --target-dir F:\SourceCodes\Rust\siumai\.codex-target -p siumai --test vertex_embedding_batch_helper_parity_test --features google-vertex --quiet`
