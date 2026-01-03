# Next Steps (Alpha.5 Split-Crate Refactor)

This document is the short, actionable follow-up plan after the current alpha.5 refactor batch.
For the full rationale and architecture direction, see:

- `docs/architecture-refactor-plan.md`
- `docs/module-split-design.md`
- `docs/roadmap-mvp.md`
- `docs/adr/README.md`

## Current checkpoint (what is already done)

- Provider-specific protocol mapping lives in provider crates. For shared protocol families, we use dedicated “standard crates” reused by multiple providers:
  - OpenAI-like: `siumai-provider-openai-compatible/src/standards/openai/*`
  - Anthropic Messages: `siumai-provider-anthropic-compatible/src/standards/anthropic/*`
- `siumai-core` may host **protocol-level shared building blocks** under `siumai-core/src/standards/*` (e.g. OpenAI-compatible wire helpers), but it does not own provider-specific mapping.
- `siumai-registry` is now decoupled from the umbrella `siumai-providers` crate and wires built-ins by depending on individual provider crates directly.
- `siumai` is now decoupled from the umbrella `siumai-providers` crate and re-exports provider-specific APIs directly from provider crates (feature-gated).
- OpenAI / Anthropic / Gemini typed `providerOptions` and typed `providerMetadata` are provider-owned and exposed via `siumai::provider_ext::<provider>::*`.
- Groq / xAI / Ollama / MiniMaxi typed `providerOptions` are provider-owned and exposed via `siumai::provider_ext::<provider>::*`.
- `siumai-core` is provider-agnostic: requests carry an open `provider_options_map` (provider-id keyed JSON object).
- The legacy closed `ProviderOptions` enum transport has been removed (breaking change).
- Low-level building blocks are exposed via `siumai::experimental::*` only (no more top-level `siumai::{execution,auth,utils,params,retry,defaults,client,observability}` re-exports).

## Immediate next actions (recommended order)

1. **Write and ship a migration guide (breaking changes)**
   - Document the removal of the legacy `ProviderOptions` enum and the `provider_options` request fields.
   - Document “before/after” imports for typed provider options (moved to `siumai::provider_ext::<provider>::*`).
   - See: `docs/migration-0.11.0-beta.5.md`

2. **Converge on one OpenAI-like strategy**
   - Treat “OpenAI-compatible vendors” as **configuration** (base URL, headers, error mapping), not as a separate “standard” split.
   - Keep the OpenAI-like protocol mapping reusable and provider-agnostic by centralizing it in a shared crate:
     - protocol building blocks in `siumai-core/src/standards/openai/compat/*`
     - reusable mapping + streaming/tool-call helpers in `siumai-provider-openai-compatible`
     - provider-level wiring (spec + vendor presets) in provider crates (e.g. `siumai-provider-openai`, `siumai-provider-groq`, `siumai-provider-xai`)

3. **Tighten re-exports and stabilize the public paths**
   - Ensure the recommended stable surface stays in `siumai::prelude::unified::*`.
   - Ensure provider-specific imports are consistently under `siumai::provider_ext::<provider>::*`.
   - Minimize blanket `pub use` from umbrella crates to reduce accidental cross-layer imports.

4. **Decide the fate of the legacy umbrella crate (`siumai-providers`)**
   - Keep it as a compatibility layer (accepts some duplication), or retire it once downstream users migrate.
   - If kept: ensure it becomes a thin re-export layer only (no cross-layer logic).
   - Decide what to do with the shared model catalog long-term:
     - split it into a small dedicated crate (recommended), or
     - keep it in the facade and accept maintenance cost.

5. **Decide provider-owned metadata scope for each provider**
   - For each built-in provider, decide whether to expose typed response metadata.
   - If needed, move typed metadata to provider crates and expose via `siumai::provider_ext::<provider>::*`.

6. **Expand the refactor safety net (smoke matrix)**
   - Keep the default profile fast (`openai` only).
   - Ensure `openai-compatible` (OpenAI + Groq + xAI) stays green for shared-protocol drift detection.
    - Keep builder ergonomics provider-owned (extension traits) to avoid re-coupling `siumai-core` to providers.

## Guardrails (refactor safety)

- Keep a small “smoke” test matrix green (nextest):
  - `cargo nextest run -p siumai --features all-providers`
- Add a lightweight dependency guardrail to prevent provider→provider dependencies from creeping back in (except approved shared family crates like `siumai-provider-openai-compatible`).
- Treat `repo-ref/` and `_third_party/` as local reference inputs; do not commit large vendor trees by default.

## Stable surface reference

- `docs/public-surface.md` (recommended stable module paths for the `siumai` facade)
