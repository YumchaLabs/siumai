# Next Steps (Alpha.5 Split-Crate Refactor)

This document is the short, actionable follow-up plan after the current alpha.5 refactor batch.
For the full rationale and architecture direction, see:

- `docs/architecture-refactor-plan.md`
- `docs/module-split-design.md`
- `docs/roadmap-mvp.md`
- `docs/adr/README.md`

## Current checkpoint (what is already done)

- Provider protocol mapping (`standards/*`) is provider-owned, not `siumai-core` owned.
- OpenAI / Anthropic / Gemini typed `providerOptions` and typed `providerMetadata` are provider-owned and exposed via `siumai::provider_ext::<provider>::*`.
- `siumai-core` no longer depends on provider-specific typed structs; provider options are passed through as JSON payloads.

## Immediate next actions (recommended order)

1. **Finish provider-owned typed extensions for the remaining providers**
   - Move typed options + typed metadata for `groq`, `xai`, `minimaxi`, `ollama` (and any other built-ins) into their provider crates.
   - Keep `siumai-core` provider-agnostic; expose typed access via `siumai::provider_ext::<provider>::*`.

2. **Converge on one OpenAI-like strategy**
   - Treat “OpenAI-compatible vendors” as **configuration** (base URL, headers, error mapping), not as a separate “standard” split.
   - Consolidate duplicated OpenAI-like streaming parsing and request mapping into the shared OpenAI-like layer under `siumai-provider-openai` (family implementation), then have Groq/xAI/etc. depend on it.

3. **Deprecate the legacy closed `ProviderOptions` enum surface**
   - Prefer `provider_options_map` (provider-id keyed JSON object) everywhere.
   - Keep a temporary compatibility bridge for downstream users, but stop expanding the enum for new provider features.

4. **Tighten re-exports and stabilize the public paths**
   - Ensure the recommended stable surface stays in `siumai::prelude::unified::*`.
   - Ensure provider-specific imports are consistently under `siumai::provider_ext::<provider>::*`.
   - Minimize blanket `pub use` from umbrella crates to reduce accidental cross-layer imports.

5. **Write and ship a migration guide**
   - Document the “before/after” imports (e.g., move from `siumai::types::*` to `siumai::provider_ext::<provider>::*` for typed options/metadata).
   - Document how to attach provider options via `ChatRequest::with_*_options(T: Serialize)` and how providers parse options from `provider_options_map`.

## Guardrails (refactor safety)

- Keep a small “smoke” test matrix green (nextest):
  - `cargo nextest run -p siumai --features all-providers`
- Treat `repo-ref/` and `_third_party/` as local reference inputs; do not commit large vendor trees by default.

