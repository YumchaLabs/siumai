# Roadmap (MVP) — Fearless Modular Refactor

This document is a concrete MVP plan for the ongoing modular refactor.
It is designed to be executed incrementally while preserving developer velocity.

## Current status (beta.5)

- M1 (open `providerOptions` map + merge semantics): implemented.
- M2 (move `standards/*` out of `siumai-core`): implemented (provider-owned).
- A (recommended entry for OpenAI-compatible vendors): use vendor presets like `Siumai::builder().moonshot()` / `LlmBuilder::new().moonshot()` (implemented as `openai().compatible("<vendor>")`).
- M6 (provider-first crates): OpenAI + Ollama + Anthropic + Gemini + Groq + xAI + MiniMaxi extracted into provider crates (compat shims kept in `siumai-providers`).

## Guiding principles

- Keep the stable public surface small: **6 model families** only.
- Treat provider-specific features as explicit extensions.
- Prefer incremental migration with compatibility bridges over “big bang” rewrites.
- Optimize for Rust reality: compile time, feature gating, and clear crate ownership.

## MVP definition (what “done” means)

MVP is complete when all items below are true:

1. **Provider options become open**
   - Requests carry `providerOptions` as a provider-id keyed JSON map (pass-through).
   - Providers can parse/validate their own options without touching core enums.
2. **Protocol mapping ownership is provider-side**
   - `siumai-core` no longer owns provider-specific protocol mapping modules (`standards/*`).
3. **Facade surface remains stable**
   - `siumai::prelude::unified::*` remains the recommended surface.
   - Provider-specific features remain accessible via `hosted_tools::*` + `provider_ext::*`.
4. **Refactor safety net stays green**
   - `./scripts/test-fast.sh` and `./scripts/test-smoke.sh` pass for at least the `openai` profile.

## Milestones

### M0 — Baseline and guardrails

- Document crate ownership and dependency rules (see `docs/module-split-design.md`).
- Add CI-like scripts or ensure existing scripts are used consistently.

**Acceptance**
- Clear “what belongs where” rules exist in docs.
- `cargo test -p siumai-core --lib` stays green.

### M1 — Introduce open `providerOptions` map (compat layer)

Add a provider-id keyed JSON map to request types (language, embedding, image, rerank, speech, transcription).

**Implementation notes**
- Rust representation: requests carry `provider_options_map: ProviderOptionsMap` (re-exported from `siumai-core::types`), and builders expose `with_provider_option(provider_id, json)` helpers.
- Keep the existing typed/legacy options temporarily (if needed) but make the map the preferred path.
- Provide helper APIs:
  - set/get provider options by provider id
  - merge semantics: request overrides builder defaults

**Acceptance**
- At least one provider (OpenAI) reads `providerOptions` and behaves correctly.
- No core enum changes needed to add a new provider option key.

### M2 — Move `standards/*` out of `siumai-core`

Move provider protocol mapping modules out of `siumai-core` into provider-owned code.

Two acceptable paths:

- **Path A (recommended MVP)**: extract the OpenAI-like protocol into `siumai-provider-openai`.
- **Path B (later)**: extract additional shared layers only if reuse proves necessary.

**Acceptance**
- `siumai-core` no longer contains provider-specific mapping modules.
- Providers compile with the same behavior (no user-visible changes).

### M3 — Shared OpenAI-like adapter layer (internal first)

Create a shared OpenAI-like standard layer for vendors to reuse:

- request mapping (`chat/completion/embedding/image/...`)
- SSE/stream parsing and tool-call delta handling
- finish reason & usage normalization

Multiple providers (OpenAI-compatible vendors, Groq/xAI/Minimaxi, etc.) depend on this shared layer.

**Acceptance**
- No duplicated stream parsing logic between `openai` and `openai_compatible`.
- Adding a new OpenAI-like vendor requires only configuration + vendor-specific deltas.

### M4 — Tighten public exports and reduce accidental coupling

- Avoid blanket re-exports from `siumai-providers` and `siumai-registry`.
- Ensure the primary entry points remain:
  - `siumai::prelude::unified::*`
  - `siumai::prelude::extensions::*`
  - `siumai::provider_ext::<provider>::*`
- Keep OpenAI-compatible vendor presets on `SiumaiBuilder` / `LlmBuilder` (e.g. `moonshot()`, `deepseek()`) as ergonomic entry points.
  The canonical underlying form remains `openai().compatible("<vendor>")` for consistency and tooling.

**Acceptance**
- Downstream crates can still use the recommended surface without importing internal modules.
- Public docs point to stable paths; internal paths are not encouraged.
- Internal code and tests avoid deprecated vendor shortcuts (compat remains for downstream users).

### M5 — Registry built-ins remain optional and composable

Strengthen the “registry is abstraction-first” stance:

- `siumai-registry` works without built-in providers by default (already partially done via `builtins`).
- External crates can implement and register providers without pulling built-ins.

**Acceptance**
- `siumai-registry` builds and functions with `default-features = false` and without `builtins`.
- A runnable “no builtins” example exists: `siumai-registry/examples/no_builtins_custom_factory.rs`.
- Documentation exists for external factory injection: `docs/registry-without-builtins.md`.

### M6 — Provider-first crates (post-MVP)

Split provider implementations into provider crates while keeping `siumai-providers` as a thin umbrella.

**Acceptance**
- `siumai-providers` can enable providers via crate features without directly containing provider code.
- At least one provider implementation is successfully extracted into a provider crate (start with OpenAI; Ollama is a good second extraction).
- `cargo nextest run -p siumai --tests --no-default-features --features all-providers` remains green.

## Timeline suggestion (pragmatic)

- Sprint 1: M1 (providerOptions map) + minimal OpenAI integration
- Sprint 2: M2 (move standards) + M3 (openai-like shared adapter) for chat + streaming
- Sprint 3: M3 expanded to embedding/image + M4 export tightening
- Sprint 4: M5 polish + docs + migration notes
  - Sprint 5: M6 provider-first crates extraction

## Risks and mitigations

- **Breaking change risk (provider options)**
  - Mitigation: keep compatibility bridge during M1; deprecate typed enum gradually.
- **Refactor churn**
  - Mitigation: keep each milestone small; keep `test-fast` and `test-smoke` green.
- **Feature matrix explosion**
  - Mitigation: define a “smoke profile” subset (openai, openai-like) and expand gradually.
