# Siumai Modular Architecture Refactor Plan

Status: In Progress (Core/Standards/Provider crates partially extracted)
Owners: Core Maintainers
Last Updated: 2025-11-13

## TL;DR

We will modularize Siumai into multiple crates: a lean `core`, separate `standards` (OpenAI, Anthropic, Gemini), per-provider crates, and a thin aggregator crate. This removes cross-provider coupling (e.g., MiniMaxi inadvertently depending on OpenAI/Anthropic providers), reduces build times, and clarifies extension points.

Key decisions:
- Providers depend only on `core` and the required `standards`, never on other providers.
- “OpenAI-compatible” providers are handled by a dedicated `openai-compatible` provider crate with vendor adapters (OpenRouter, DeepSeek, Together, etc.) guarded by fine‑grained features. No dependency on the `openai` provider crate is introduced.
- Backward compatibility is preserved via the aggregator crate (`siumai`) re‑exporting public APIs.

---

## Current State

Siumai has been partially migrated to the target layout:

- Core & standards:
  - `siumai-core` hosts errors, traits, types, execution, streaming, retry, and shared utils.
  - `siumai-std-openai` and `siumai-std-anthropic` host OpenAI/Anthropic standards (chat/embedding/image/rerank/streaming) and adapters, fully decoupled from concrete providers.
- Providers:
  - `siumai-provider-openai` contains a minimal `OpenAiStandardAdapter`, stateless routing helpers (`chat_path/embedding_path/image_generation_path/image_edit_path/image_variation_path`), OpenAI JSON header construction (`build_openai_json_headers`), and Responses SSE event name constants (`RESPONSES_EVENT_*`).
  - `siumai-provider-openai-compatible` contains adapter/types/registry/helpers for generic OpenAI-compatible providers, including `build_json_headers_with_provider` with a conservative alias policy for OpenRouter.
  - `siumai-provider-anthropic` and `siumai-provider-minimaxi` are scaffolded and will gradually receive migrated implementations.
- Aggregator (`siumai`):
  - Uses feature-gated bridges (`std-openai-external`, `provider-openai-external`, `provider-openai-compatible-external`) to switch between in-crate and external implementations without changing public APIs.
  - `minimaxi` no longer hard-depends on `openai`/`anthropic` provider features; it relies on standards and core.

Key outcomes so far:
- Standards are provider-agnostic and live outside the aggregator.
- OpenAI-compatible no longer depends on the OpenAI provider crate.
- OpenRouter/DeepSeek/SiliconFlow etc. live behind the `siumai-provider-openai-compatible` adapter/registry, with conservative header policies.

---

## Design Principles

1. Single‑direction dependencies: `core` <- `standards` <- `providers` <- `siumai(aggregator)` <- `extras`.
2. Providers never depend on other providers; they depend on standards and core only.
3. Standards implement API shapes (request/response/stream transformers) and expose provider‑specific adapters where needed.
4. Capabilities live in `core` via traits; providers implement them piecemeal.
5. Backward compatibility: keep public APIs stable by re‑exporting from the aggregator crate.

---

## Target Workspace Layout

```
crates/
  siumai-core/                 # traits, types, error, core (ProviderSpec/Context), execution, streaming, retry, utils
  siumai-std-openai/           # OpenAI standard (chat/responses, embedding, image, audio, rerank), adapters
  siumai-std-anthropic/        # Anthropic Messages standard (chat/stream), adapters
  siumai-std-gemini/           # Gemini standard, adapters

  siumai-provider-openai/      # Native OpenAI provider (depends on core + std-openai)
  siumai-provider-anthropic/   # Native Anthropic provider (depends on core + std-anthropic)
  siumai-provider-minimaxi/    # MiniMaxi (chat via std-anthropic; image/audio via std-openai)
  siumai-provider-openai-compatible/
                               # Generic OpenAI-compatible provider; vendor adapters under features (openrouter/deepseek/...)

  siumai/                      # Aggregator (thin): re-exports, builders, feature routing to providers
  siumai-extras/               # Optional: telemetry, server adapters, schema, MCP (unchanged)
```

Dependency graph (acyclic):

```
core ← standards ← providers ← siumai(aggregator) ← extras
```

---

## Crate Responsibilities

### siumai-core
- Capability traits: Chat/Embedding/Rerank/Audio/Vision/Image/Files/Moderation/etc.
- ProviderSpec/ProviderContext, transformer bundle types.
- Execution (HTTP client, headers, middleware/interceptors, retry policy, stream handling).
- Types (requests/responses/options), errors, utils.

### siumai-std-*
- Pure “standards”: request/response/stream transformers for a given API shape.
- Adapter interfaces to inject provider-specific quirks (headers, endpoints, SSE event normalization, prompt caching flags, etc.).

### siumai-provider-*
- Implements provider builders, defaults (baseURL), model constants, capability selection.
- Picks appropriate standards and provides adapters for provider diffs.
- No cross‑provider dependencies.

### siumai (aggregator)
- Re-export public APIs and builders.
- Map features to provider crates.
- Preserve the current external surface area.

---

## Naming and Packaging (aligned with Vercel AI SDK)

- `siumai-provider-openai` → OpenAI native provider.
- `siumai-provider-anthropic` → Anthropic native provider.
- `siumai-provider-openai-compatible` → Generic OpenAI-compatible provider.
  - Vendor adapters inside this crate (OpenRouter, DeepSeek, Together, SiliconFlow, etc.), toggled via sub-features.
  - Provide convenience constructors (e.g., `openrouter()`, `deepseek()`) that preconfigure baseURL/headers.
  - Keep a generic `with_base_url()` path for arbitrary OpenAI-compatible endpoints.

Rationale: Avoid crate explosion for long-tail vendors while remaining open to spinning out dedicated crates later if a vendor diverges significantly or adds unique capabilities.

---

## OpenAI-Compatible Strategy

Two layers:
1) Generic OpenAI-compatible client (in `siumai-provider-openai-compatible`):
   - Depends on `siumai-core` + `siumai-std-openai`.
   - Accepts custom baseURL and header injection.
   - Exposes a pluggable adapter trait for vendor-specific tweaks (endpoints, headers, JSON shapes, error mapping).

2) Vendor convenience adapters (feature-gated):
   - `feature = "openrouter"`: adds `openrouter()` constructor with defaults and header mapping.
   - `feature = "deepseek"`: adds `deepseek()` constructor; may handle minor request/response differences via the adapter.
   - Additional vendors added as optional features (Together, SiliconFlow, Moonshot, etc.).

No dependency on `siumai-provider-openai` is introduced.

Spin-out criteria for dedicated crates:
- Significant API divergence from OpenAI (custom endpoints beyond trivial mapping).
- Unique capabilities not representable via the OpenAI standard or provider options.
- Maintenance or versioning concerns that benefit from separation.

---

## Feature Flags and Dependencies

Principles:
- Providers’ features never imply other providers.
- Standards are enabled by either the provider that needs them or by direct selection (for expert use).
- Fine-grained features for vendor adapters inside `openai-compatible`.

Examples:
- `minimaxi` feature: depends on `hex` only; pulls `std-anthropic` and `std-openai` as direct dependencies of the provider crate; does not enable `openai`/`anthropic` providers.
- `openai-compatible` feature: enables the generic OpenAI-compatible provider; sub-features enable vendor adapters.

---

## Migration Plan (Phased)

Phase 0: Immediate Hygiene (1–2 PRs)
- Remove cross-provider coupling in features:
  - `minimaxi` no longer enables `openai` provider (keeps `anthropic` temporarily, because Anthropic standard currently reuses provider transformers; will be decoupled in Phase 1/2).
  - Ensure standards modules remain enabled via minimal gates (e.g., `any(feature = "minimaxi", feature = "openai")`) until split.
- Make `openai-compatible` a separate feature flag. Now decoupled from `openai` provider (uses standards/openai transformers + utils); no provider cross-dependency.

Phase 1: Extract Standards (OpenAI, Anthropic) into `siumai-std-*`
- Move `src/standards/openai/*` → `crates/siumai-std-openai`.
- Move `src/standards/anthropic/*` → `crates/siumai-std-anthropic`.
- Adjust imports in providers to point to `siumai-std-*`.

Phase 2: Extract Core
- Move `traits`, `types`, `core`, `execution`, `streaming`, `retry_api`, `utils/cancel` to `crates/siumai-core`.
- Make `siumai-std-*` depend on `siumai-core`.

Phase 3: Extract Provider Crates
- Split OpenAI, Anthropic, MiniMaxi into provider crates.
- Introduce `siumai-provider-openai-compatible` with generic adapter + vendor sub-features.
- Keep `siumai` as aggregator re-exporting the public API.

Phase 4: Documentation and Examples
- Update README and docs to describe “core vs standards vs providers”.
- Add examples per provider crate and a unified example in `siumai`.

Phase 5: Optional Provider Splits
- Consider separate crates for vendors under openai-compatible if divergence grows.

Acceptance criteria per phase:
- All crates `cargo check` under feature matrices.
- No cross-provider dependencies.
- Public APIs in `siumai` remain source-compatible for common flows.

---

## CI and Validation

Matrix checks (non-exhaustive examples):
- Core: `cargo check -p siumai-core` (no features).
- Standards: `cargo check -p siumai-std-openai`, `siumai-std-anthropic`.
- Providers: single-feature checks: `openai`, `anthropic`, `minimaxi`, `openai-compatible`.
- Combinations: `minimaxi` alone; `openai` alone; `anthropic` alone; `openai-compatible + openrouter`; etc.

Use `cargo-hack` to generate feature power sets where appropriate, excluding meta “all-providers”.

---

## Backward Compatibility

- Keep `siumai` crate re-exporting the same user-facing modules (prelude/builders/types), minimizing breaking changes.
- Deprecate old feature combos with warnings and migration hints.
- Maintain examples showing both per-provider and unified clients.

---

## Risks and Mitigations

- Risk: Docs.rs cross-crate discoverability. Mitigation: central docs in `siumai`, deep links to standards/providers.
- Risk: Version skew across crates. Mitigation: workspace unified versioning and CI gating releases.
- Risk: Vendor divergence within `openai-compatible`. Mitigation: start with feature-gated adapters; spin out when necessary.

---

## FAQ

Q: Should OpenRouter and DeepSeek live inside `siumai-provider-openai-compatible`?

A: Yes, initially. Put them as vendor adapters inside `siumai-provider-openai-compatible`, guarded by fine‑grained features (e.g., `openrouter`, `deepseek`). This avoids crate explosion, keeps maintenance centralized, and leverages the OpenAI standard adapter. If a vendor diverges (unique endpoints/capabilities) beyond what the OpenAI standard can model, we can later spin it out into a dedicated provider crate without breaking the generic path.

Q: Can traits and adapters still extend cleanly across crates?

A: Yes. All capability traits, ProviderSpec, and transformer traits live in `siumai-core` and are object‑safe via `async-trait`. Standards and providers implement them in their own crates with no need to depend on other providers.

---

## Post-Refactor Usage Sketch

```rust
// Aggregator crate (unchanged developer experience)
use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // OpenAI
    let openai = LlmBuilder::new().openai().api_key("...").build().await?;

    // OpenAI-compatible (OpenRouter)
    let or = LlmBuilder::new()
        .openai_compatible()
        .openrouter()        // available with feature = "openrouter"
        .api_key("...")
        .build().await?;

    // MiniMaxi (chat via Anthropic standard, image/audio via OpenAI standard)
    let mm = LlmBuilder::new().minimaxi().api_key("...").build().await?;

    Ok(())
}
```

---

## Immediate Next Steps (Suggested)

1) Decouple `minimaxi` feature from `openai`/`anthropic` providers (standards only).
2) Make `openai-compatible` independent and avoid gating under `feature = "openai"`.
3) Add CI jobs for single‑provider builds to prevent regressions.
