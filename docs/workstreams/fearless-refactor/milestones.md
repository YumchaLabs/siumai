# Fearless Refactor Workstream — Milestones

Last updated: 2026-02-27

This workstream is tracked by **milestones** with clear acceptance criteria.

## M0 — Baseline and safety net

Acceptance criteria:

- Core crates compile under the common feature sets.
- Key integration tests compile (`--no-run`) for `siumai` and `siumai-registry`.
- No accidental API surface explosion.

Status: ✅ done (ongoing validation)

## M1 — Spec/runtime split (medium granularity)

Acceptance criteria:

- `siumai-spec` exists and owns provider-agnostic types/tools/errors.
- `siumai-core` depends on `siumai-spec` (not the other way around).
- Downstream code can still import types via `siumai-core` re-exports during transition.

Status: ✅ done

## M2 — Single source of truth for provider resolution

Acceptance criteria:

- Provider id alias normalization exists in one module and is used by:
  - unified builder (`SiumaiBuilder::provider_id`)
  - unified build path
  - registry handle (when safe)
- Add tests for alias normalization.

Status: ✅ done

## M3 — Factories own defaults (API key, base_url)

Acceptance criteria:

- Unified build path does not resolve env vars for API keys.
- Unified build path does not hardcode provider default base URLs (except where necessary).
- Each `ProviderFactory` documents and implements its precedence rules.

Status: ✅ done

## M4 — Reduce build match complexity (next milestone)

Acceptance criteria:

- `provider/build.rs` becomes mostly “select factory + build context”.
- Provider-specific wiring lives in factories only.
- Variants (e.g. `openai-chat`, `azure-chat`) are handled without new branching in build.

Status: ⏳ in progress

