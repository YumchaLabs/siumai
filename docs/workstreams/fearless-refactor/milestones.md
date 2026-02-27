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

Status: ✅ done

Notes:

- `siumai-registry/src/provider/build.rs` now follows a “build `BuildContext` + select `ProviderFactory`” flow.
- Provider variants are routed via `provider_id` (e.g. `openai-chat`, `openai-responses`, `azure-chat`).

## M5 — Builder routing key simplification

Acceptance criteria:

- Unified builder no longer stores redundant `provider_type` state for routing.
- Routing is driven by `provider_id` (including variants like `openai-chat`).
- Keep any remaining `ProviderType` usage limited to introspection/capabilities, not routing.

Status: ✅ done

## M6 — Resolver cleanup (provider_id-first helpers)

Acceptance criteria:

- Provider inference can return a canonical `provider_id` directly.
- OpenAI-compatible behaviors use `provider_id` predicates (no `ProviderType` routing helpers).
- Keep backwards-compatible inference API only as a thin wrapper (if needed).

Status: ✅ done

## M7 — Factory contract tests (precedence rules)

Acceptance criteria:

- Add a small set of no-network “contract tests” for `ProviderFactory` implementations.
- Validate BuildContext precedence (at least):
  - `ctx.http_client` overrides `ctx.http_config`
  - `ctx.api_key` overrides env fallback (where applicable)
  - `ctx.base_url` overrides defaults (where applicable)

Status: ✅ done
