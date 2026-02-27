# Fearless Refactor Workstream — Design

Last updated: 2026-02-27

## Why this workstream exists

The `siumai` workspace reached a point where internal module boundaries were too blurry:

- Spec-level types (requests/responses/messages/tools) lived together with runtime code.
- Provider resolution rules were duplicated across builder/registry paths.
- “Just add one more provider” often meant touching multiple crates/files.

This workstream is a **fearless refactor**: we accept breaking changes internally (and, when needed, publicly)
to reduce coupling and simplify long-term maintenance.

## Design principles

1. **Crate ownership is the architecture**
   - If two concerns change at different rates, they should not be owned by the same crate.
2. **Single source of truth for routing**
   - Provider id normalization and inference rules must live in one place.
   - Provider id constants + variant parsing should be centralized (avoid duplicated strings).
3. **Factories own provider-specific defaults**
   - API key/env/base_url defaults must live in `ProviderFactory` implementations, not in a top-level match.
4. **Stable surfaces via re-export shims**
   - We can move internals while keeping downstream imports stable during transition periods.

## Target layering (medium granularity)

We intentionally avoid Vercel AI SDK’s very fine package granularity. The goal is a practical Rust layout:

1. **`siumai-spec` (spec / provider-agnostic types)**
   - Requests/responses/messages/tool schemas
   - Lightweight configs referenced by types (e.g. telemetry config)
   - Error types shared across crates (dependency-light)

2. **`siumai-core` (runtime / execution)**
   - HTTP runtime, streaming, retry, middleware
   - Auth helpers, interceptors, tracing glue
   - Re-export shims for `siumai-spec` during migration

3. **`siumai-registry` (routing / factories / handles)**
   - Registry handle API (`provider:model`) and caching
   - `ProviderFactory` trait + built-in factories behind features
   - Provider id resolver (normalization + conservative inference)

4. **Provider crates**
   - Protocol mapping, typed provider metadata/options, client implementation

## Key decisions (already landed)

### D1: Introduce `siumai-spec`

Move provider-agnostic types/tools/errors into `siumai-spec`, and keep `siumai-core` as the runtime owner.

- Benefit: reduces coupling and future split cost.
- Trade-off: adds one more crate (acceptable).

### D2: Centralize provider resolution

Create a single resolver in `siumai-registry` to handle:

- alias normalization (`google` → `gemini`, etc.)
- `provider_id` → `ProviderType` mapping
- conservative inference from model id prefix (only when safe)

### D3: Delegate API key + base_url defaults to factories

Remove env-var and vendor-specific resolution from the unified build “router”.

- Benefit: each provider factory owns its own rules; the unified layer stays thin.
- Trade-off: more responsibility in factories (which is the correct ownership).

### D4: Keep “metadata-only” providers out of built-ins

Some provider ids are feature-gated and have metadata (capabilities/base_url), but do not yet have
built-in factories (e.g. rerank-only providers). These ids are **reserved** and:

- are not registered into the default built-in catalog
- fail fast when selected by `provider_id` (instead of silently falling back to OpenAI-compatible)

## Invariants

These should remain true throughout the refactor:

- Spec types must not depend on provider protocol crates.
- Provider routing rules are not duplicated across the codebase.
- Registry handles normalize common aliases only when it is safe (avoid surprising custom registries).
- `BuildContext` is the primary “configuration transport” to factories.
- A small set of no-network factory “contract tests” protects shared precedence rules during refactors.

## Known risks

- Moving types may accidentally pull runtime dependencies into `siumai-spec`.
  - Mitigation: feature-gate heavier deps (e.g. `reqwest`) and keep conversions optional.
- Routing compatibility regressions for alias ids / variants.
  - Mitigation: add targeted tests around alias normalization and variant routing.
- Tests that mutate process-global env vars can become flaky when run in parallel.
  - Mitigation: use a shared env lock + guards for env mutation tests.

## References

- `docs/architecture/architecture-refactor-plan.md`
- `docs/architecture/module-split-design.md`
- `docs/architecture/registry-without-builtins.md`
