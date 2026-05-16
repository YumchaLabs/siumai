# Docs Index

This folder is intentionally organized by **concern**. Prefer the migration guides, architecture
documents, release process, and active workstreams for current planning.

## Alignment (Vercel parity + provider audits)

- Main checklist: `docs/alignment/provider-implementation-alignment.md`
- Current provider capability matrix: `docs/workstreams/fearless-refactor-v4/provider-capability-alignment-matrix.md`
- Vercel fixture parity: `docs/alignment/vercel-ai-fixtures-alignment.md`
- Streaming bridge notes: `docs/alignment/streaming-bridge-alignment.md`
- Official API audits: `docs/alignment/official/*`

## Architecture (crate split + public surface)

- Split design (ownership + dependencies): `docs/architecture/module-split-design.md`
- Public facade surface: `docs/architecture/public-surface.md`
- Provider extensions: `docs/architecture/provider-extensions.md`
- Registry without built-ins: `docs/architecture/registry-without-builtins.md`

## Migration

- Beta.7 migration guide: `docs/migration/migration-0.11.0-beta.7.md`
- Beta.6 migration guide: `docs/migration/migration-0.11.0-beta.6.md`
- Beta.5 migration guide: `docs/migration/migration-0.11.0-beta.5.md` (historical; split-crate breaking changes)

## Operations

- Release process: `docs/releasing.md`

## Workstreams

- Fearless registry facade construction boundary: `docs/workstreams/fearless-registry-facade-construction-boundary/`
  - centralizes built-in provider factory selection inside `siumai-registry` and keeps facade tests
    from depending on concrete built-in factory structs for normal provider construction
- Fearless core provider alias extraction: `docs/workstreams/fearless-core-provider-alias-extraction/`
  - extracts provider-specific model alias and recommendation logic out of `siumai-core` and into
    registry/provider-owned boundaries
- Fearless boundary hardening: `docs/workstreams/fearless-boundary-hardening/`
  - next fearless-refactor execution track for boundary hardening and removal of unnecessary
    compatibility or redundant code once canonical paths are tested and documented
- Fearless spec/core boundary convergence: `docs/workstreams/fearless-spec-core-boundary-convergence/`
  - tracks the next boundary pass for keeping `siumai-spec` data-only, keeping `siumai-core`
    provider-agnostic, and moving bridge/protocol/provider residue to owning crates
- Fearless ContentPart boundary split: `docs/workstreams/fearless-content-part-boundary-split/`
  - splits the deferred legacy `ContentPart` dual provider-map problem into a dedicated
    compatibility lane with directional request/response adapters
- Fearless vision compatibility removal: `docs/workstreams/fearless-vision-compat-removal/`
  - removes the deprecated dedicated vision compatibility surface in favor of multimodal chat and
    image-family APIs
- Fearless architecture convergence: `docs/workstreams/fearless-architecture-convergence/`
- Current V4 refactor tracking: `docs/workstreams/fearless-refactor-v4/`
- AI SDK structural alignment: `docs/workstreams/ai-sdk-structural-alignment/`
  - covers the next semantic refactor pass against `repo-ref/ai` provider v3/v4 contracts
  - includes the current structural audit for `providerOptions` / `providerMetadata`, stable
    content/stream shapes, and usage convergence
  - prompt/content boundary review: `docs/workstreams/ai-sdk-structural-alignment/prompt-boundary-review.md`
  - runtime consumer parity: `docs/workstreams/ai-sdk-structural-alignment/runtime-consumer-parity.md`
- Protocol bridge + gateway runtime: `docs/workstreams/protocol-bridge-gateway/`
  - covers the hybrid bridge strategy: normalized backbone + selected direct bridges + gateway policy
  - migration note: `docs/workstreams/protocol-bridge-gateway/migration.md`
  - route recipes: `docs/workstreams/protocol-bridge-gateway/route-recipes.md`
- Completed typed stream cleanup: `docs/workstreams/typed-stream-only/todo.md`
