# Docs Index

This folder is intentionally organized by **concern**. Prefer the migration guides, architecture
documents, release process, and active workstreams for current planning.

## Alignment (Vercel parity + provider audits)

- Main checklist: `docs/alignment/provider-implementation-alignment.md`
- High-level matrix: `docs/alignment/provider-feature-alignment.md`
- Vercel fixture parity: `docs/alignment/vercel-ai-fixtures-alignment.md`
- Streaming bridge notes: `docs/alignment/streaming-bridge-alignment.md`
- Official API audits: `docs/alignment/official/*`

## Architecture (crate split + public surface)

- Refactor plan: `docs/architecture/architecture-refactor-plan.md`
- Split design (ownership + dependencies): `docs/architecture/module-split-design.md`
- Provider extensions: `docs/architecture/provider-extensions.md`

## Migration

- Beta.7 migration guide: `docs/migration/migration-0.11.0-beta.7.md`
- Beta.6 migration guide: `docs/migration/migration-0.11.0-beta.6.md`
- Beta.5 migration guide: `docs/migration/migration-0.11.0-beta.5.md` (historical; split-crate breaking changes)

## Operations

- Release process: `docs/releasing.md`

## Workstreams

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
