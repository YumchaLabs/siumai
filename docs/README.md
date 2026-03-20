# Docs Index

This folder is intentionally organized by **concern** to keep the Alpha.5 refactor documentation navigable.

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

- Beta.6 migration guide: `docs/migration/migration-0.11.0-beta.6.md`
- Beta.5 migration guide: `docs/migration/migration-0.11.0-beta.5.md` (historical; split-crate breaking changes)

## Roadmap

- Next steps: `docs/roadmap/next-steps.md`
- MVP roadmap: `docs/roadmap/roadmap-mvp.md`

## Workstreams

- Fearless refactor (design + TODOs + milestones): `docs/workstreams/fearless-refactor/`
- Protocol bridge + gateway runtime: `docs/workstreams/protocol-bridge-gateway/`
  - covers the hybrid bridge strategy: normalized backbone + selected direct bridges + gateway policy
