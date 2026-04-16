# Perplexity Package Surface Alignment - Milestones

Last updated: 2026-04-13

## PPSA-M0 - Upstream boundary and drift scope locked

Acceptance criteria:

- the audited AI SDK Perplexity reference files are identified
- the package is explicitly classified as chat/language-model-only
- the remaining local drift is narrowed to the typed option/wire boundary

Current state:

- `repo-ref/ai/packages/perplexity/src/index.ts` and `perplexity-provider.ts` are the main
  package-boundary reference files
- the upstream package is still chat-only and intentionally rejects non-text families
- the remaining local drift was narrowed to `PerplexityOptions` using wire snake_case on the
  public typed surface

Status: completed

## PPSA-M1 - Public typed option surface aligned

Acceptance criteria:

- public `PerplexityOptions` serialization uses AI SDK-style camelCase
- legacy snake_case input aliases remain accepted
- request-extension helpers keep writing onto the canonical `providerOptions["perplexity"]` root

Current state:

- `PerplexityOptions` and `PerplexityWebSearchOptions` now serialize with camelCase field names
- serde aliases still accept the older snake_case input keys
- `PerplexityChatRequestExt` continues to attach provider-owned typed options under the canonical
  `perplexity` provider root

Status: completed

## PPSA-M2 - Wire normalization and precedence aligned

Acceptance criteria:

- shared compat request shaping explicitly lowers Perplexity camelCase fields to wire snake_case
- nested `webSearchOptions` fields normalize onto `web_search_options`
- public camelCase typed options win over legacy snake_case aliases when both are present

Current state:

- shared compat normalization now rewrites all known Perplexity public typed fields onto the wire
  contract explicitly
- nested `searchContextSize` / `userLocation` also normalize at the same boundary
- regression coverage now proves camelCase typed values survive even when older snake_case aliases
  are present alongside them

Status: completed

## PPSA-M3 - Docs and regression coverage landed

Acceptance criteria:

- a dedicated workstream documents the Perplexity wrapper contract
- structural audit docs mention the public typed option/wire split explicitly
- changelog notes exist under `Unreleased`

Current state:

- dedicated `docs/workstreams/perplexity-package-surface-alignment/` docs now exist
- structural alignment docs now record that Perplexity public typed options stay camelCase while
  the compat boundary owns the snake_case wire mapping
- `CHANGELOG.md` now records the public typed surface and wire-normalization fix under
  `Unreleased`

Status: completed

## PPSA-M4 - Future re-audit trigger recorded

Acceptance criteria:

- future cleanup no longer depends on rediscovering why the current boundary exists

Current state:

- the only expected future trigger is upstream AI SDK changing the current Perplexity package
  boundary
- TypeScript-only package exports remain intentionally deferred

Status: in progress
