# Perplexity Package Surface Alignment - TODO

Last updated: 2026-04-13

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Boundary audit

- [x] Audit the AI SDK Perplexity package boundary in `repo-ref/ai/packages/perplexity`.
- [x] Confirm that the public wrapper remains chat/language-model-only.
- [x] Narrow the remaining local drift to the typed option/wire boundary.

## Track B - Public typed options

- [x] Move `PerplexityOptions` serialization onto AI SDK-style camelCase.
- [x] Move `PerplexityWebSearchOptions` serialization onto AI SDK-style camelCase.
- [x] Keep snake_case option aliases accepted for backward-compatible input.
- [x] Keep `PerplexityChatRequestExt` on the canonical `providerOptions["perplexity"]` root.

## Track C - Wire normalization

- [x] Add explicit Perplexity option normalization on the shared compat boundary.
- [x] Normalize nested `webSearchOptions.*` fields onto `web_search_options.*`.
- [x] Make camelCase public typed options win over legacy snake_case aliases.
- [x] Preserve unknown extra Perplexity parameters as passthrough fields.

## Track D - Validation

- [x] Add typed option serialization/alias regression coverage.
- [x] Add request-extension regression coverage for the canonical public typed surface.
- [x] Add compat normalization regression coverage for the final wire body.
- [x] Keep public-path/provider-runtime request parity coverage green.

## Track E - Docs and migration notes

- [x] Add a dedicated Perplexity workstream under `docs/workstreams`.
- [x] Update structural-alignment docs to record the public typed option/wire split.
- [x] Update `CHANGELOG.md` under `Unreleased`.

## Track F - Intentional deferrals

- [-] Expose TypeScript-only `PerplexityProviderSettings` or `VERSION` on the Rust facade.
  - deferred intentionally because Rust already uses `PerplexityConfig`, builders, and config-first
    constructors as the stable settings story
- [-] Invent non-text families on the Perplexity wrapper boundary.
  - rejected intentionally because the audited AI SDK package remains language-model-only

## Track G - Remaining follow-up

- [ ] Re-audit this workstream if `repo-ref/ai/packages/perplexity` gains a broader provider
  boundary in the future.
