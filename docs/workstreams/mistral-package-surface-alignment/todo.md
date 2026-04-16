# Mistral Package Surface Alignment - TODO

Last updated: 2026-04-13

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Boundary audit

- [x] Audit the AI SDK Mistral package boundary in `repo-ref/ai/packages/mistral`.
- [x] Confirm the audited family split is chat/language-model + embedding.
- [x] Confirm completion/image stay intentionally unsupported on the wrapper boundary.

## Track B - Public/runtime surface

- [x] Keep `Provider::mistral()` / `Siumai::builder().mistral()` on the audited package boundary.
- [x] Keep registry/public-path coverage for chat/chat-stream/embedding.
- [x] Keep completion-family rejection coverage on the same wrapper boundary.

## Track C - Typed options and wire normalization

- [x] Expose typed Mistral language-model options on the provider-owned/public surface.
- [x] Keep public typed option serialization camelCase.
- [x] Normalize known Mistral public typed fields onto the provider wire contract.
- [x] Keep snake_case aliases accepted where migration compatibility matters.

## Track D - Model/default/catalog surface

- [x] Reuse curated Mistral chat constants from the audited package subset.
- [x] Reuse `mistral-embed` as the explicit embedding default.
- [x] Keep provider catalog/public tests aligned to the same package-owned constants.

## Track E - Docs and migration notes

- [x] Add a dedicated Mistral workstream under `docs/workstreams`.
- [x] Update structural-alignment docs to record the explicit Mistral package boundary.
- [x] Update `CHANGELOG.md` under `Unreleased`.

## Track F - Intentional deferrals

- [-] Expose TypeScript-only `MistralProviderSettings` or `VERSION` on the Rust facade.
  - deferred intentionally because Rust already uses `MistralConfig`, builders, and config-first
    constructors as the stable settings story
- [-] Invent completion/image public families on the Mistral wrapper boundary.
  - rejected intentionally because the audited AI SDK package does not expose them

## Track G - Remaining follow-up

- [ ] Re-audit this workstream if `repo-ref/ai/packages/mistral` changes its package boundary in
  the future.
