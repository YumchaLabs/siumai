# MoonshotAI Package Surface Alignment - TODO

Last updated: 2026-04-13

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Canonical identity

- [x] Audit the AI SDK MoonshotAI package boundary in `repo-ref/ai/packages/moonshotai`.
- [x] Make `moonshotai` the canonical public/runtime id.
- [x] Retain `moonshot` only as a hidden migration alias.
- [x] Sweep public examples/docs from `moonshot` / `.moonshot()` to `moonshotai` /
  `.moonshotai()`.

## Track B - Typed options and wire normalization

- [x] Add typed MoonshotAI language-model options on the provider-owned/public surface.
- [x] Add a typed request-extension helper for `providerOptions["moonshotai"]`.
- [x] Normalize `thinking.budgetTokens -> thinking.budget_tokens` on the canonical compat path.
- [x] Normalize `reasoningHistory -> reasoning_history` on the canonical compat path.
- [x] Canonicalize provider-option keys so legacy `moonshot` input resolves onto `moonshotai`.

## Track C - Model/default surface

- [x] Align curated Moonshot model constants with the audited Kimi K2 + Moonshot V1 subset.
- [x] Align built-in default/base-url metadata with canonical `moonshotai`.
- [x] Reuse the curated Moonshot subset in the provider catalog and public facade.

## Track D - Public boundary guards

- [x] Add public import/runtime guards for `provider_ext::moonshotai::*`.
- [x] Add public-path parity tests for canonical builder/provider/config/registry routing.
- [x] Add URL/base-url alignment coverage for canonical id plus hidden alias fallback.
- [x] Lock the intentional absence of completion and the other non-text families on the MoonshotAI
  wrapper boundary.

## Track E - Docs and migration notes

- [x] Add a dedicated MoonshotAI workstream under `docs/workstreams`.
- [x] Update structural-alignment docs to record the canonical `moonshotai` package boundary.
- [x] Update `CHANGELOG.md` under `Unreleased`.
- [x] Update example/docs references that still implied `.moonshot()` was public.

## Track F - Intentional deferrals

- [-] Expose TypeScript-only `MoonshotAIProviderSettings` or `VERSION` on the Rust facade.
  - deferred intentionally because Rust already uses `MoonshotAIConfig`, builders, and config-first
    constructors as the stable settings story
- [-] Invent a MoonshotAI image/completion/embedding public surface ahead of upstream.
  - rejected intentionally because the audited AI SDK package is chat/language-model-only

## Track G - Remaining follow-up

- [ ] Decide whether the hidden low-level `moonshot` alias should be deleted entirely after
  downstream migration.
- [ ] Re-audit this workstream if `repo-ref/ai/packages/moonshotai` gains a broader provider
  boundary in the future.
