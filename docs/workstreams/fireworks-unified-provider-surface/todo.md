# Fireworks Unified Provider Surface - TODO

Last updated: 2026-04-08

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Canonical provider identity

- [x] Audit the AI SDK Fireworks provider surface in `repo-ref/ai/packages/fireworks/src/*`.
- [x] Record the old Siumai drift between first-class identity and actual unified runtime surface.
- [x] Keep `fireworks` as the canonical built-in provider id on the stable/public/catalog layers.
- [x] Align native metadata and provider catalog output with the unified Fireworks capability set.

## Track B - Runtime architecture

- [x] Reuse the shared OpenAI-compatible runtime for chat/completion/embedding/transcription.
- [x] Add a provider-owned Fireworks image client for generation/edit.
- [x] Support the audited Fireworks image route split:
  - sync `/workflows/{model}/text_to_image`
  - async `/workflows/{model}` + `/get_result`
  - legacy `/image_generation/{model}`
- [x] Aggregate those lanes inside one registry-layer unified provider factory/client.
- [-] Build a dedicated provider-owned Fireworks text crate.
  - rejected for now because the shared compat runtime already matches the audited Fireworks text
    surface with much lower maintenance cost

## Track C - Defaults and request semantics

- [x] Add the audited Fireworks image family default (`accounts/fireworks/models/flux-1-dev-fp8`)
  to the shared compat family-default table.
- [x] Add a Kontext edit fallback model for unified Fireworks image edits.
- [x] Preserve `providerOptions.fireworks` precedence on provider-owned image requests.
- [x] Match the audited Fireworks image warning/error semantics closely enough for the known
  `size`, `aspectRatio`, `mask`, multi-image, and async-poll cases.
- [x] Normalize Fireworks compat chat request shaping for:
  - `thinking.budgetTokens`
  - `reasoningHistory`
  - Fireworks-only reasoning-effort level down-mapping

## Track D - Public facade and regression coverage

- [x] Expose `Provider::fireworks()` / `Siumai::builder().fireworks()` on the unified builder path.
- [x] Add registry contract tests for Fireworks unified capability and provider-owned image routes.
- [x] Add public-path tests for provider-owned default image routing and async Kontext edit routing.
- [x] Add compile/public-surface coverage for the promoted Fireworks facade entry points.

## Track E - Docs and follow-up

- [x] Create a dedicated Fireworks workstream folder under `docs/workstreams/`.
- [x] Update structural-alignment docs that previously treated Fireworks image support as an open gap.
- [x] Update unreleased changelog sections instead of writing release notes.
- [~] Keep auditing Fireworks typed options/model coverage against `repo-ref/ai/packages/fireworks/src/*`.
  - [x] Promote typed Fireworks language-model options for `thinking` and `reasoningHistory`
    through `provider_options.fireworks`, `FireworksChatRequestExt`, and `siumai::provider_ext::fireworks`.
  - [x] Lock typed-options-to-wire parity for the audited chat quirks
    (`thinking.budgetTokens`, `reasoningHistory`, Fireworks reasoning-effort down-mapping).
  - [x] Promote the audited AI SDK Fireworks curated model subset into provider-owned Rust
    constants (`chat` / `completion` / `embedding` / `image`) and reuse that set for public catalog output.
  - [~] Curated model-list drift still needs periodic review as upstream Fireworks package models change.
