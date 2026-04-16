# DeepInfra Unified Provider Surface - Milestones

Last updated: 2026-04-10

## DIP-M0 - Scope locked

Acceptance criteria:

- the AI SDK DeepInfra reference surface is identified
- the Siumai drift between compat preset and first-class provider story is written down explicitly
- the workstream chooses a canonical provider identity

Status: completed

## DIP-M1 - First-class provider identity exists

Acceptance criteria:

- `deepinfra` is registered as a built-in provider id
- native metadata exists for DeepInfra
- stable/provider catalog layers no longer downgrade DeepInfra to `Custom("deepinfra")`

Current state:

- registry/provider wiring now includes `deepinfra`
- native metadata now declares the unified DeepInfra capability set
- `ProviderType::DeepInfra` is now part of the stable/provider catalog path

Status: completed

## DIP-M2 - Unified runtime surface exists

Acceptance criteria:

- one built `deepinfra` client exposes chat/completion/embedding plus provider-owned image support
- text families reuse the shared OpenAI-compatible runtime
- image generation/edit use the provider-owned DeepInfra routes

Current state:

- `DeepInfraProviderFactory` now aggregates compat text-family clients with a provider-owned image
  client
- `DeepInfraUnifiedClient` delegates text families and image families behind one provider id
- public/contract tests now pin `/openai/chat/completions`, `/inference/{model}`, and
  `/openai/images/edits`

Status: completed

## DIP-M3 - Semantic/runtime edge cases are aligned

Acceptance criteria:

- DeepInfra-specific usage drift is normalized before entering the stable `Usage` layer
- hybrid registry registration does not overwrite native metadata with compat metadata

Current state:

- provider-aware OpenAI-compatible usage parsing now corrects DeepInfra reasoning/completion totals
- compat adapter registration now merges into native registry records instead of overwriting them

Status: completed

## DIP-M4 - Docs and audit handoff are complete

Acceptance criteria:

- workstream docs describe the chosen DeepInfra architecture
- structural-alignment docs mention DeepInfra as closed rather than open
- changelog `Unreleased` sections capture the user-facing implications

Current state:

- dedicated DeepInfra workstream docs now exist
- structural-alignment todo/matrix entries now describe DeepInfra as a closed provider-surface gap
- root and crate changelogs now include the DeepInfra alignment pass
- public `provider_ext::deepinfra` model constants plus provider-catalog curated model reuse now
  keep the facade and registry on the same audited DeepInfra subset
- the shared compat DeepInfra preset now also advertises `completion` explicitly in static
  metadata, so the audited config/runtime/provider story no longer depends on inferred completion
  support alone

Status: completed
