# Groq Browser Search Alignment - TODO

Last updated: 2026-04-14

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Reference audit

- [x] Audit `repo-ref/ai/packages/groq/src/tool/browser-search.ts`.
- [x] Audit `repo-ref/ai/packages/groq/src/groq-prepare-tools.ts`.
- [x] Audit `repo-ref/ai/packages/groq/src/groq-browser-search-models.ts`.
- [x] Record the supported-model contract and the exact unsupported warning details.

## Track B - Shared/public surface

- [x] Add a stable shared `groq.browser_search` provider-defined tool factory.
- [x] Make the dynamic `provider_defined_tool(...)` resolver understand `groq.browser_search`.
- [x] Expose Groq provider-tool factories on `provider_ext::groq::{tools, provider_tools}`.
- [x] Add public-surface compile guards for the new Groq tool modules.

## Track C - Runtime wiring

- [x] Add a provider-owned Groq middleware that injects the native `browser_search` wire tool.
- [x] Keep mixed function tools intact instead of replacing the serialized tool array.
- [x] Preserve `tool_choice` when browser search is the only tool.
- [x] Suppress the generic compat warning for `groq.browser_search` and emit Groq-owned
  unsupported-model warnings instead.

## Track D - Regression coverage

- [x] Add shared tool-factory tests.
- [x] Add compat warning allowlist tests behind `siumai-provider-openai-compatible`'s
  `openai-standard` feature.
- [x] Add Groq runtime tests for supported and unsupported models.
- [x] Add Groq runtime tests for mixed tools and `tool_choice: required`.

## Track E - Docs and maintenance

- [x] Create a dedicated workstream folder in `docs/workstreams/`.
- [x] Update `Unreleased` changelog sections for the shared/facade/provider/runtime changes.
- [x] Split the broader `@ai-sdk/groq` typed/model parity follow-up into its own
  `docs/workstreams/groq-package-surface-alignment/` workstream.
- [-] Continue auditing the rest of the Groq AI SDK surface inside this browser-search workstream.
  - moved to the dedicated `groq-package-surface-alignment` workstream to keep runtime tool
    parity separate from wider package-surface parity
- [-] Introduce a fake private scratch field on `ChatRequest` just to shuttle provider-owned tool
  state through the compat runtime.
  - rejected because the JSON-body middleware hook plus compat warning allowlist is smaller,
    clearer, and does not risk leaking transient provider state into the public request surface
