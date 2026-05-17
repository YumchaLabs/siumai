# Workstream Index

Last updated: 2026-05-17

This index is the navigation surface for `docs/workstreams/`. It records what can be inferred from existing workstream files; it does not rewrite historical status by assumption.

## Status Rules

- Prefer `WORKSTREAM.json` or `workstream.json` when present.
- Fall back to a top-level `Status:` line in `TODO.md`, `todo.md`, `DESIGN.md`, or `design.md`.
- `unknown` means the lane has no recognized status source. It is not automatically active.
- Before resuming an `unknown` lane, normalize it by reading its docs and adding a machine-readable `WORKSTREAM.json` or opening a new follow-on workstream.
- Do not reopen a closed lane for mechanical cleanup; start from a concrete behavior, provider, public contract, or documentation gap.

## Summary

- Total workstream directories: 65
- Machine-readable status files: 9
- Closed or closed-like lanes: 11
- Active-like lanes: 0
- Unknown legacy lanes: 54

## Inventory

| Workstream | Status | Source | Machine-readable |
| --- | --- | --- | --- |
| `docs/workstreams/ai-sdk-structural-alignment` | unknown | `none` | no |
| `docs/workstreams/anthropic-files-shared-contract-alignment` | unknown | `none` | no |
| `docs/workstreams/anthropic-package-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/anthropic-vertex-package-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/anthropic-vertex-stream-compat-alignment` | unknown | `none` | no |
| `docs/workstreams/bedrock-embedding-alignment` | unknown | `none` | no |
| `docs/workstreams/bedrock-image-alignment` | unknown | `none` | no |
| `docs/workstreams/bedrock-protocol-boundary-cleanup` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/cohere-unified-provider-surface` | unknown | `none` | no |
| `docs/workstreams/completion-family-alignment` | unknown | `none` | no |
| `docs/workstreams/completion-metadata-boundary-convergence` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/data-content-error-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/deepinfra-unified-provider-surface` | unknown | `none` | no |
| `docs/workstreams/fearless-architecture-convergence` | unknown | `none` | no |
| `docs/workstreams/fearless-boundary-hardening` | unknown | `none` | no |
| `docs/workstreams/fearless-content-part-boundary-split` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-core-provider-alias-extraction` | unknown | `none` | no |
| `docs/workstreams/fearless-language-extension-handle-isolation` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-provider-composite-client-isolation` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-refactor` | unknown | `none` | no |
| `docs/workstreams/fearless-refactor-v3` | unknown | `none` | no |
| `docs/workstreams/fearless-refactor-v4` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-registry-facade-construction-boundary` | Closed. All planned registry/facade construction-boundary tasks are complete; remaining | `TODO.md` | no |
| `docs/workstreams/fearless-spec-core-boundary-convergence` | Closed. Remaining broader `ContentPart` replacement work is deferred to a future | `TODO.md` | no |
| `docs/workstreams/fearless-vision-compat-removal` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fireworks-unified-provider-surface` | unknown | `none` | no |
| `docs/workstreams/generate-object-structured-output-alignment` | unknown | `none` | no |
| `docs/workstreams/generate-text-output-alignment` | unknown | `none` | no |
| `docs/workstreams/google-package-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/google-vertex-package-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/google-vertex-typed-option-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/groq-browser-search-alignment` | unknown | `none` | no |
| `docs/workstreams/groq-package-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/language-model-call-options-alignment` | unknown | `none` | no |
| `docs/workstreams/minimaxi-unified-provider-surface` | unknown | `none` | no |
| `docs/workstreams/mistral-package-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/moonshotai-package-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/ollama-unified-provider-surface` | unknown | `none` | no |
| `docs/workstreams/openai-compatible-package-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/openai-typed-option-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/perplexity-package-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/prompt-call-settings-alignment` | unknown | `none` | no |
| `docs/workstreams/prompt-model-message-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/protocol-bridge-gateway` | unknown | `none` | no |
| `docs/workstreams/provider-option-alias-alignment` | unknown | `none` | no |
| `docs/workstreams/provider-settings-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/provider-utils-tooling-runtime-alignment` | unknown | `none` | no |
| `docs/workstreams/request-options-alignment` | unknown | `none` | no |
| `docs/workstreams/shared-data-content-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/shared-type-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/stream-delta-lossless-boundary` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/stream-metadata-parity-hardening` | unknown | `none` | no |
| `docs/workstreams/togetherai-unified-provider-surface` | unknown | `none` | no |
| `docs/workstreams/typed-stream-only` | unknown | `none` | no |
| `docs/workstreams/upload-file-call-boundary-alignment` | unknown | `none` | no |
| `docs/workstreams/upload-file-input-shape-alignment` | unknown | `none` | no |
| `docs/workstreams/upload-file-result-surface-alignment` | unknown | `none` | no |
| `docs/workstreams/upload-file-wrapper-removal-alignment` | unknown | `none` | no |
| `docs/workstreams/upload-skill-provider-wrapper-removal-alignment` | unknown | `none` | no |
| `docs/workstreams/vertex-maas-unified-provider-surface` | unknown | `none` | no |
| `docs/workstreams/video-generate-result-materialization-alignment` | unknown | `none` | no |
| `docs/workstreams/video-model-family-alignment` | unknown | `none` | no |
| `docs/workstreams/video-provider-reference-materialization-alignment` | unknown | `none` | no |
| `docs/workstreams/workstream-status-hygiene` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/xai-package-surface-alignment` | unknown | `none` | no |
