# Workstream Index

Last updated: 2026-05-18

This index is the navigation surface for `docs/workstreams/`. It records what can be inferred from existing workstream files; it does not rewrite historical status by assumption.

## Status Rules

- Prefer `WORKSTREAM.json` or `workstream.json` when present.
- Fall back to a top-level `Status:` line in `TODO.md`, `todo.md`, `DESIGN.md`, or `design.md`.
- `unknown` means the lane has no recognized status source. It is not automatically active.
- `superseded` means the lane is historical context; resume through its listed successor or a new
  narrow workstream, not by reopening the legacy TODO list.
- Closed-like counts include `closed`, `completed`, and `superseded`; deferred lanes are tracked
  separately.
- Before resuming an `unknown` lane, normalize it by reading its docs and adding a machine-readable `WORKSTREAM.json` or opening a new follow-on workstream.
- Do not reopen a closed lane for mechanical cleanup; start from a concrete behavior, provider, public contract, or documentation gap.

## Summary

- Total workstream directories: 71
- Machine-readable status files: 71
- Closed or closed-like lanes: 69
- Active-like lanes: 0
- Deferred lanes: 2
- Unknown legacy lanes: 0

## Inventory

| Workstream | Status | Source | Machine-readable |
| --- | --- | --- | --- |
| `docs/workstreams/ai-sdk-provider-interface-convergence` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/ai-sdk-structural-alignment` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/anthropic-files-shared-contract-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/anthropic-package-surface-alignment` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/anthropic-vertex-package-surface-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/anthropic-vertex-stream-compat-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/bedrock-embedding-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/bedrock-image-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/bedrock-protocol-boundary-cleanup` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/cohere-unified-provider-surface` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/completion-family-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/completion-metadata-boundary-convergence` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/data-content-error-surface-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/deepinfra-unified-provider-surface` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/deepseek-package-surface-alignment` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-architecture-convergence` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-boundary-hardening` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-content-part-boundary-split` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-core-provider-alias-extraction` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-language-extension-handle-isolation` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-provider-composite-client-isolation` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-refactor` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-refactor-v3` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-refactor-v4` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-registry-facade-construction-boundary` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-spec-core-boundary-convergence` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fearless-vision-compat-removal` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/fireworks-unified-provider-surface` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/generate-object-structured-output-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/generate-text-output-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/google-interactions-runtime-alignment` | completed | `WORKSTREAM.json` | yes |
| `docs/workstreams/google-package-surface-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/google-vertex-package-surface-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/google-vertex-typed-option-surface-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/groq-browser-search-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/groq-package-surface-alignment` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/language-model-call-options-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/minimaxi-unified-provider-surface` | deferred | `WORKSTREAM.json` | yes |
| `docs/workstreams/mistral-package-surface-alignment` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/moonshotai-package-surface-alignment` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/ollama-unified-provider-surface` | deferred | `WORKSTREAM.json` | yes |
| `docs/workstreams/openai-compatible-package-surface-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/openai-compatible-reasoning-policy-alignment` | completed | `WORKSTREAM.json` | yes |
| `docs/workstreams/openai-compatible-usage-policy-alignment` | completed | `WORKSTREAM.json` | yes |
| `docs/workstreams/openai-typed-option-surface-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/perplexity-package-surface-alignment` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/prompt-call-settings-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/prompt-model-message-surface-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/protocol-bridge-gateway` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/provider-option-alias-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/provider-settings-surface-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/provider-surface-second-pass` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/provider-utils-tooling-runtime-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/request-options-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/shared-data-content-surface-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/shared-type-surface-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/stream-delta-lossless-boundary` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/stream-metadata-parity-hardening` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/togetherai-unified-provider-surface` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/typed-stream-only` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/upload-file-call-boundary-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/upload-file-input-shape-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/upload-file-result-surface-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/upload-file-wrapper-removal-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/upload-skill-provider-wrapper-removal-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/vertex-maas-unified-provider-surface` | superseded | `WORKSTREAM.json` | yes |
| `docs/workstreams/video-generate-result-materialization-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/video-model-family-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/video-provider-reference-materialization-alignment` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/workstream-status-hygiene` | closed | `WORKSTREAM.json` | yes |
| `docs/workstreams/xai-package-surface-alignment` | closed | `WORKSTREAM.json` | yes |
