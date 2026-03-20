# Fearless Refactor V4 - Release Notes Draft

Last updated: 2026-03-20

## Summary

This release completes the V4 architecture pivot for `siumai`.

The main outcome is a clearer, more stable public story:

- model-family-first execution is now the default architecture
- registry-first and config-first are now the canonical construction paths
- builders remain available, but only as ergonomic wrappers
- provider-specific request/response complexity stays inside provider-owned crates

## Highlights

### Family-model-first architecture

- family traits now act as the primary execution contract across text, embedding, image, rerank,
  speech, and transcription paths
- registry handles now behave like family-model objects rather than temporary generic-client
  adapters
- `LlmClient` remains available, but it is no longer the architectural center of the system

### Public API convergence

- the recommended public path is now explicit:
  1. registry-first
  2. config-first
  3. builder convenience last
- docs, examples, and facade exports now tell the same construction story
- provider-owned extension surfaces remain available through `provider_ext::<provider>`

### Provider contract hardening

Recent validation coverage for the V4 line includes:

- OpenAI facade contract sweep: `144 passed, 1 skipped`
- OpenAI targeted live smoke (`gpt-5.2` text + embeddings): passed
- Anthropic / Google / Vertex contract sweep: `238 passed, 0 skipped`
- Groq / xAI / DeepSeek / Ollama contract sweep: `275 passed, 0 skipped`

## Notable fixes

### OpenAI Responses streaming parity

- native OpenAI `/responses` streaming no longer sends Chat Completions-only
  `stream_options.include_usage`
- the issue was discovered through real `gpt-5.2` live validation
- targeted regression coverage now locks the route-specific behavior

### Ollama request-option parity

- Ollama request-side escape hatches now accept both:
  - the canonical flattened `providerOptions["ollama"]` shape
  - the historical nested `providerOptions["ollama"].extra_params` shape
- registry embedding handles now preserve provider-specific request config on
  `embed_with_config(...)` request-aware paths before falling back to the generic family-model
  route

## Migration guidance

### Recommended construction order

Prefer the following order for new code:

1. registry-first for application-level setup
2. config-first for provider-owned direct usage
3. builder APIs only for convenience or migration

### What users do not need to change

- users do not need to rename existing spec types just to mirror AI SDK naming
- users do not need to stop using builders immediately
- users do not need to migrate to low-level provider clients unless they need provider-owned
  features

### What users should update over time

- prefer stable family helpers such as `text`, `embedding`, `image`, `rerank`, `speech`, and
  `transcription`
- treat `provider_ext::<provider>` as the long-term home for provider-specific knobs
- treat compatibility/vendor-view surfaces as explicit compatibility layers, not the default path

## Known scope boundaries

The following remain intentionally outside the V4 release scope:

- a new Stable hosted-search surface across providers
- full symmetry across every provider package
- immediate builder removal
- naming-only churn to mirror external SDK terminology

## Recommended post-release follow-up

After the V4 release, the next focused work should use:

- `typed-metadata-boundary-matrix.md` as the response-side typing backlog
- `provider-capability-alignment-matrix.md` as the provider-capability backlog

## Short changelog version

V4 completes the shift to a family-model-first architecture, keeps builders as convenience-only
wrappers, aligns the public story around registry-first and config-first construction, and closes
major provider parity gaps across OpenAI, Anthropic, Google/Vertex, Groq, xAI, DeepSeek, and
Ollama with broad no-network contract coverage plus targeted live validation.
