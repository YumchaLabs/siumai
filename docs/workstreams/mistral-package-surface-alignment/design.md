# Mistral Package Surface Alignment - Design

Last updated: 2026-04-13

## Problem

Compared with `repo-ref/ai/packages/mistral`, Siumai had already landed most of the runtime/public
surface needed for the Mistral wrapper path:

- canonical `mistral` is already a first-class provider identity
- top-level builder/provider/config/registry paths already keep the audited package boundary:
  chat + chat-stream + embedding are available, while completion stays intentionally unsupported
- curated chat and embedding model constants already follow the audited AI SDK subset
- typed `MistralChatOptions` / `MistralLanguageModelOptions` already exist on the provider-owned
  surface and normalize onto the Mistral wire contract

What was still missing was documentation clarity and an explicit package-boundary record in
`docs/workstreams`, comparable to the workstreams already added for `moonshotai` and `perplexity`.

Without that dedicated record, the current Mistral boundary was easy to rediscover incorrectly as
"just another generic OpenAI-compatible preset", even though the audited upstream package has a
more specific public contract:

- language model support
- embedding model support
- no shared completion model surface
- no image model surface

## Goals

- Record the audited `@ai-sdk/mistral` package boundary explicitly.
- Keep Siumai aligned on the same public/runtime split:
  chat/language-model + embedding, but not completion/image.
- Document the existing typed option lane and wire normalization rules.
- Document the intentional Rust-side deferrals for TypeScript-only package exports.

## Non-goals

- Do not invent a native Mistral runtime outside the shared OpenAI-compatible stack.
- Do not invent `completionModel()` or `imageModel()` support on the Mistral wrapper boundary.
- Do not mirror TypeScript-only exports such as `MistralProviderSettings` or `VERSION` on the Rust
  facade.

## Chosen design

### 1. Keep the audited package boundary: chat plus embedding

After re-checking:

- `repo-ref/ai/packages/mistral/src/index.ts`
- `repo-ref/ai/packages/mistral/src/mistral-provider.ts`
- `repo-ref/ai/packages/mistral/src/mistral-chat-options.ts`

the upstream package boundary is:

- `languageModel(modelId)` / `chat(modelId)`
- `embedding(modelId)` / `embeddingModel(modelId)`
- no completion model surface
- no image model surface

Siumai mirrors that same split.

### 2. Keep the shared OpenAI-compatible runtime

Mistral remains a provider-owned wrapper over the shared OpenAI-compatible runtime.

That matches the current AI SDK architecture: the package owns the public surface and typed
options, but transport execution still reuses the shared chat-completions/embeddings machinery.

### 3. Public typed options stay package-level

The public Rust surface already exposes:

- `MistralChatOptions`
- `MistralLanguageModelOptions`
- `MistralReasoningEffort`
- `MistralChatRequestExt`

These types mirror the audited AI SDK option lane:

- `safePrompt`
- `documentImageLimit`
- `documentPageLimit`
- `structuredOutputs`
- `strictJsonSchema`
- `parallelToolCalls`
- `reasoningEffort`

### 4. Wire normalization remains explicit

The compat request boundary owns the lowering from public typed options onto Mistral's wire keys:

- `safePrompt -> safe_prompt`
- `documentImageLimit -> document_image_limit`
- `documentPageLimit -> document_page_limit`

The public typed package shape stays camelCase, while the transport layer owns the final
provider-specific wire contract.

### 5. Curated model/default coverage stays package-owned

The public/constants/catalog story is aligned to the audited Mistral package subset:

- curated chat models such as `mistral-large-latest`, `mistral-small-latest`,
  `magistral-medium-latest`
- curated embedding support through `mistral-embed`
- explicit embedding default reuse on the shared compat family-default table

That keeps catalog/default behavior package-owned instead of scattering handwritten strings.

## Validation

This workstream is currently locked by:

- typed option serialization tests in
  `siumai-provider-openai-compatible/src/provider_options/mistral.rs`
- compat normalization tests in
  `siumai-protocol-openai/src/standards/openai/compat/spec.rs`
- provider/runtime/public-path parity tests in
  `siumai/tests/provider_public_path_parity_test.rs`
- URL alignment tests in
  `siumai/tests/mistral_openai_compat_url_alignment_test.rs`

## Remaining follow-up

- Re-audit if the upstream AI SDK Mistral package adds or removes families.
- Keep TypeScript-only exports intentionally deferred unless a broader Rust package-facade pattern
  emerges first.
