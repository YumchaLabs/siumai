# PMX-050 Mistral Package Surface Audit

Status: Done
Date: 2026-05-25

## Scope

This audit compares the local AI SDK reference package `@ai-sdk/mistral` against Siumai's current
OpenAI-compatible Mistral preset, registry integration, and facade exports.

Upstream authority:

- `repo-ref/ai/packages/mistral/src/mistral-provider.ts`
- `repo-ref/ai/packages/mistral/src/mistral-chat-language-model.ts`
- `repo-ref/ai/packages/mistral/src/mistral-chat-language-model-options.ts`
- `repo-ref/ai/packages/mistral/src/mistral-embedding-model.ts`
- `repo-ref/ai/packages/mistral/src/mistral-embedding-options.ts`
- `repo-ref/ai/packages/mistral/src/convert-to-mistral-chat-messages.ts`
- `repo-ref/ai/packages/mistral/src/mistral-prepare-tools.ts`
- `repo-ref/ai/packages/mistral/src/map-mistral-finish-reason.ts`

Siumai authority:

- `siumai-provider-openai-compatible/src/providers/openai_compatible/providers/models/mistral.rs`
- `siumai-provider-openai-compatible/src/provider_options/mistral.rs`
- `siumai-provider-openai-compatible/src/providers/openai_compatible/config/builtin_providers.rs`
- `siumai-provider-openai-compatible/src/providers/openai_compatible/ext/request_options.rs`
- `siumai-protocol-openai/src/standards/openai/compat/spec.rs`
- `siumai-registry/src/provider_catalog.rs`
- `siumai-registry/src/registry/factories/contract_tests.rs`
- `siumai/src/provider_ext/mistral.rs`
- `siumai/tests/provider_public_path_parity/openai_compatible_audio_public_path.rs`
- `siumai/tests/public_surface_imports_test.rs`

## Decision

Keep Mistral on the shared OpenAI-compatible runtime. The AI SDK package has a dedicated wrapper, but its
current language and embedding endpoints still fit Siumai's OpenAI-compatible preset with provider-owned
request normalization, model constants, typed options, registry catalog entries, and facade helpers.

Do not open a native Mistral provider lane from this audit. Reconsider only if upstream Mistral behavior moves
away from OpenAI-compatible chat/embedding transport or needs provider-specific stream/message machinery that
cannot be represented by the shared runtime.

## Findings

| Area | Upstream AI SDK behavior | Siumai result | Status |
| --- | --- | --- | --- |
| Provider factory | `createMistral()` plus default `mistral`, default base URL `https://api.mistral.ai/v1`, `MISTRAL_API_KEY`, custom headers/fetch/id. | `MistralProviderSettings`, `mistral()`, `create_mistral()`, `Provider::mistral()`, `SiumaiBuilder::mistral()`, config overrides, registry overrides. | Green |
| Families | Chat/language model and embedding model. Image model explicitly throws `NoSuchModelError`; deprecated text-embedding aliases map to embedding. | Chat/language and embedding are supported; completion/image and other families are rejected before transport. Deprecated JS callable aliases are not mirrored as Rust names. | Green with intentional Rust API divergence |
| Model ids | 18 chat ids plus open string; `mistral-embed` for embeddings. | Provider-owned constants include the 18 audited chat ids and `mistral-embed`; registry catalog lists chat and embedding ids. | Green |
| Chat provider options | `safePrompt`, `documentImageLimit`, `documentPageLimit`, `structuredOutputs`, `strictJsonSchema`, `parallelToolCalls`, `reasoningEffort`. | `MistralChatOptions` / `MistralLanguageModelOptions` serialize camelCase and accept snake_case aliases; request ext merges under provider key `mistral`. | Green |
| Unsupported common settings | AI SDK warns and does not send `topK`, `frequencyPenalty`, or `presencePenalty`. | `frequency_penalty` and `presence_penalty` were already stripped. PMX-050 added `top_k` stripping for Mistral when not explicitly passed through as a provider option. | Fixed |
| Stop and seed mapping | `stopSequences` maps to `stop`; `seed` maps to `random_seed`. | PMX-050 keeps `stop` in the Mistral request body and maps `seed` to `random_seed`. | Fixed |
| Reasoning effort | Supported for `mistral-small-latest`, `mistral-small-2603`, `mistral-medium-3`, and `mistral-medium-3.5`; unsupported models drop the option and warn. | PMX-050 expanded the keep-list to include the two medium models. Unsupported models still drop `reasoning_effort`. | Fixed |
| Structured outputs | `structuredOutputs` defaults true; `strictJsonSchema` defaults false; JSON object mode injects a JSON instruction. | Existing Mistral protocol tests cover JSON schema default strict false, `structuredOutputs: false` object mode, strict option cleanup, and JSON instruction injection. | Green |
| Tools | Empty tool arrays are omitted; provider-defined tools warn; `required` maps to `any`; named tool choice filters tools and forces `any`; `parallelToolCalls` only applies when tools exist. | Existing protocol tests cover `required` -> `any`, named tool filtering, provider-option normalization, and `parallel_tool_calls`. Provider-defined tool warning parity remains a lower-level runtime warning divergence, not a public surface blocker. | Green with minor warning-surface divergence |
| Finish reason | `model_length` maps to length. | Runtime Mistral test covers `model_length` finish normalization. | Green |
| Embeddings | `/embeddings`, body includes `model`, `input`, `encoding_format: "float"`; max 32 values; no parallel calls. | Existing config/registry/facade parity tests cover `/embeddings` URL and request body through builder, config, and registry paths. Max-per-call and parallel-call metadata are not exposed as identical Rust fields. | Green with intentional Rust API divergence |

## Fixed Gaps

- `siumai-protocol-openai/src/standards/openai/compat/spec.rs` now strips Mistral `top_k` from common
  parameters unless the caller intentionally passes `top_k` through the provider-options escape hatch.
- `siumai-protocol-openai/src/standards/openai/compat/spec.rs` now keeps Mistral `stop` when
  `stopSequences` are set, matching AI SDK's `stopSequences` -> `stop` mapping.
- `siumai-protocol-openai/src/standards/openai/compat/spec.rs` now preserves Mistral `reasoning_effort` for
  `mistral-medium-3` and `mistral-medium-3.5`, matching the current AI SDK support set.

## Intentional Divergences

- Rust does not mirror JavaScript callable provider objects or deprecated `textEmbedding` /
  `textEmbeddingModel` names one-for-one. The stable Rust shape is builder/config/registry/facade based.
- Runtime warning payloads do not attempt to be byte-for-byte equivalent to AI SDK warnings. The no-network gate
  proves emitted request bodies avoid unsupported parameters.
- A native Mistral provider is deferred because the audited behavior remains representable by the
  OpenAI-compatible runtime.
- Live credential tests are not required for PMX-050; all gates stay no-network.

## Validation

See `EVIDENCE_AND_GATES.md` for fresh command output recorded after the PMX-050 fix.
