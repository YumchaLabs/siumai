# Groq Package Surface Alignment - Design

Last updated: 2026-04-15

## Problem framing

`repo-ref/ai/packages/groq` is no longer just a thin "Groq exists" reference for Siumai. It is the
canonical upstream package boundary we use to audit:

- typed language-model options
- transcription option aliases
- provider-defined tool exports
- built-in model ids and curated model groups
- request-shaping details and response metadata behavior

The earlier `groq-browser-search-alignment` workstream closed one concrete provider-tool/runtime
gap, but it did not close the broader package-surface drift.

The next parity pass therefore needs a dedicated workstream that treats `@ai-sdk/groq` as a whole
package contract rather than mixing every follow-up into the browser-search scope.

## Scope

This workstream tracks parity for the upstream `repo-ref/ai/packages/groq/src/*` package surface:

- `groq-chat-options.ts`
- `groq-transcription-options.ts`
- `groq-provider.ts`
- `groq-tools.ts`
- `groq-browser-search-models.ts`
- `get-response-metadata.ts`
- related tests and changelog entries when they describe externally visible behavior

Current high-priority parity areas:

- typed chat option enums and builder/config coverage
- curated model id freshness
- keeping the AI SDK-aligned chat/transcription catalog separate from provider-owned Groq-only
  extensions such as TTS
- response metadata / warnings / request-shaping deltas that affect the public Rust contract
- provider settings parity (`baseURL`, `apiKey`, `headers`, `fetch`) where the upstream package
  defines a stable provider-construction contract

## Design principles

### 1. Match upstream package boundaries first

If `@ai-sdk/groq` exports a public type alias, option enum, tool constructor, or model id, Siumai
should expose a closely comparable provider-owned/public Rust surface unless there is a deliberate
Rust-specific reason not to.

### 2. Prefer typed public surfaces over raw `providerOptions` escape hatches

Raw `provider_options_map["groq"]` remains necessary as an escape hatch, but it should not be the
default way to access stable upstream options that already have a clear typed contract in AI SDK.

### 3. Keep provider-owned behavior inside the Groq package layer

When parity requires request shaping, warnings, or provider-native tools, the implementation should
stay in the Groq provider crate rather than leaking temporary state into shared request structs.

### 4. Do not invent upstream surfaces

This audit follows the upstream package boundary. If `repo-ref/ai/packages/groq` does not expose a
first-class surface, we do not add one merely for symmetry. That rule is especially important for
modalities such as image generation, where provider-specific APIs may exist elsewhere but are not a
shared `@ai-sdk/groq` contract.

### 5. Keep provider-owned extras clearly labeled

Siumai can keep provider-owned Groq features that exist outside `@ai-sdk/groq` when they are
backed by a real runtime capability, but those extras should not silently pollute the audited AI
SDK catalog. Groq TTS is the current example: it remains supported on the Rust provider surface,
but it is documented and grouped as a provider-owned extension rather than an upstream chat-model
contract. The same rule applies to discovery paths: provider-owned audio helpers should live under
explicit escape-hatch modules such as `ext::audio_options`, not in the main AI SDK-aligned
`options::*` lane.

## Current implemented parity in this workstream

The first completed slice aligns the Groq typed/model surface with audited upstream changes:

- `GroqServiceTier` now includes `performance`
- `GroqReasoningEffort` now includes `low`, `medium`, and `high`
- `GroqLanguageModelOptions` now serializes the audited AI SDK-style provider-option keys
  (`serviceTier`, `reasoningEffort`, `reasoningFormat`, `topLogprobs`, `parallelToolCalls`,
  `user`, `structuredOutputs`, `strictJsonSchema`) instead of exposing Groq wire-shape keys on the
  public Rust options surface
- Groq config/builder/request-ext helpers now cover the same typed option set and no longer force
  callers back to raw `providerOptions.groq` objects for the remaining audited chat-option fields
- Groq config defaults and request-level typed options now both normalize back to the Groq wire
  shape (`service_tier`, `reasoning_effort`, `reasoning_format`, `parallel_tool_calls`,
  `top_logprobs`) before transport, so config-first and request-first paths stay aligned
- the public/facade Groq `options::*` lane now stays AI SDK-shaped instead of also re-exporting
  Rust-only `GroqSttOptions` / `GroqTtsOptions`; those concrete audio escape hatches remain under
  `ext::audio_options`
- `structuredOutputs: false` now also works on the Groq provider-owned path, downgrading JSON
  schema requests to `response_format = { "type": "json_object" }` with the expected AI SDK-style
  warning
- Groq response metadata now also preserves the upstream stable Groq metadata trio
  (`id`, `modelId`, `timestamp`) on both the provider-owned spec path and the OpenAI-compatible
  config/runtime path, and the typed Groq response helper can expose the same values on
  non-stream and stream-end responses
- the built-in Groq preview model constant `KIMI_K2_INSTRUCT` now points to
  `moonshotai/kimi-k2-instruct-0905`
- the built-in Groq chat catalog now matches the currently audited `@ai-sdk/groq`
  `GroqChatModelId` union, including the missing `gemma2-9b-it`, `llama-guard-3-8b`,
  `llama3-{8b,70b}-8192`, `qwen-qwq-32b`, `qwen-2.5-32b`, and
  `deepseek-r1-distill-qwen-32b` ids
- older non-package catalog entries such as `compound-beta`, the Llama 3.2 vision previews, Groq
  tool-use previews, and `gemma-7b-it` are removed from the public Groq model catalog
- provider-owned Groq TTS models remain available, but they now live in a separate provider-owned
  speech catalog instead of being mixed into the AI SDK-aligned chat model groups
- config/builder/spec/client tests now lock those values through the provider-owned runtime path
- Groq provider construction now also aligns more closely with `groq-provider.ts` settings:
  the OpenAI-compatible `groq` preset uses `GROQ_API_KEY` as its canonical environment variable,
  `GroqConfig::from_env()` and `GroqConfig::with_api_key(...)` mirror the stable
  `apiKey`-configuration lane, and `GroqBuilder::headers(...)` keeps an AI SDK-style alias on top
  of the existing Rust builder/config HTTP configuration surface
- the stable Rust facade now also re-exports `GroqBuilder` under `provider_ext::groq::*`, keeping
  the provider-owned public construction lane visible alongside `GroqClient` / `GroqConfig`
  instead of forcing callers back through unrelated root-level discovery paths

## Follow-up audit questions

- Which Groq AI SDK chat options are already covered by shared/common request fields versus still
  missing from `GroqLanguageModelOptions`?
- Are there remaining Groq-specific response metadata fields that should be surfaced as stable
  provider metadata instead of raw passthrough blobs only?
- Should the public facade eventually expose a tighter Groq-specific `model_sets` module so
  provider-owned speech/transcription catalogs do not need to share naming conventions with the
  AI SDK-aligned chat package surface?
