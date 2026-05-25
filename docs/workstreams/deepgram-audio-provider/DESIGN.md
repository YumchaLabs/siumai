# Deepgram Audio Provider

Status: Draft
Last updated: 2026-05-26

## Why This Lane Exists

The AI SDK provider market expansion lane ranked Deepgram as the first dedicated audio-provider candidate:
`@ai-sdk/deepgram` has the strongest audio/media package download signal in the sampled npm window and maps to
Siumai's existing speech and transcription model families without introducing queued video/media task semantics.

This lane turns that decision into a bounded provider implementation.

## Relevant Authority

- Decision source:
  - `docs/workstreams/ai-sdk-provider-market-expansion/AUDIO_MEDIA_DECISION.md`
- Upstream AI SDK package:
  - `repo-ref/ai/packages/deepgram/src/index.ts`
  - `repo-ref/ai/packages/deepgram/src/deepgram-provider.ts`
  - `repo-ref/ai/packages/deepgram/src/deepgram-speech-model.ts`
  - `repo-ref/ai/packages/deepgram/src/deepgram-speech-options.ts`
  - `repo-ref/ai/packages/deepgram/src/deepgram-speech-model-options.ts`
  - `repo-ref/ai/packages/deepgram/src/deepgram-transcription-model.ts`
  - `repo-ref/ai/packages/deepgram/src/deepgram-transcription-options.ts`
  - `repo-ref/ai/packages/deepgram/src/deepgram-transcription-model-options.ts`
- Siumai owners:
  - `siumai-core::speech::SpeechModel`
  - `siumai-core::transcription::TranscriptionModel`
  - `siumai/src/speech.rs`
  - `siumai/src/transcription.rs`
  - `siumai-registry/src/registry/entry/handles/audio.rs`
  - `siumai-registry/src/registry/entry/factory.rs`
  - `siumai/src/provider_ext/*`
- ADRs:
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0002-provider-crates-by-provider.md`
  - `docs/adr/0003-provider-ext-export-policy.md`
  - `docs/adr/0004-experimental-surface-policy.md`

## Market And Package Snapshot

The frozen market evidence window from the provider expansion lane is `2026-04-25` through `2026-05-24`.
`@ai-sdk/deepgram` recorded 605,045 downloads in that window.

AI SDK package surface:

| Area | Upstream behavior |
| --- | --- |
| Factory | `createDeepgram()` plus default `deepgram`. |
| Auth | `DEEPGRAM_API_KEY` fallback, `authorization: Token <key>`. |
| Base API | `https://api.deepgram.com`. |
| Families | `speech()` and `transcription()`; language, embedding, and image models throw unsupported-model errors. |
| Speech endpoint | `POST /v1/speak?model=<speech-model>&...` with JSON `{ text }`, binary audio response. |
| Transcription endpoint | `POST /v1/listen?model=<stt-model>&...` with raw audio body and `Content-Type` from input media type. |
| Model ids | Speech ids include `aura-*`; transcription ids include `base`, `enhanced`, `nova`, `nova-2`, `nova-3`, and variants. |

## Problem

Siumai already has stable Rust-first speech and transcription families, but it does not expose a first-class
Deepgram provider root, registry factory, provider catalog entry, or Deepgram-specific options. Users who want
AI SDK-aligned Deepgram audio behavior currently have no package-equivalent Siumai path.

## Target State

When this workstream closes:

- `siumai-provider-deepgram` exists as a provider-owned crate for Deepgram speech and transcription.
- Deepgram settings support explicit API key, `DEEPGRAM_API_KEY`, base URL override, HTTP config, headers,
  custom transport, retry options, and interceptors using local provider patterns.
- Speech generation supports the AI SDK-aligned `/v1/speak` request shape, output-format mapping, binary audio
  response handling, warnings for unsupported shared speech controls, and provider-scoped metadata.
- Transcription supports the AI SDK-aligned `/v1/listen` request shape, raw audio upload, media type header,
  transcript/segments/language/duration extraction, and provider-scoped metadata.
- Registry and facade paths expose Deepgram under stable provider roots without widening `prelude::unified`.
- No-network tests prove auth/header/URL/body behavior, public imports, registry construction, capability
  metadata, and unsupported family rejection before transport use.

## In Scope

- New `siumai-provider-deepgram` workspace crate.
- Provider-owned settings, model constants, speech options, transcription options, error mapping, and client.
- Speech and transcription family trait implementations.
- Registry factory, native metadata, provider catalog, feature wiring, and builder/facade exports.
- Focused no-network tests and examples or docs that demonstrate stable usage.

## Out Of Scope

- ElevenLabs provider implementation.
- Fal, Replicate, or common media polling foundations.
- Live Deepgram credential tests as required gates.
- Streaming audio or callback workflows unless already needed for parity-critical request shape.
- Full mirroring of JavaScript callable provider object syntax.
- `prelude::unified` widening.

## Architecture Direction

Deepgram should be a native audio provider crate, not an OpenAI-compatible preset. Its wire protocol is
provider-owned and limited to audio endpoints.

Implementation should follow the provider split already used by Siumai:

- keep protocol and typed provider options inside `siumai-provider-deepgram`;
- expose public Rust convenience under `siumai::provider_ext::deepgram` and `siumai::providers::deepgram`;
- wire `siumai-registry` through a feature-gated `DeepgramProviderFactory`;
- use `ProviderCapabilities::with_speech().with_transcription().with_audio()` or equivalent metadata;
- reject chat, completion, embedding, image, video, and rerank before transport use.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Deepgram can be implemented as a narrow native provider crate. | High | AI SDK provider exposes only speech and transcription. | If Deepgram adds broader families upstream, split expansion after this lane. |
| Siumai's stable speech/transcription family traits can carry the core behavior. | High | Existing `speech::synthesize` and `transcription::transcribe` helpers already map AI SDK-style results. | If option shape is insufficient, add provider-owned extensions rather than changing core first. |
| No live credentials are needed for first closeout. | High | Existing provider lanes rely on no-network URL/body/header tests. | Add optional examples, not mandatory release gates. |
| Output-format validation can start with AI SDK parity and Deepgram documented combinations. | Medium | AI SDK maps common `outputFormat` values to encoding/container/sample-rate query params. | If exact Deepgram docs drift, preserve escape-hatch provider options and document divergence. |

## Closeout Condition

This lane can close when Deepgram has speech and transcription support through provider crate, registry, and
facade paths, with no-network gates proving AI SDK-aligned request behavior and explicit documentation for
deferred options or intentional Rust API divergences.
