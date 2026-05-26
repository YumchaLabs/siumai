# ElevenLabs Audio Provider

Status: Closed
Last updated: 2026-05-26

## Why This Lane Exists

The AI SDK provider market expansion lane ranked ElevenLabs as the second dedicated audio-provider
candidate after Deepgram. Deepgram has now closed, so this lane turns the queued ElevenLabs follow-on
into a bounded provider implementation.

ElevenLabs is still narrow enough for Siumai's existing speech and transcription families, but it has
more voice-centric TTS behavior than Deepgram. This lane exists to capture those differences before
implementation starts.

## Relevant Authority

- Decision source:
  - `docs/workstreams/ai-sdk-provider-market-expansion/AUDIO_MEDIA_DECISION.md`
- Predecessor lane:
  - `docs/workstreams/deepgram-audio-provider`
- Upstream AI SDK package:
  - `repo-ref/ai/packages/elevenlabs/src/index.ts`
  - `repo-ref/ai/packages/elevenlabs/src/elevenlabs-provider.ts`
  - `repo-ref/ai/packages/elevenlabs/src/elevenlabs-speech-model.ts`
  - `repo-ref/ai/packages/elevenlabs/src/elevenlabs-speech-options.ts`
  - `repo-ref/ai/packages/elevenlabs/src/elevenlabs-speech-model-options.ts`
  - `repo-ref/ai/packages/elevenlabs/src/elevenlabs-speech-api-types.ts`
  - `repo-ref/ai/packages/elevenlabs/src/elevenlabs-transcription-model.ts`
  - `repo-ref/ai/packages/elevenlabs/src/elevenlabs-transcription-options.ts`
  - `repo-ref/ai/packages/elevenlabs/src/elevenlabs-transcription-model-options.ts`
  - `repo-ref/ai/packages/elevenlabs/src/elevenlabs-api-types.ts`
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

The frozen market evidence window from the provider expansion lane is `2026-04-25` through
`2026-05-24`. `@ai-sdk/elevenlabs` recorded 533,132 downloads in that window.

AI SDK package surface:

| Area | Upstream behavior |
| --- | --- |
| Factory | `createElevenLabs()` plus default `elevenLabs`; deprecated lowercase alias `elevenlabs`. |
| Auth | `ELEVENLABS_API_KEY` fallback, `xi-api-key: <key>`. |
| Base API | `https://api.elevenlabs.io`. |
| Families | `speech()` and `transcription()`; language, embedding, and image models throw unsupported-model errors. |
| Default callable | Calling the provider with `scribe_v1` returns a transcription model. |
| Speech endpoint | `POST /v1/text-to-speech/{voiceId}` with JSON body, query `output_format` and `enable_logging`, binary audio response. |
| Transcription endpoint | `POST /v1/speech-to-text` with multipart form data and JSON response. |
| Speech model ids | `eleven_v3`, `eleven_multilingual_v2`, `eleven_flash_v2_5`, `eleven_flash_v2`, `eleven_turbo_v2_5`, `eleven_turbo_v2`, `eleven_monolingual_v1`, `eleven_multilingual_v1`, plus custom strings. |
| Transcription model ids | `scribe_v1`, `scribe_v1_experimental`, plus custom strings. |
| Default voice id | AI SDK uses `21m00Tcm4TlvDq8ikWAM` when no voice is supplied. |

## Problem

Siumai now has a first-class Deepgram audio provider, but ElevenLabs remains missing from the provider
crate, registry, catalog, and facade surfaces. Users who want AI SDK-aligned ElevenLabs speech or
transcription currently have no stable Siumai package-equivalent path.

## Target State

When this workstream closes:

- `siumai-provider-elevenlabs` exists as a provider-owned crate for ElevenLabs speech and transcription.
- ElevenLabs settings support explicit API key, `ELEVENLABS_API_KEY`, base URL override, HTTP config,
  headers, custom transport, retry options, and interceptors using local provider patterns.
- Speech generation supports `/v1/text-to-speech/{voiceId}`, AI SDK output-format mapping,
  voice-specific options, binary audio response handling, warnings for unsupported shared speech controls,
  and provider-scoped metadata.
- Transcription supports `/v1/speech-to-text` multipart upload, typed transcription options, word/segment
  extraction, language, duration, and provider-scoped metadata.
- Registry and facade paths expose ElevenLabs under stable provider roots without widening
  `prelude::unified`.
- No-network tests prove auth/header/URL/body behavior, public imports, registry construction, capability
  metadata, and unsupported family rejection before transport use.

## In Scope

- New `siumai-provider-elevenlabs` workspace crate.
- Provider-owned settings, model constants, speech options, transcription options, error mapping, and client.
- Speech and transcription family trait implementations.
- Registry factory, native metadata, provider catalog, feature wiring, and builder/facade exports.
- Focused no-network tests that cover both AI SDK-aligned request shapes.

## Out Of Scope

- Live ElevenLabs credential tests as required gates.
- Voice listing, voice cloning, pronunciation-dictionary management resources, or account APIs.
- Full mirroring of JavaScript callable provider object syntax.
- Fal, Replicate, queued media polling, or broad asynchronous media task foundations.
- Streaming audio workflows unless already required by the stable Siumai speech/transcription traits.
- `prelude::unified` widening.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| ElevenLabs should be a native audio provider crate, not an OpenAI-compatible preset. | High | AI SDK uses provider-owned `xi-api-key`, TTS, and STT endpoints. | Revisit crate boundary only if a shared audio protocol emerges. |
| Siumai's stable speech/transcription families can carry the core behavior. | High | Deepgram closed through the same families; AI SDK ElevenLabs exposes only speech and transcription. | Add provider-owned extensions before changing core traits. |
| Voice id can be represented as a speech request voice field or provider option without adding a new core concept. | Medium | AI SDK maps `voice` into the TTS path and defaults it when omitted. | If the current request shape is insufficient, add a provider-owned extension trait first. |
| No live credentials are needed for first closeout. | High | Existing provider lanes use no-network URL/body/header tests. | Add optional examples, not mandatory release gates. |
| Multipart upload can reuse existing transport/test infrastructure or a narrow helper inside the provider crate. | Medium | Siumai already has raw-audio Deepgram tests and HTTP transport abstractions. | If multipart transport is missing, ELA-020 must add the smallest reusable helper under provider ownership or split a prerequisite task. |

## Architecture Direction

ElevenLabs should follow the closed Deepgram lane's architecture, with provider-specific differences made
explicit:

- keep protocol mapping and typed provider options inside `siumai-provider-elevenlabs`;
- expose public Rust convenience under `siumai::provider_ext::elevenlabs` and
  `siumai::providers::elevenlabs`;
- wire `siumai-registry` through a feature-gated `ElevenLabsProviderFactory`;
- use provider metadata that declares speech, transcription, and audio capabilities;
- reject chat, completion, embedding, image, video, and rerank before transport use;
- use `xi-api-key` auth instead of Deepgram's `authorization: Token ...`;
- use `/v1/text-to-speech/{voiceId}` for TTS and `/v1/speech-to-text` multipart form data for STT;
- keep provider-specific voice settings, pronunciation dictionaries, seed/context text, normalization,
  diarization, timestamp granularity, and file format options in typed provider option structs.

## Closeout Condition

This lane can close when ElevenLabs has speech and transcription support through provider crate, registry,
and facade paths, with no-network gates proving AI SDK-aligned request behavior and explicit documentation
for deferred resources or intentional Rust API divergences.

Closeout status: CLOSED on 2026-05-26 after fresh provider-crate, registry, facade, formatting,
JSON, and whitespace gates. Voice resources, live credential tests, and queued media foundations
remain separate follow-ons.
