# PMX-070 Audio And Media Provider Decision

Status: Done
Date: 2026-05-26

## Scope

This decision compares the local AI SDK reference packages for Deepgram, ElevenLabs, Fal, and
Replicate against Siumai's current stable media families, registry handles, provider catalog shape,
and provider-extension policy.

Upstream authority:

- `repo-ref/ai/packages/deepgram/src/index.ts`
- `repo-ref/ai/packages/deepgram/src/deepgram-provider.ts`
- `repo-ref/ai/packages/deepgram/src/*speech*`
- `repo-ref/ai/packages/deepgram/src/*transcription*`
- `repo-ref/ai/packages/elevenlabs/src/index.ts`
- `repo-ref/ai/packages/elevenlabs/src/elevenlabs-provider.ts`
- `repo-ref/ai/packages/elevenlabs/src/*speech*`
- `repo-ref/ai/packages/elevenlabs/src/*transcription*`
- `repo-ref/ai/packages/fal/src/index.ts`
- `repo-ref/ai/packages/fal/src/fal-provider.ts`
- `repo-ref/ai/packages/fal/src/*image*`
- `repo-ref/ai/packages/fal/src/*speech*`
- `repo-ref/ai/packages/fal/src/*transcription*`
- `repo-ref/ai/packages/fal/src/*video*`
- `repo-ref/ai/packages/replicate/src/index.ts`
- `repo-ref/ai/packages/replicate/src/replicate-provider.ts`
- `repo-ref/ai/packages/replicate/src/*image*`
- `repo-ref/ai/packages/replicate/src/*video*`

Siumai authority:

- `siumai/src/{image,speech,transcription,video}.rs`
- `siumai-registry/src/registry/entry/handles/{image,audio,video}.rs`
- `siumai-registry/src/registry/entry/factory.rs`
- `siumai-registry/src/native_provider_metadata.rs`
- `siumai-registry/src/provider_catalog.rs`
- `docs/adr/0003-provider-ext-export-policy.md`
- `docs/adr/0004-experimental-surface-policy.md`

## Decision

Do not start broad audio/media provider implementation inside this market-expansion workstream.
PMX-070 should close as a decision task and split dedicated follow-ons only after the lane closes.

Near-term priority order:

| Rank | Package | npm downloads, 2026-04-25..2026-05-24 | AI SDK families | Decision |
| ---: | --- | ---: | --- | --- |
| 1 | `@ai-sdk/deepgram` | 605,045 | Speech, transcription | First dedicated audio-provider candidate. |
| 2 | `@ai-sdk/elevenlabs` | 533,132 | Speech, transcription | Second audio-provider candidate, especially if TTS quality/voice workflows become product priority. |
| 3 | `@ai-sdk/replicate` | 257,948 | Image, video | Defer to a media-provider lane with prediction polling and dynamic model-input gates. |
| 4 | `@ai-sdk/fal` | 192,339 | Image, speech, transcription, video | Defer to a broad media-provider lane after common queue/polling policy is explicit. |

Recommended follow-on split:

- `deepgram-audio-provider`: narrow first audio lane for speech and transcription.
- `elevenlabs-audio-provider`: second audio lane, or a second task in a broader audio workstream if Deepgram
  lands cleanly.
- `media-task-polling-foundation`: shared policy for queued media tasks, polling intervals, timeout defaults,
  cancellation, materialization, and provider references before adopting Fal or Replicate video paths.
- `replicate-image-video-provider`: image/video provider lane after polling foundation is in place.
- `fal-media-provider`: broad media provider lane after polling foundation and model-specific request-shape
  strategy are in place.

## Findings

| Provider | Upstream package surface | Siumai fit | Cost and risk | Result |
| --- | --- | --- | --- | --- |
| Deepgram | `createDeepgram`, `deepgram`, `DEEPGRAM_API_KEY`, `authorization: Token ...`, base API `https://api.deepgram.com`, `transcription()` at `/v1/listen`, `speech()` at `/v1/speak`, no language/embedding/image model support. | Siumai already has stable speech and transcription families plus registry audio handles. No first-class Deepgram provider root exists. | Low to medium. The surface is narrow and mostly synchronous, but provider-specific audio options, output-format mapping, and model catalog policy still need no-network tests. | Pick first if Siumai wants an AI SDK-aligned audio provider. |
| ElevenLabs | `createElevenLabs`, `elevenLabs`, deprecated `elevenlabs`, `ELEVENLABS_API_KEY`, `xi-api-key`, base API `https://api.elevenlabs.io`, `transcription()` at `/v1/speech-to-text`, `speech()` at `/v1/text-to-speech/{voiceId}`. | Siumai speech/transcription families can carry the core behavior. No first-class ElevenLabs provider root exists. | Medium. More voice/TTS-specific knobs than Deepgram, including voice id separation, voice settings, pronunciation dictionaries, seeds, context text, and normalization controls. | Strong second audio candidate; avoid bundling with Deepgram unless the lane is explicitly audio-focused. |
| Replicate | `createReplicate`, `replicate`, `REPLICATE_API_TOKEN`, image and video families, model/version id split, predictions endpoints, `Prefer: wait`, optional polling, output download/materialization. | Siumai has image and video families, including video task/materialization concepts. No first-class Replicate provider root exists. | High. Dynamic model inputs, versioned model ids, Flux-2 multi-image special cases, prediction polling, timeout behavior, and output materialization need dedicated gates. | Defer. It is a good image/video candidate after queued media task policy is settled. |
| Fal | `createFal`, `fal`, `FAL_API_KEY` with `FAL_KEY` fallback, image/speech/transcription/video families, `https://fal.run` plus `https://queue.fal.run` queue paths, polling for transcription and video. | Siumai has all required stable families, but Fal spans too many media surfaces for a small polish task. No first-class Fal provider root exists. | High. Queue response handling, per-family polling defaults, provider-specific model options, arbitrary passthrough fields, and broad model catalog churn create maintenance risk. | Defer. Split only after polling foundation and request-shape strategy are explicit. |

## Rationale

Deepgram and ElevenLabs are the only packages in this group with both meaningful market signal and a narrow
implementation boundary. They map to Siumai's existing speech/transcription families without requiring new
image/video task semantics.

Replicate and Fal are not bad targets, but they should not be adopted as tail-end work in the current lane.
Both pull in provider-owned asynchronous media behavior and model-specific request shapes that deserve a
separate design and test plan. Fal is broadest, while Replicate has stronger download signal and a clearer
image/video scope. Neither should block this market-expansion lane from closing.

Siumai should also avoid adding `prelude::unified` exports for these providers during initial adoption. Follow
the provider-extension policy: expose a stable `siumai::provider_ext::<provider>` root, registry/catalog wiring,
and focused no-network tests first; widen convenience exports only after the provider proves stable.

## Non-Goals

- No live credential tests are required for PMX-070.
- No Deepgram, ElevenLabs, Fal, or Replicate implementation lands in PMX-070.
- No broad AI SDK media package parity commitment is created by this decision.
- No DeepInfra full catalog parity decision is reopened here.

## Validation

Fresh PMX-070 validation is recorded in `EVIDENCE_AND_GATES.md`.
