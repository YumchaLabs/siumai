# ElevenLabs Voice Resources

Status: Active
Last updated: 2026-05-26

## Why This Lane Exists

The `elevenlabs-audio-provider` lane closed first-class speech and transcription support. That lane
intentionally deferred account-scoped voice resources because they are not part of the stable
speech/transcription model families and would have expanded the provider crate beyond the first
AI SDK-aligned package surface.

ElevenLabs is voice-centric: users need to discover available voice IDs before calling TTS, and the
provider already accepts voice IDs and pronunciation dictionary locators on speech requests. This
follow-on adds provider-owned resource clients without turning voice management into a core Siumai
family.

## Relevant Authority

- Closed predecessor lane:
  - `docs/workstreams/elevenlabs-audio-provider`
- Upstream AI SDK package:
  - `repo-ref/ai/packages/elevenlabs/src/index.ts`
  - `repo-ref/ai/packages/elevenlabs/src/elevenlabs-provider.ts`
  - `repo-ref/ai/packages/elevenlabs/src/elevenlabs-speech-model.ts`
- Official ElevenLabs API docs:
  - `https://elevenlabs.io/docs/api-reference/voices/search.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/get.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/ivc/create.mdx`
  - `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/list.mdx`
  - `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/create-from-rules.mdx`
  - `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/create-from-file.mdx`
- Local public-surface policy:
  - `docs/architecture/public-surface.md`
  - `docs/adr/0003-provider-ext-export-policy.md`

## Resource Snapshot

AI SDK `@ai-sdk/elevenlabs` does not expose a voice resource client. It exposes provider creation,
speech, transcription, model IDs, speech voice ID type, typed speech options, typed transcription
options, and `VERSION`.

Official ElevenLabs resources that matter to Siumai TTS users:

| Area | Official API shape | Siumai lane decision |
| --- | --- | --- |
| Voice catalog | `GET /v2/voices` with search, filters, pagination, and `voice_ids` lookup. | In scope first because it lets users discover voice IDs for TTS. |
| Voice detail | `GET /v1/voices/{voice_id}`. | In scope first as a read-only companion to catalog listing. |
| Voice settings | `GET /v1/voices/settings/default`, `GET/POST /v1/voices/{voice_id}/settings/edit`. | Defer until read-only catalog proves the resource shape. |
| IVC clone | `POST /v1/voices/add` multipart. | Defer or split because it is mutation-heavy and may require verification semantics. |
| Voice delete/update/samples/PVC | Multiple `voices/*` mutation and sample endpoints. | Out of first slice; candidate follow-ons. |
| Pronunciation dictionaries | List/get/create/update/download/rule mutation endpoints. | Separate task after voice catalog; used by existing TTS locator options. |

## Problem

Siumai users can call ElevenLabs TTS once they know a voice ID, but the facade currently provides no
typed, no-network-tested way to list or retrieve ElevenLabs voices. Copying raw endpoints into user
code couples applications to provider JSON shapes and loses Siumai's configured API key, base URL,
custom headers, transport, interceptor, and retry wiring.

## Target State

When this workstream closes:

- `siumai-provider-elevenlabs` owns resource clients under the provider module, starting with a
  read-only `ElevenLabsVoices` client.
- `siumai::provider_ext::elevenlabs::resources::*` and `siumai::providers::elevenlabs::resources::*`
  expose resource clients and typed request/response structs.
- Voice catalog listing and voice detail retrieval reuse `ElevenLabsConfig` auth, base URL,
  headers, custom transport, interceptors, and retry options.
- No-network tests prove `xi-api-key`, base URL override, custom headers, query mapping, pagination
  fields, and response mapping.
- Pronunciation dictionary resource scope is either implemented as a separate bounded task or
  explicitly split.
- `prelude::unified` and the stable speech/transcription traits remain unchanged.

## In Scope

- Provider-owned resource clients for ElevenLabs voice catalog and voice metadata.
- Typed query and response structs for `GET /v2/voices` and `GET /v1/voices/{voice_id}`.
- Facade re-exports under `provider_ext::elevenlabs::resources`.
- No-network tests using the existing HTTP transport/test-support patterns.
- Documentation of intentional divergence from AI SDK package surface.

## Out Of Scope

- Adding a new generic core `VoiceManagementCapability` before multiple providers need it.
- Live ElevenLabs credential tests as required gates.
- Voice cloning, PVC verification, voice sample management, and voice settings mutation in the
  first implementation task.
- Conversational agent, workspace sharing, history, dubbing, audio-native, or account APIs.
- Widening `prelude::unified`.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Voice resources should stay provider-owned, not core-owned. | High | AI SDK package does not expose resource clients; Siumai provider_ext policy already has `resources::*`. | Add an ADR only if a second provider needs the same cross-provider voice management contract. |
| Read-only voice catalog is the correct first slice. | High | TTS already needs voice IDs; official list/get endpoints are simpler than clone/PVC mutation endpoints. | Split to docs-only if the current HTTP test harness cannot capture GET query behavior. |
| Existing ElevenLabs config and transport wiring can be reused. | High | The closed audio lane already supports auth, base URL, headers, transport, interceptors, and retry wiring. | Add a narrow provider-local resource wiring helper before changing shared transport. |
| Pronunciation dictionary resources belong in this lane but not in the first task. | Medium | Speech options already accept dictionary locators; official API has multiple dictionary endpoints. | Split dictionary resources into a separate workstream if the endpoint surface grows too large. |

## Architecture Direction

Follow existing provider resource patterns:

- place resource clients inside `siumai-provider-elevenlabs/src/providers/elevenlabs`;
- keep resource structs provider-owned and serde-tolerant with `#[serde(flatten)]` metadata where
  official response models are broad or likely to evolve;
- expose only stable resource roots from the facade, not from `prelude::unified`;
- prefer typed query builders for documented query parameters but preserve unknown response fields
  so endpoint drift does not break users unnecessarily;
- reuse provider-local HTTP wiring rather than introducing a generic voice-resource trait in this
  lane.

## Closeout Condition

This lane can close when ElevenLabs voice catalog resources are available through provider and
facade paths with no-network request/response tests, and remaining mutation-heavy voice or
pronunciation dictionary APIs are either implemented in bounded tasks or split into documented
follow-ons.
