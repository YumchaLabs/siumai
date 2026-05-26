# ElevenLabs PVC Voice Workflow

Status: Closed
Last updated: 2026-05-26

## Why This Lane Exists

Professional Voice Cloning (PVC) is not a single voice-create call. The ElevenLabs API models PVC
as a workflow that starts with metadata, then adds samples, optionally runs speaker separation,
updates sample selections, starts training, and completes captcha or manual verification. Siumai
already exposes read-only voice resources, settings mutation, deletion, and IVC creation, but it
does not expose this PVC workflow.

## Relevant Authority

- ADRs:
  - `docs/adr/0003-provider-ext-export-policy.md`
- Architecture:
  - `docs/architecture/public-surface.md`
- Related workstreams:
  - `docs/workstreams/elevenlabs-voice-resources`
  - `docs/workstreams/elevenlabs-voice-mutation-resources`
- External references:
  - `repo-ref/ai/packages/elevenlabs`
  - `https://elevenlabs.io/docs/llms.txt`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/create.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/update.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/train.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/create.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/update.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/delete.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/get-audio.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/get-waveform.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/get-speaker-separation-status.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/separate-speakers.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/get-separated-speaker-audio.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/verification/request.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/verification/captcha.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/verification/captcha/verify.mdx`

## Problem

Applications using Siumai cannot drive the documented ElevenLabs PVC workflow without dropping to
raw HTTP calls. That leaves path encoding, multipart file ownership, per-request HTTP overrides,
response typing, and no-network test coverage inconsistent with the rest of the ElevenLabs resource
surface.

## Target State

Siumai exposes provider-owned PVC resource methods on `ElevenLabsVoices` under the existing
`provider_ext::elevenlabs::resources::*` facade path. The API should cover the documented PVC
metadata, sample, speaker-separation, training, captcha, and manual-verification endpoints with
typed requests and responses, preserve unknown response fields where useful, and avoid widening
`prelude::unified`.

## In Scope

- PVC metadata JSON endpoints:
  - create PVC voice
  - update PVC voice metadata
  - start PVC training
- PVC sample workflow endpoints:
  - add samples
  - update sample metadata/trim/speaker selection
  - delete sample
  - get sample preview audio
  - get sample visual waveform
  - get speaker separation status
  - start speaker separation
  - get separated speaker audio
- PVC verification endpoints:
  - request manual verification
  - get captcha response
  - submit captcha recording
- Facade exports under `provider_ext::elevenlabs::resources::*` and
  `providers::elevenlabs::resources::*`.
- Focused no-network tests using the existing custom transport harness.
- Changelog and closeout evidence.

## Out Of Scope

- Live ElevenLabs credential tests.
- A generic cross-provider voice-management trait.
- Exporting PVC APIs from `prelude::unified`.
- Changing the existing speech or transcription family APIs.
- Ordinary voice `POST /v1/voices/{voice_id}/edit`; that is the next smaller follow-on after this
  lane closes.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| AI SDK has no voice resource client for these endpoints. | High | `repo-ref/ai/packages/elevenlabs` audit from the previous lane. | If upstream adds one, re-audit names and shape before exposing new public paths. |
| PVC belongs under the existing provider-owned `ElevenLabsVoices` resource client. | High | ADR-0003 and current voice resource API shape. | A separate client would fragment the public surface and duplicate transport helpers. |
| Official PVC docs describe JSON and multipart response bodies accurately enough for no-network tests. | Medium | Current `.mdx` OpenAPI pages fetched on 2026-05-26. | If response schemas drift, keep flatten `extra` fields or split uncertain endpoints. |
| Captcha GET currently documents an empty JSON response body. | Medium | Official captcha `.mdx` page shows an empty object schema. | If live behavior returns audio/text, the typed response should preserve unknown fields until audited. |

## Architecture Direction

Keep PVC as a provider-owned extension on `ElevenLabsVoices`, using the existing
`resource_http::{execute_get_json, execute_post_json, execute_delete_json, execute_multipart_json}`
helpers. JSON request builders should reject missing required fields and allow documented empty
optional bodies only where the API examples and schema permit `{}`. Multipart requests should reuse
`ElevenLabsVoiceSampleFile` for file ownership, filenames, and MIME types instead of introducing a
parallel upload abstraction.

Response types should model stable documented fields and use `#[serde(flatten)]` for forward
compatibility. Status-only responses should reuse `ElevenLabsVoiceStatusResponse` when the wire
shape is the same.

## Closeout Condition

This lane can close when:

- PVC workflow resource methods and types are implemented,
- facade exports and compile-surface tests cover the public paths,
- focused provider tests validate request paths, JSON bodies, multipart forms, headers, and response
  mapping,
- `CHANGELOG.md` records the shipped PVC surface,
- fresh verification gates are recorded in `EVIDENCE_AND_GATES.md`,
- and ordinary `edit_voice` remains split for the next follow-on.
