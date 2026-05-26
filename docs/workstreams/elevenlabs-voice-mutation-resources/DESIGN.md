# ElevenLabs Voice Mutation Resources

Status: Closed
Last updated: 2026-05-26

## Why This Lane Exists

The closed `elevenlabs-voice-resources` lane added read-only `ElevenLabsVoices`
catalog resources under `provider_ext::elevenlabs::resources::*`. Applications can
list and retrieve voices, but they still cannot manage user-owned voices, edit
voice settings, fetch or delete samples, or start the documented IVC/PVC voice
creation flows.

This lane covers provider-owned ElevenLabs voice mutation resources while keeping
Siumai's unified speech/transcription families unchanged.

## Relevant Authority

- Closed predecessor lane:
  - `docs/workstreams/elevenlabs-voice-resources`
- Official ElevenLabs docs:
  - `https://elevenlabs.io/docs/llms.txt`
  - `https://elevenlabs.io/docs/api-reference/voices/settings/get-default.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/settings/get.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/settings/update.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/delete.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/samples/get.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/samples/delete.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/ivc/create.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/update.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/create.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/update.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/train.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/create.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/update.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/delete.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/samples/get-audio.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/verification/request.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/verification/captcha.mdx`
  - `https://elevenlabs.io/docs/api-reference/voices/pvc/verification/captcha/verify.mdx`
- AI SDK reference:
  - `repo-ref/ai/packages/elevenlabs`
- Local public-surface policy:
  - `docs/architecture/public-surface.md`
  - `docs/adr/0003-provider-ext-export-policy.md`

## API Snapshot

| Area | Official API shape | Lane decision |
| --- | --- | --- |
| Voice settings | `GET /v1/voices/settings/default`, `GET /v1/voices/{voice_id}/settings`, `POST /v1/voices/{voice_id}/settings/edit` JSON settings. | First implementation slice. It reuses existing settings response semantics and proves JSON mutation wiring on `ElevenLabsVoices`. |
| Delete voice | `DELETE /v1/voices/{voice_id}` returns JSON status. | Follow-on after settings. Requires a shared DELETE JSON helper or a local resource helper extension. |
| Voice samples | `GET /v1/voices/{voice_id}/samples/{sample_id}/audio` and `DELETE /v1/voices/{voice_id}/samples/{sample_id}`. | Split decision task. The delete endpoint is simple; the audio endpoint needs a re-audit because current docs describe an audio URL but an empty JSON response schema. |
| IVC create | `POST /v1/voices/add` multipart `name`, `files`, optional `remove_background_noise`, `description`, `labels`; returns `voice_id` and `requires_verification`. | Bounded multipart follow-on after JSON mutation and delete behavior are proven. |
| Edit voice | `POST /v1/voices/{voice_id}/edit` multipart metadata and optional files; returns JSON status. | Pair with IVC multipart wiring or split if response/request shape grows. |
| PVC create/update/train | JSON endpoints under `/v1/voices/pvc`. | Separate PVC slice. Keep training and verification workflow state explicit and no-network first. |
| PVC samples and verification | Multipart, JSON, audio preview, captcha, and manual verification endpoints. | Audit as a workflow boundary before implementation; no live credential gate in this lane unless explicitly accepted. |

## Target State

Closed target state:

- `ElevenLabsVoices` owns provider-specific mutation/read helpers that are not part of the unified
  speech/transcription families.
- Public resource exports remain under `siumai::provider_ext::elevenlabs::resources::*` and
  `siumai::providers::elevenlabs::resources::*`.
- The first JSON settings slice is implemented with typed request structs and no-network tests.
- Delete/sample/IVC/PVC endpoints are either implemented in bounded slices or explicitly split with
  evidence.
- `prelude::unified` remains unchanged.

## In Scope

- Provider-owned ElevenLabs voice settings, voice deletion, sample, IVC, and PVC resource methods.
- Typed request/response structs for accepted JSON, multipart, binary, or empty-response endpoints.
- Shared resource HTTP wiring for auth headers, base URL override, request header merge, custom
  transport, interceptors, and retry behavior.
- No-network tests for path encoding, body shape, response mapping, and public facade exports.
- CHANGELOG and workstream closeout.

## Out Of Scope

- Live ElevenLabs credential tests as required gates.
- A generic cross-provider voice-management trait.
- Changes to TTS/STT unified request options or `prelude::unified`.
- Voice library marketplace endpoints unless they are explicitly brought into a later follow-on.

## Architecture Direction

Reuse the resource pattern from the closed read-only voice resources and pronunciation dictionary
mutation lanes:

- keep methods on `ElevenLabsVoices`;
- extend `resource_http` only when the lane needs a reusable verb helper;
- keep request structs endpoint-specific and typed;
- preserve provider drift with `#[serde(flatten)]` on response structs;
- model mutation responses narrowly (`status`, `voice_id`, `requires_verification`, and extra
  fields) instead of exposing raw `serde_json::Value` as the primary API;
- keep PVC workflow methods explicit rather than hiding training, captcha, and verification behind a
  single high-level state machine.

## Closeout Condition

This lane is closed. The accepted voice mutation slices are implemented, focused provider/facade
gates pass, CHANGELOG records user-visible resource additions, and remaining PVC, voice edit, and
sample audio workflow gaps have clear follow-on decisions.
