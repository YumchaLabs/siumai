# ElevenLabs Pronunciation Dictionary Mutations

Status: Active
Last updated: 2026-05-26

## Why This Lane Exists

The closed `elevenlabs-voice-resources` lane added read-only pronunciation dictionary metadata under
`provider_ext::elevenlabs::resources::*`. That gives users dictionary IDs and latest version IDs,
but it does not let applications create or update dictionaries before attaching them to TTS requests
with `ElevenLabsPronunciationDictionaryLocator`.

This lane covers provider-owned mutation/download APIs for pronunciation dictionaries while keeping
the unified speech/transcription families unchanged.

## Relevant Authority

- Closed predecessor lane:
  - `docs/workstreams/elevenlabs-voice-resources`
- Official ElevenLabs docs:
  - `https://elevenlabs.io/docs/llms.txt`
  - `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/create-from-rules.mdx`
  - `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/create-from-file.mdx`
  - `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/update.mdx`
  - `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/rules/add.mdx`
  - `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/rules/remove.mdx`
  - `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/rules/set.mdx`
  - `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/download.mdx`
- Existing Siumai APIs:
  - `siumai-provider-elevenlabs/src/providers/elevenlabs/pronunciation_dictionaries.rs`
  - `siumai-provider-elevenlabs/src/providers/elevenlabs/options.rs`
  - `siumai-provider-elevenlabs/src/providers/elevenlabs/resource_http.rs`
- Local public-surface policy:
  - `docs/adr/0003-provider-ext-export-policy.md`

## API Snapshot

| Area | Official API shape | Lane decision |
| --- | --- | --- |
| Create from rules | `POST /v1/pronunciation-dictionaries/add-from-rules`, JSON `rules`, `name`, optional `description`, `workspace_access`; returns `id` and `version_id`. | First implementation slice. It is stable JSON and completes the TTS locator creation path. |
| Create from file | `POST /v1/pronunciation-dictionaries/add-from-file`, multipart `name`, `file`, optional `description`, `workspace_access`; returns `id` and `version_id`. | Second slice after JSON creation proves shared mutation wiring. |
| Update metadata | `PATCH /v1/pronunciation-dictionaries/{id}`, JSON `archived` and/or `name`; returns dictionary metadata without changing version. | Bounded follow-on slice. |
| Rule mutation | `POST /{id}/add-rules`, `remove-rules`, `set-rules`; returns `id`, `version_id`, `version_rules_num`. | Bounded follow-on after create requests share rule request structs. |
| Download by version | Docs index lists `download.mdx`, but direct page fetch returned HTTP 500 during opening. | Audit before implementation; do not block the first mutation slice on this endpoint. |

## Target State

When this workstream closes:

- `ElevenLabsPronunciationDictionaries` owns mutation methods in the provider crate instead of
  creating another facade family.
- Public resource exports remain under `siumai::provider_ext::elevenlabs::resources::*` and
  `siumai::providers::elevenlabs::resources::*`.
- Create-from-rules is implemented with typed request/rule structs and no-network JSON tests.
- Create-from-file, metadata update, rule mutation, and download-by-version are either implemented
  in bounded slices or explicitly split/deferred with evidence.
- Returned `id`/`version_id` fields are easy to feed into existing
  `ElevenLabsPronunciationDictionaryLocator`.
- `prelude::unified` and speech/transcription traits remain unchanged.

## In Scope

- Provider-owned pronunciation dictionary mutation/download methods.
- Typed request/response structs for documented JSON/multipart/binary endpoints.
- No-network tests proving auth headers, base URL override, request header merge, path encoding,
  JSON/multipart body shape, binary download handling if implemented, response mapping, and unknown
  metadata preservation.
- Facade public-surface compile tests.
- CHANGELOG and workstream closeout.

## Out Of Scope

- Live ElevenLabs credential tests as required gates.
- Changing TTS request option names or unified speech APIs.
- Generic cross-provider pronunciation dictionary traits.
- Voice clone/update/delete/settings/sample/PVC APIs; those belong to a separate voice mutation
  workstream.

## Architecture Direction

Reuse the closed voice resources design:

- keep methods on `ElevenLabsPronunciationDictionaries`;
- reuse `resource_http` for auth/header/base URL/transport/interceptor/retry wiring, extending it
  for JSON, multipart, PATCH, POST, and binary GET as needed;
- keep request builders typed and endpoint-specific rather than exposing raw `serde_json::Value`
  maps as the main API;
- share alias/phoneme rule request/response structs where add/set/create endpoints use the same
  schema;
- preserve unknown provider fields with `#[serde(flatten)]` on response structs.

## Closeout Condition

This lane can close when the accepted mutation/download slices are implemented or explicitly split,
focused provider/facade gates pass, CHANGELOG records user-visible resource additions, and remaining
download or mutation endpoints have a clear follow-on decision.
