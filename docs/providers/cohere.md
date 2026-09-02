# Cohere Provider Support

- Provider identity: `cohere`
- Technical platform: `public-api`
- Portable families: Embedding and Rerank
- Provider-native surface: model-less v2 audio transcription
- Primary API mode: native v2
- Embedding evidence verified: 2026-08-14
- Audio transcription evidence verified: 2026-08-15
- Provider crate: `siumai-provider-cohere`

Known model constants and the provider profile catalog are dated ergonomic hints, not execution
allowlists. Private, proxied, and future model identifiers remain valid when their requests can be
encoded by the v2 wire contract.

## Embedding output dimensions

Cohere's v2 Embed reference documents `output_dimension` as an optional integer with the exact
values `256`, `512`, `1024`, and `1536`. It also describes the current product availability as
Embed v4 and newer. Siumai treats the four-value domain as a stable wire invariant while leaving
model availability to Cohere and the host application.

The provider therefore:

- encodes an explicit valid dimension for known, private, or future model identifiers;
- rejects values outside the documented four-value domain before transport;
- rejects a conflict between portable `EmbeddingRequest::dimensions` and typed Cohere options;
- verifies that every returned float vector matches the requested dimension.

This boundary deliberately does not infer eligibility from a model-name prefix or copy Cohere's
current catalog into runtime validation. Cohere remains authoritative and may reject a valid wire
value for a model or account that does not offer the product capability.

## Evidence

| Exact slice | Official source | Verified |
|---|---|---|
| v2 Embed request, `output_dimension` values, and current product guidance | [Cohere Embed v2 reference](https://docs.cohere.com/v2/reference/embed) | 2026-08-14 |
| v2 audio transcription multipart request and text response | [Cohere Create Transcription](https://docs.cohere.com/reference/create-audio-transcription) | 2026-08-15 |

## Audio transcription

`provider.transcriptions()` exposes Cohere's synchronous `POST /v2/audio/transcriptions`
operation as a provider-owned resource rather than inventing a model identifier that the upstream
request does not have. `CohereTranscriptionRequest` requires an explicit open language value,
accepts the documented MP3, WAV, FLAC, AAC, M4A, and OGG media families, supports the documented
`0..=1` temperature range, and limits the complete audio input to 25 MB before multipart
encoding. Submission is never automatically replayed because a successful request may be
billable even when its response is lost.

The response is returned through Siumai's standard `TranscriptionResponse`; Cohere currently
provides only the final text, so language, duration, confidence, segments, and usage remain absent
or unknown instead of being synthesized from the request.

Deterministic offline fixtures cover future-model dimension encoding, invalid values, canonical
versus typed conflicts, response vector-length mismatches, and the exact model-less transcription
multipart contract. They do not perform live, credentialed, or billable calls.
