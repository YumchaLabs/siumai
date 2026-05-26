# ElevenLabs Audio Provider — Milestones

Status: Active
Last updated: 2026-05-26

## M0 — Scope Freeze

Exit criteria:

- ElevenLabs' AI SDK package surface is inventoried.
- The lane explicitly excludes voice resource APIs, live credential gates, Fal, Replicate, queued media polling, and `prelude::unified` widening.
- The first implementation boundary is speech plus transcription only.

Gate:

- Documentation consistency across DESIGN.md, TODO.md, WORKSTREAM.json, HANDOFF.md, and EVIDENCE_AND_GATES.md.

## M1 — Provider Crate

Exit criteria:

- `siumai-provider-elevenlabs` exists and owns ElevenLabs config, auth, model ids, options, error mapping, and client behavior.
- Speech request tests cover `/v1/text-to-speech/{voiceId}`, `xi-api-key`, text JSON body, output-format query mapping, `enable_logging`, voice settings, and binary response handling.
- Transcription request tests cover `/v1/speech-to-text`, multipart form fields, media type/file extension behavior, typed transcription options, text/segment/language/duration extraction, and response metadata.

Gate:

- `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs --no-fail-fast`
- `cargo fmt --check -p siumai-provider-elevenlabs`

## M2 — Registry And Metadata

Exit criteria:

- ElevenLabs is feature-gated in workspace, registry, and facade crates.
- Native provider metadata reports speech/transcription/audio capabilities.
- Registry handles build ElevenLabs speech and transcription family models and reject unsupported families before transport use.

Gate:

- `cargo nextest run -p siumai-registry --features elevenlabs elevenlabs --no-fail-fast`
- `cargo fmt --check -p siumai-registry`

## M3 — Public Surface

Exit criteria:

- `siumai::provider_ext::elevenlabs` and `siumai::providers::elevenlabs` expose settings, builders, model constants, and typed options.
- `Provider::elevenlabs()` and `SiumaiBuilder::elevenlabs()` or equivalent Rust-native entry points are covered when they fit existing patterns.
- Public-surface import tests compile with the `elevenlabs` feature.
- `prelude::unified` is not widened.

Gate:

- `cargo nextest run -p siumai --features elevenlabs elevenlabs --no-fail-fast`
- targeted public-surface import test
- `cargo fmt --check -p siumai`

## M4 — Closeout

Exit criteria:

- All task ledger items are done, split, or explicitly deferred.
- Evidence gates are refreshed.
- HANDOFF.md states the next task or lane closure.

Gate:

- `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs --no-fail-fast`
- `cargo nextest run -p siumai-registry --features elevenlabs elevenlabs --no-fail-fast`
- `cargo nextest run -p siumai --features elevenlabs elevenlabs --no-fail-fast`
- `cargo fmt --check -p siumai-provider-elevenlabs`
- `cargo fmt --check -p siumai-registry`
- `cargo fmt --check -p siumai`
- `python -m json.tool docs\workstreams\elevenlabs-audio-provider\WORKSTREAM.json`
- `git diff --check`
