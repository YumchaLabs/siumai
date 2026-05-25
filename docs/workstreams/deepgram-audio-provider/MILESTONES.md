# Deepgram Audio Provider — Milestones

Status: Draft
Last updated: 2026-05-26

## M0 — Scope Freeze

Exit criteria:

- Deepgram's AI SDK package surface is inventoried.
- The lane explicitly excludes ElevenLabs, Fal, Replicate, and live credential gates.
- The first implementation boundary is speech plus transcription only.

Gate:

- Documentation consistency across DESIGN.md, TODO.md, WORKSTREAM.json, HANDOFF.md, and EVIDENCE_AND_GATES.md.

## M1 — Provider Crate

Exit criteria:

- `siumai-provider-deepgram` exists and owns Deepgram config, auth, model ids, options, error mapping, and client behavior.
- Speech request tests cover `/v1/speak`, `authorization: Token ...`, text JSON body, output-format query mapping, and binary response handling.
- Transcription request tests cover `/v1/listen`, media-type header, raw audio body, query options, transcript segments, language, and duration extraction.

Gate:

- `cargo nextest run -p siumai-provider-deepgram --features deepgram --no-fail-fast`
- `cargo fmt --check -p siumai-provider-deepgram`

## M2 — Registry And Metadata

Exit criteria:

- Deepgram is feature-gated in workspace, registry, and facade crates.
- Native provider metadata reports speech/transcription/audio capabilities.
- Registry handles build Deepgram speech and transcription family models and reject unsupported families before transport use.

Gate:

- `cargo nextest run -p siumai-registry --features deepgram deepgram --no-fail-fast`
- `cargo fmt --check -p siumai-registry`

## M3 — Public Surface

Exit criteria:

- `siumai::provider_ext::deepgram` and `siumai::providers::deepgram` expose settings, builders, model constants, and typed options.
- `Provider::deepgram()` and `SiumaiBuilder::deepgram()` or equivalent Rust-native entry points are covered when they fit existing patterns.
- Public-surface import tests compile with the `deepgram` feature.
- `prelude::unified` is not widened.

Gate:

- `cargo nextest run -p siumai --features deepgram deepgram --no-fail-fast`
- targeted public-surface import test
- `cargo fmt --check -p siumai`

## M4 — Closeout

Exit criteria:

- All task ledger items are done, split, or explicitly deferred.
- Evidence gates are refreshed.
- HANDOFF.md states the next task or lane closure.
