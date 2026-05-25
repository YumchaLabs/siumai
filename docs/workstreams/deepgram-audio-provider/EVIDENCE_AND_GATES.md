# Deepgram Audio Provider — Evidence And Gates

Status: Closed
Last updated: 2026-05-26

## Market Evidence

Source: npm downloads API, `last-month`, inherited from
`docs/workstreams/ai-sdk-provider-market-expansion/EVIDENCE_AND_GATES.md`.

Window returned by npm: `2026-04-25` through `2026-05-24`.

| Package | Downloads |
| --- | ---: |
| `@ai-sdk/deepgram` | 605,045 |
| `@ai-sdk/elevenlabs` | 533,132 |
| `@ai-sdk/replicate` | 257,948 |
| `@ai-sdk/fal` | 192,339 |

Interpretation: Deepgram is the first audio/media follow-on because it combines the strongest package signal in
this group with a narrow speech/transcription implementation boundary.

## Baseline Gates

Use these before closing a task:

```powershell
python .agents\skills\siumai-ai-sdk-maintenance\scripts\resolve_ai_sdk_repo.py
cargo fmt --check -p <touched-package>
git diff --check
```

## Focused Gates By Task

Provider crate:

```powershell
cargo nextest run -p siumai-provider-deepgram --features deepgram --no-fail-fast
cargo fmt --check -p siumai-provider-deepgram
```

Registry:

```powershell
cargo nextest run -p siumai-registry --features deepgram deepgram --no-fail-fast
cargo fmt --check -p siumai-registry
```

Facade:

```powershell
cargo nextest run -p siumai --features deepgram deepgram --no-fail-fast
cargo fmt --check -p siumai
```

Closeout:

```powershell
python -m json.tool docs\workstreams\deepgram-audio-provider\WORKSTREAM.json
git diff --check
```

## Required No-Network Coverage

- `DEEPGRAM_API_KEY` fallback and explicit API-key override.
- `authorization: Token <key>` header.
- Base URL override.
- Custom headers and request headers merge behavior.
- Speech URL `/v1/speak` with `model` and provider option query params.
- Speech JSON body `{ "text": ... }`.
- Speech binary response mapped into `TtsResponse`/high-level speech result fields.
- Transcription URL `/v1/listen` with model and provider option query params.
- Transcription request body uses raw audio and `Content-Type` from input media type.
- Transcription response maps transcript text, words/segments, language, duration, response headers, and provider metadata.
- Unsupported language/chat/completion/embedding/image/video/rerank families fail before transport use.
- Public facade imports compile under `--features deepgram`.

## Evidence Log

| Date | Task | Evidence | Result |
| --- | --- | --- | --- |
| 2026-05-26 | DGA-010 | `python .agents\skills\siumai-ai-sdk-maintenance\scripts\resolve_ai_sdk_repo.py` | Pass: resolved local AI SDK reference at `repo-ref/ai`. |
| 2026-05-26 | DGA-010 | `repo-ref/ai/packages/deepgram/src/{index.ts,deepgram-provider.ts,deepgram-speech-model.ts,deepgram-transcription-model.ts,*options.ts}` reviewed against Siumai speech/transcription family surfaces. | Done: implementation boundary is native Deepgram speech plus transcription only. |
| 2026-05-26 | DGA-010 | `python -m json.tool docs\workstreams\deepgram-audio-provider\WORKSTREAM.json`; `git diff --check` | Pass: WORKSTREAM.json parsed successfully; no whitespace errors. |
| 2026-05-26 | DGA-020 | `cargo nextest run -p siumai-provider-deepgram --features deepgram --no-fail-fast` | Pass: 10 tests covered typed options, request option merging, env key constant, explicit key override, header merging, runtime helpers, `/v1/speak` JSON request/auth/query behavior, `/v1/listen` raw audio request/auth/query behavior, and transcription response mapping. |
| 2026-05-26 | DGA-020 | `cargo fmt --check --package siumai-provider-deepgram` | Pass: provider crate formatting check completed. |
| 2026-05-26 | DGA-020 | `git diff --check` | Pass: no whitespace errors; Git reported expected LF-to-CRLF working-copy warnings for touched root Cargo files. |
| 2026-05-26 | DGA-030 | `cargo nextest run -p siumai-registry --features deepgram deepgram --no-fail-fast` | Pass: 11 focused tests covered Deepgram native metadata/catalog, builder selector, speech/transcription factory paths, provider-specific registry build overrides, `/v1/speak` JSON request/auth/query behavior, `/v1/listen` raw audio request/auth/query behavior, unsupported non-audio handle rejection before transport use, and `SiumaiBuilder` STT model routing. |
| 2026-05-26 | DGA-030 | `cargo fmt --check -p siumai-registry` | Pass: registry crate formatting check completed. |
| 2026-05-26 | DGA-040 | `cargo nextest run -p siumai --features deepgram deepgram --no-fail-fast` | Pass: 2 focused facade tests covered `provider_ext::deepgram`, `providers::deepgram`, `create_deepgram()`, `Provider::deepgram()`, `DeepgramClient`/`DeepgramConfig`, model constant re-exports, and typed speech/transcription request options. |
| 2026-05-26 | DGA-040 | `cargo fmt --check -p siumai` | Pass: facade crate formatting check completed after `cargo fmt -p siumai`. |
| 2026-05-26 | DGA-040 | `cargo check -p siumai --tests --no-default-features --features deepgram` | Not used as a DGA-040 gate: facade library compiles far enough to build the Deepgram feature, but an existing OpenAI tooling public-surface test references `siumai::tools::openai` without enabling the `openai`/`protocol-openai` feature. |
| 2026-05-26 | DGA-040 | `python -m json.tool docs\workstreams\deepgram-audio-provider\WORKSTREAM.json`; `git diff --check` | Pass: WORKSTREAM.json parsed successfully; no whitespace errors. |
| 2026-05-26 | DGA-050 | `cargo nextest run -p siumai-provider-deepgram --features deepgram --no-fail-fast`; `cargo fmt --check -p siumai-provider-deepgram` | Pass: 10 provider-crate tests passed; formatting check passed. |
| 2026-05-26 | DGA-050 | `cargo nextest run -p siumai-registry --features deepgram deepgram --no-fail-fast`; `cargo fmt --check -p siumai-registry` | Pass: 11 registry focused tests passed; formatting check passed. The registry test build emitted existing unused-code/import warnings in `contract_tests.rs`. |
| 2026-05-26 | DGA-050 | `cargo nextest run -p siumai --features deepgram deepgram --no-fail-fast`; `cargo fmt --check -p siumai` | Pass: 2 facade public-surface tests passed; formatting check passed. |
| 2026-05-26 | DGA-050 | `python -m json.tool docs\workstreams\deepgram-audio-provider\WORKSTREAM.json`; `git diff --check` | Pass: WORKSTREAM.json parsed successfully; no whitespace errors. |
