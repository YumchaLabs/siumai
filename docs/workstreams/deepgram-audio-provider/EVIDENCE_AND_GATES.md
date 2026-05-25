# Deepgram Audio Provider — Evidence And Gates

Status: Draft
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
