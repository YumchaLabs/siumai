# ElevenLabs Audio Provider — Evidence And Gates

Status: Active
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

Interpretation: ElevenLabs is the second audio-provider follow-on because it combines strong package
signal with a speech/transcription implementation boundary that does not require queued media task
semantics.

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
cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs --no-fail-fast
cargo fmt --check -p siumai-provider-elevenlabs
```

Registry:

```powershell
cargo nextest run -p siumai-registry --features elevenlabs elevenlabs --no-fail-fast
cargo fmt --check -p siumai-registry
```

Facade:

```powershell
cargo nextest run -p siumai --features elevenlabs elevenlabs --no-fail-fast
cargo fmt --check -p siumai
```

Closeout:

```powershell
python -m json.tool docs\workstreams\elevenlabs-audio-provider\WORKSTREAM.json
git diff --check
```

## Required No-Network Coverage

- `ELEVENLABS_API_KEY` fallback and explicit API-key override.
- `xi-api-key: <key>` header.
- Base URL override.
- Custom headers and request headers merge behavior.
- Speech URL `/v1/text-to-speech/{voiceId}` with default voice id and explicit voice id behavior.
- Speech query mapping for AI SDK output formats and `enable_logging`.
- Speech JSON body maps `text`, `model_id`, `language_code`, `voice_settings`, pronunciation dictionary locators, seed, previous/next text, previous/next request ids, and text normalization options.
- Speech warnings cover unsupported shared controls such as `instructions`.
- Speech binary response maps into `TtsResponse`/high-level speech result fields.
- Transcription URL `/v1/speech-to-text`.
- Transcription request body uses multipart form data with `model_id`, `file`, `diarize`, language, audio-event tags, speaker count, timestamp granularity, and file format fields.
- Transcription response maps transcript text, word segments, language, duration, response headers, and provider metadata.
- Unsupported language/chat/completion/embedding/image/video/rerank families fail before transport use.
- Public facade imports compile under `--features elevenlabs`.

## Evidence Log

| Date | Task | Evidence | Result |
| --- | --- | --- | --- |
| 2026-05-26 | ELA-010 | `python .agents\skills\siumai-ai-sdk-maintenance\scripts\resolve_ai_sdk_repo.py` | Pass: resolved local AI SDK reference at `repo-ref/ai`. |
| 2026-05-26 | ELA-010 | `repo-ref/ai/packages/elevenlabs/src/{index.ts,elevenlabs-provider.ts,elevenlabs-speech-model.ts,elevenlabs-transcription-model.ts,*options.ts,*api-types.ts}` reviewed against Siumai speech/transcription family surfaces and the closed Deepgram provider lane. | Done: implementation boundary is native ElevenLabs speech plus transcription only. |
| 2026-05-26 | ELA-010 | `python -m json.tool docs\workstreams\elevenlabs-audio-provider\WORKSTREAM.json`; `git diff --check` | Pass: WORKSTREAM.json parsed successfully; no whitespace errors. |
