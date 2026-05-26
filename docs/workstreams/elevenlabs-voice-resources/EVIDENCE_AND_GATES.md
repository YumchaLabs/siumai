# ElevenLabs Voice Resources — Evidence And Gates

Status: Active
Last updated: 2026-05-26

## Source Evidence

| Source | Evidence | Interpretation |
| --- | --- | --- |
| `repo-ref/ai/packages/elevenlabs/src/index.ts` | Exports provider creation, speech/transcription model IDs and options, and `VERSION`; no voice resource client. | Voice resources are a Siumai provider-owned extension, not AI SDK package parity. |
| `https://elevenlabs.io/docs/api-reference/voices/search.mdx` | Documents `GET /v2/voices` with search/filter/pagination query parameters. | Read-only catalog is the first implementation slice. |
| `https://elevenlabs.io/docs/api-reference/voices/get.mdx` | Documents `GET /v1/voices/{voice_id}` for voice metadata. | Detail retrieval is part of the first voice catalog slice. |
| `https://elevenlabs.io/docs/api-reference/voices/ivc/create.mdx` | Documents multipart `POST /v1/voices/add`. | Voice cloning is mutation-heavy and deferred from ELVR-020. |
| `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/list.mdx` | Documents `GET /v1/pronunciation-dictionaries` with pagination. | Dictionary metadata is a candidate follow-on task. |

## Baseline Gates

Use these before closing a task:

```powershell
python .agents\skills\siumai-ai-sdk-maintenance\scripts\resolve_ai_sdk_repo.py
cargo fmt --check -p <touched-package>
git diff --check
```

## Focused Gates By Task

Voice catalog:

```powershell
cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices --no-fail-fast
cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast
cargo fmt --check -p siumai-provider-elevenlabs -p siumai
```

Closeout:

```powershell
python -m json.tool docs\workstreams\elevenlabs-voice-resources\WORKSTREAM.json
git diff --check
```

## Required No-Network Coverage

- `ELEVENLABS_API_KEY` fallback and explicit API-key override continue to work for resources.
- `xi-api-key: <key>` header.
- Base URL override.
- Custom headers and request headers merge behavior.
- `GET /v2/voices` query mapping for `next_page_token`, `page_size`, `search`, `sort`,
  `sort_direction`, `voice_type`, `category`, `fine_tuning_state`, `collection_id`,
  `include_total_count`, and repeated/list `voice_ids`.
- `GET /v1/voices/{voice_id}` URL encoding.
- Voice list response maps `voices`, `has_more`, `total_count`, and `next_page_token`.
- Voice response maps stable top-level fields such as `voice_id`, `name`, `category`, `labels`,
  `description`, `preview_url`, `settings`, `sharing`, `verified_languages`, and preserves unknown
  fields for provider drift.
- Public facade imports compile under `--features elevenlabs`.

## Evidence Log

| Date | Task | Evidence | Result |
| --- | --- | --- | --- |
| 2026-05-26 | ELVR-010 | `python .agents\skills\siumai-ai-sdk-maintenance\scripts\resolve_ai_sdk_repo.py` | Pass: resolved local AI SDK reference at `repo-ref/ai`. |
| 2026-05-26 | ELVR-010 | `repo-ref/ai/packages/elevenlabs/src/{index.ts,elevenlabs-provider.ts,elevenlabs-speech-model.ts}` reviewed with official ElevenLabs voice and pronunciation dictionary API docs. | Done: resource APIs are outside AI SDK package parity and should stay provider-owned. |
