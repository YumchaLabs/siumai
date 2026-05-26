# ElevenLabs Voice Resources — Evidence And Gates

Status: Closed
Last updated: 2026-05-26

## Source Evidence

| Source | Evidence | Interpretation |
| --- | --- | --- |
| `repo-ref/ai/packages/elevenlabs/src/index.ts` | Exports provider creation, speech/transcription model IDs and options, and `VERSION`; no voice resource client. | Voice resources are a Siumai provider-owned extension, not AI SDK package parity. |
| `https://elevenlabs.io/docs/api-reference/voices/search.mdx` | Documents `GET /v2/voices` with search/filter/pagination query parameters. | Read-only catalog is the first implementation slice. |
| `https://elevenlabs.io/docs/api-reference/voices/get.mdx` | Documents `GET /v1/voices/{voice_id}` for voice metadata. | Detail retrieval is part of the first voice catalog slice. |
| `https://elevenlabs.io/docs/api-reference/voices/ivc/create.mdx` | Documents multipart `POST /v1/voices/add`. | Voice cloning is mutation-heavy and deferred from ELVR-020. |
| `https://elevenlabs.io/docs/llms.txt` | Lists voice library, PVC, IVC, samples, settings, edit, delete, and similar-voices endpoints. | Voice mutation/samples/PVC form a separate resource family and should split from this read-only lane. |
| `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/list.mdx` | Documents `GET /v1/pronunciation-dictionaries` with pagination. | Dictionary metadata is a candidate follow-on task. |
| `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/get.mdx` | Documents `GET /v1/pronunciation-dictionaries/{pronunciation_dictionary_id}` returning metadata and latest-version rules. | Read-only dictionary detail is narrow enough for ELVR-030; mutation/download remains out of scope. |

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

Pronunciation dictionary metadata:

```powershell
cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs pronunciation --no-fail-fast
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
| 2026-05-26 | ELVR-020 | `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices --no-fail-fast` | Pass: 3 voice resource tests cover list/get request construction, auth/header/base URL reuse, explicit key override, query mapping, URL encoding, response field mapping, and unknown metadata preservation. |
| 2026-05-26 | ELVR-020 | `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast` | Pass: facade resource imports compile through `siumai::provider_ext::elevenlabs::resources` and alias `siumai::providers::elevenlabs::resources`. |
| 2026-05-26 | ELVR-020 | `cargo fmt --check -p siumai-provider-elevenlabs -p siumai` | Pass: touched packages are formatted. |
| 2026-05-26 | ELVR-020 | `python -m json.tool docs\workstreams\elevenlabs-voice-resources\WORKSTREAM.json` | Pass: workstream metadata remains valid JSON. |
| 2026-05-26 | ELVR-020 | `git diff --check` | Pass: no whitespace errors in the working diff. |
| 2026-05-26 | ELVR-030 | `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/{list,get}.mdx` reviewed. | Decision: implement only read-only list/get metadata in this lane; split create/update/rule mutation and PLS download. |
| 2026-05-26 | ELVR-030 | `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs pronunciation --no-fail-fast` | Pass: 2 tests cover dictionary list/get endpoints, query mapping, auth/header/base URL reuse, URL encoding, metadata/rule mapping, and unknown metadata preservation. |
| 2026-05-26 | ELVR-030 | `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices --no-fail-fast` | Pass: 3 tests confirm the shared resource HTTP refactor preserved voice catalog behavior. |
| 2026-05-26 | ELVR-030 | `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast` | Pass: facade resource imports compile for voice and pronunciation dictionary resources. |
| 2026-05-26 | ELVR-030 | `cargo fmt --check -p siumai-provider-elevenlabs -p siumai` | Pass: touched packages are formatted. |
| 2026-05-26 | ELVR-030 | `cargo check -p siumai --lib --no-default-features --features elevenlabs` | Pass: facade compiles with only the ElevenLabs provider feature enabled; existing warnings remain in `siumai-bridge` legacy content helpers and `siumai::tools`. |
| 2026-05-26 | ELVR-030 | `python -m json.tool docs\workstreams\elevenlabs-voice-resources\WORKSTREAM.json` | Pass: workstream metadata remains valid JSON after moving current task to ELVR-040. |
| 2026-05-26 | ELVR-030 | `git diff --check` | Pass: no whitespace errors in the working diff. |
| 2026-05-26 | ELVR-040 | `https://elevenlabs.io/docs/llms.txt` voice endpoint inventory reviewed. | Decision: split voice clone/update/delete/settings/sample/PVC APIs into mutation-focused follow-ons; this lane stays read-only and ready for closeout. |
| 2026-05-26 | ELVR-050 | `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs voices --no-fail-fast` | Pass: final closeout voice resource gate, 3 tests passed. |
| 2026-05-26 | ELVR-050 | `cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs pronunciation --no-fail-fast` | Pass: final closeout pronunciation dictionary resource gate, 2 tests passed. |
| 2026-05-26 | ELVR-050 | `cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast` | Pass: final closeout facade public-surface gate, 1 test passed. |
| 2026-05-26 | ELVR-050 | `cargo fmt --check -p siumai-provider-elevenlabs -p siumai` | Pass: final closeout formatting gate. |
| 2026-05-26 | ELVR-050 | `cargo check -p siumai --lib --no-default-features --features elevenlabs` | Pass: final closeout facade compile gate with only ElevenLabs enabled; existing warnings unchanged. |
| 2026-05-26 | ELVR-050 | `python -m json.tool docs\workstreams\elevenlabs-voice-resources\WORKSTREAM.json` | Pass: closeout metadata is valid JSON. |
| 2026-05-26 | ELVR-050 | `git diff --check` | Pass: closeout diff has no whitespace errors. |
