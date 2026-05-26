# ElevenLabs Pronunciation Dictionary Mutations — Evidence And Gates

Status: Active
Last updated: 2026-05-26

## Source Evidence

| Source | Evidence | Interpretation |
| --- | --- | --- |
| `docs/workstreams/elevenlabs-voice-resources` | Closed read-only dictionary metadata resources and split mutation/download follow-ons. | This lane extends the provider-owned resource surface, not unified speech APIs. |
| `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/create-from-rules.mdx` | Documents `POST /v1/pronunciation-dictionaries/add-from-rules` with JSON `rules`, `name`, optional `description`, `workspace_access`, returning `id` and `version_id`. | First slice: stable JSON mutation that completes the TTS locator creation path. |
| `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/create-from-file.mdx` | Documents multipart `POST /v1/pronunciation-dictionaries/add-from-file`. | Second slice candidate after JSON mutation wiring is proven. |
| `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/update.mdx` | Documents `PATCH /v1/pronunciation-dictionaries/{pronunciation_dictionary_id}` for `archived` and `name` without changing version. | Bounded metadata update slice. |
| `https://elevenlabs.io/docs/api-reference/pronunciation-dictionaries/rules/{add,remove,set}.mdx` | Documents JSON rule mutation endpoints returning `id`, `version_id`, and `version_rules_num`. | Candidate shared rule-mutation slice after create-from-rules. |
| `https://elevenlabs.io/docs/llms.txt` | Lists `download.mdx` as "Get pronunciation dictionary by version". | Download is a candidate, but the direct page returned HTTP 500 during opening and needs re-audit. |

## Baseline Gates

Use these before closing a task:

```powershell
cargo fmt --check -p <touched-package>
git diff --check
```

## Focused Gates By Task

Create from rules:

```powershell
cargo nextest run -p siumai-provider-elevenlabs --features elevenlabs pronunciation --no-fail-fast
cargo nextest run -p siumai --features elevenlabs elevenlabs_voice_resources --no-fail-fast
cargo fmt --check -p siumai-provider-elevenlabs -p siumai
```

Closeout:

```powershell
python -m json.tool docs\workstreams\elevenlabs-pronunciation-dictionary-mutations\WORKSTREAM.json
git diff --check
```

## Required No-Network Coverage

- `xi-api-key` header and explicit key override continue through mutation methods.
- Base URL override.
- Custom headers and request headers merge behavior.
- JSON body shape for create-from-rules, metadata update, and accepted rule mutations.
- Multipart body shape for create-from-file if accepted.
- Path encoding for dictionary IDs.
- Response mapping for `id`, `version_id`, `version_rules_num`, dictionary metadata, and unknown
  provider fields.
- Facade imports compile under `--features elevenlabs`.

## Evidence Log

| Date | Task | Evidence | Result |
| --- | --- | --- | --- |
| 2026-05-26 | EPDM-010 | `https://elevenlabs.io/docs/llms.txt` pronunciation dictionary endpoint inventory reviewed. | Done: create-from-rules, create-from-file, update, rules add/remove/set, list/get, and download-by-version were identified. |
| 2026-05-26 | EPDM-010 | Official create/update/rules `.mdx` pages fetched with `Invoke-WebRequest`; `download.mdx` fetch returned HTTP 500 while still listed in `llms.txt`. | Decision: first executable slice is create-from-rules; download requires re-audit before implementation. |
