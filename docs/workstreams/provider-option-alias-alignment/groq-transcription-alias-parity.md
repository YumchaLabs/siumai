# Groq Transcription Alias Parity

Last updated: 2026-04-14

## Reference

- `repo-ref/ai/packages/groq/src/index.ts`
- `repo-ref/ai/packages/groq/src/groq-transcription-options.ts`
- `repo-ref/ai/packages/groq/src/groq-transcription-model.ts`

## What changed

- `siumai-provider-groq` now exposes `GroqTranscriptionModelOptions` as the audited AI SDK-style
  alias for Groq transcription-specific provider options.
- `siumai::provider_ext::groq::options::*` and the top-level
  `siumai::provider_ext::groq::*` facade now re-export the same alias.
- The concrete `GroqSttOptions` / `GroqTtsOptions` helpers remain available only under
  `siumai::provider_ext::groq::ext::audio_options::*`, keeping provider-owned audio escape hatches
  separate from the AI SDK-aligned `options::*` lane.
- `GroqSttOptions` now covers the same main option subset that AI SDK models today:
  - `language`
  - `prompt`
  - `responseFormat`
  - `temperature`
  - `timestampGranularities`

## Why this was not just a rename

The initial alias-only pass intentionally deferred Groq transcription because the Rust runtime
surface was not ready for an honest AI SDK-shaped alias:

- the typed helper lacked `language` and `timestampGranularities`
- Groq audio defaults disabled those fields during multipart shaping
- Groq STT responses discarded `language`, `duration`, and raw `segments`

This pass closes those runtime gaps first, then exposes the alias.

## Runtime alignment details

- `GroqSttOptions::to_json()` now emits AI SDK-style keys (`responseFormat`,
  `timestampGranularities`) while preserving Groq-specific `stream` as an escape hatch.
- Groq audio request shaping now forwards:
  - `language`
  - `timestamp_granularities[]`
  - existing typed/raw provider options such as `prompt`, `temperature`, and `response_format`
- Groq STT responses now preserve:
  - `language`
  - `duration`
  - `segments`
  - `x_groq`
  - `task`

## Validation

- `cargo check -p siumai-provider-groq --features groq`
- `cargo check -p siumai --features groq`
- `cargo check -p siumai-provider-groq -p siumai --all-features`
- `cargo nextest run -p siumai-provider-groq --features groq stt_options_serialize_ai_sdk_style_keys groq_client_speech_to_text_accepts_typed_transcription_options_and_preserves_metadata`
- `cargo nextest run -p siumai --features groq public_surface_groq_provider_ext_compiles`

## Remaining gap outside this note

- Groq still keeps provider-owned TTS support as a Rust-only extension because the audited
  `@ai-sdk/groq` package only exposes transcription on its audio surface.
