# Speech And Transcription Helper Parity

Last updated: 2026-04-20

## Goal

Align the public Rust speech/transcription helper semantics more closely with AI SDK
`generateSpeech()` and `transcribe()` without inventing a fake provider-agnostic audio runtime.

Upstream reference:

- `repo-ref/ai/packages/ai/src/generate-speech/generate-speech.ts`
- `repo-ref/ai/packages/ai/src/generate-speech/generate-speech-result.ts`
- `repo-ref/ai/packages/ai/src/transcribe/transcribe.ts`
- `repo-ref/ai/packages/ai/src/transcribe/transcribe-result.ts`
- `repo-ref/ai/packages/ai/src/error/no-speech-generated-error.ts`
- `repo-ref/ai/packages/ai/src/error/no-transcript-generated-error.ts`

## What Changed

- `TtsResponse` and `SttResponse` now carry best-effort final `response` metadata on the stable
  Rust surface, matching the role of AI SDK speech/transcription response metadata more closely.
- `TtsResponse`, `SttResponse`, `SpeechResult`, and `TranscriptionResult` now also carry an
  optional best-effort final `request` envelope, so helper callers can inspect the serialized JSON
  request body on HTTP JSON speech routes without dropping down into ad hoc transport logging.
- The raw stable audio response types now also expose optional `warnings` plus
  `provider_metadata`, which makes the shared Rust provider contract much closer to AI SDK
  `SpeechModelV4Result` / `TranscriptionModelV4Result` instead of forcing every high-level helper
  to invent those slots ad hoc.
- The stable `TtsRequest` surface now also carries AI SDK-auditable speech call-option fields
  directly instead of forcing callers through provider-owned escape hatches for common cases:
  `instructions` and `language` are first-class request fields, and
  `with_output_format(...)` is now available as an AI SDK-style alias for `with_format(...)`.
- The native OpenAI/Azure speech paths now also mirror the audited AI SDK warning semantics more
  closely for shared speech request fields: unsupported `outputFormat` values warn and fall back
  to `mp3`, while `language` now lowers into the JSON wire payload when a provider explicitly opts
  into that field through its `OpenAiAudioDefaults` instead of remaining a silent no-op on the
  shared typed request surface.
- The shared `AudioExecutor` now captures response headers plus model identity for successful TTS
  and STT calls instead of dropping that information below the provider boundary.
- The shared `AudioExecutor` now also captures best-effort serialized JSON request bodies for
  successful HTTP JSON speech/STT calls. Multipart routes intentionally keep `request = None`
  unless a provider-owned path chooses to reconstruct that body itself.
- OpenAI, OpenAI-compatible, Azure, Groq, xAI, and MiniMaxi audio paths now all preserve that
  `request` / `response` envelope when they lower executor results into stable Rust response
  structs.
- The public facade now returns high-level helper result objects instead of exposing only the raw
  provider responses:
  - `speech::synthesize(...)` now returns `speech::SpeechResult` /
    `speech::GenerateSpeechResult`.
  - `transcription::transcribe(...)` now returns `transcription::TranscriptionResult`.
- `siumai::types` and `prelude::unified` now also expose passive AI SDK result-envelope carriers
  for import/serde parity: `GeneratedAudioFile`, `SpeechResult`,
  `Experimental_SpeechResult`, `TranscriptionResult`, `Experimental_TranscriptionResult`, and
  `TranscriptionSegment`. These mirror the upstream `generate-speech` and `transcribe` result
  object shapes without replacing the existing Rust runtime helper result structs in the
  `speech` / `transcription` modules.
- Those helper results are intentionally dual-purpose:
  - they expose AI SDK-style `audio | segments | warnings | responses | provider_metadata`
  - they also keep compatibility mirrors such as `audio_data`, `format`, `duration`,
    `sample_rate`, `confidence`, `words`, and `metadata`, so existing Rust call sites do not need
    a pointless all-at-once rewrite
- The public helper lane now mirrors AI SDK empty-result semantics more closely:
  - `siumai::speech::synthesize(...)` returns `LlmError::NoSpeechGenerated` when the provider
    succeeds but returns empty audio bytes.
  - `siumai::transcription::transcribe(...)` returns `LlmError::NoTranscriptGenerated` when the
    provider succeeds but returns blank transcript text.
- Both specialized errors now carry the final response metadata array, so helper callers can debug
  provider-specific filtered/empty outputs without falling back to generic parse errors.

## Architectural Decision

The response metadata capture lives in the shared executor, not in ad hoc provider wrappers.

Reasoning:

- AI SDK treats empty-result speech/transcription failures as helper-level semantics, but those
  errors still carry provider response metadata.
- If Rust only added `response: None` at each provider boundary, the surface would compile but the
  higher-level parity would still be false because the specialized error types would lose the final
  response envelope.
- AI SDK result contracts also reserve a `request` slot, but the upstream OpenAI implementations do
  not populate it uniformly: speech sets `request.body`, transcription currently does not on the
  multipart path.
- Putting the shared metadata capture into `AudioExecutor` keeps the contract honest, preserves the
  JSON request body where it actually exists, and avoids repeating header/body bookkeeping logic
  across every provider.

## Compatibility Notes

- This change does not invent a second parallel audio helper API; it upgrades the current
  Rust-first `speech::synthesize(...)` and `transcription::transcribe(...)` facades to the richer
  high-level result shape while the lower-level raw provider responses remain available through the
  family traits (`SpeechModel::synthesize` / `TranscriptionModel::transcribe`) and
  `into_tts_response()` / `into_stt_response()` conversions on the helper results.
- Provider-owned audio option helpers remain as compatibility escape hatches, but common speech
  request fields no longer need to be tunneled through `providerOptions` just to reach the shared
  request contract.
- Providers that do not use the shared audio executor can still populate `response` manually, but
  the current audited providers now inherit the stable behavior from the executor path directly.
- The new `request` slot is intentionally best-effort. JSON routes can preserve the final
  stringified body directly; multipart routes generally cannot without rebuilding provider-specific
  form-data serialization details, so `None` is the truthful default there.
- Audio translation still uses the same stable `SttResponse` shape, so provider-owned translation
  paths can preserve the same response metadata contract when available.

## Remaining Follow-up

- The result-side metadata contract is now close to AI SDK parity, including the optional
  `request` slot on the stable/helper response types.
- The shared transcription request shape now also matches the AI SDK boundary more honestly:
  `SttRequest` no longer keeps top-level `language` or `timestamp_granularities`; those knobs now
  live where upstream puts them, under provider-owned `providerOptions` (for example
  `OpenAiSttOptions` / `GroqSttOptions`).
- If Siumai later introduces richer batched speech/transcription helpers, the stable response shape
  may need the same multi-call metadata discussion that already exists for image/video helpers.
